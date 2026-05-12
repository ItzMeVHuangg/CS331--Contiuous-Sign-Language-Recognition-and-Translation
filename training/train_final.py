# -*- coding: utf-8 -*-
"""
final_system/train_final.py
----------------------------
Unified CSLR ? SLT training pipeline for thesis report.

Supported pipelines:
    --encoder resnet34  ?  ResNet34-2D + BiLSTM (Variant A) ? SLT (Variant Y)
    --encoder swin      ?  Video Swin-T + BiLSTM (Variant G) ? SLT (Variant Y)

Usage:
    # Full pipeline (CSLR then SLT automatically):
    python final_system/train_final.py --config final_system/config_resnet34.yaml --encoder resnet34

    # CSLR only (skip SLT):
    python final_system/train_final.py --config final_system/config_resnet34.yaml --encoder resnet34 --stage cslr

    # SLT only (provide existing CSLR checkpoint):
    python final_system/train_final.py --config final_system/config_resnet34.yaml --encoder resnet34 --stage slt --cslr_ckpt path/to/best_cslr.pth

    # Resume CSLR from latest checkpoint (auto-detected):
    python final_system/train_final.py --config final_system/config_resnet34.yaml --encoder resnet34 --stage all
"""

import sys
import gc
import json
import random
import argparse
import math
import time
import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from tqdm import tqdm

# -- Add project root to path --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset       import build_vocabularies, PhoenixDataset, collate_fn, build_transforms
from models.bilstm_ctc  import BiLSTM_CTC, CTCCriterion
from utils.ctc_decoder  import batch_ctc_decode
from utils.metrics      import compute_wer, compute_bleu, compute_rouge, compute_meteor

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def _worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


# ------------------------------------------------------------------------------
# Model builders
# ------------------------------------------------------------------------------

def build_encoder(cfg, encoder_type: str) -> nn.Module:
    if encoder_type == "resnet34":
        from models.cnn_encoder import CNNEncoder
        c = cfg["cnn"]
        return CNNEncoder(
            backbone     = "resnet34",
            pretrained   = c["pretrained"],
            out_features = c["out_features"],
            freeze_bn    = c["freeze_bn"],
        )
    elif encoder_type == "swin":
        from models.video_swin_encoder import VideoSwinEncoder
        c = cfg["video_swin"]
        return VideoSwinEncoder(
            backbone          = c.get("backbone", "swin3d_t"),
            pretrained        = c.get("pretrained", True),
            out_features      = c.get("out_features", 512),
            clip_len          = c.get("clip_len", 16),
            clip_stride       = c.get("clip_stride", 8),
            max_clips_per_fwd = c.get("max_clips_per_fwd", 4),
        )
    else:
        raise ValueError(f"Unknown encoder: '{encoder_type}'")


def get_encoder_feat_dim(cfg, encoder_type: str) -> int:
    if encoder_type == "resnet34":
        return cfg["cnn"]["out_features"]
    elif encoder_type == "swin":
        return cfg["video_swin"]["out_features"]
    raise ValueError(encoder_type)


def build_cslr_model(cfg, num_classes: int, encoder_type: str) -> nn.Module:
    from models.temporal_pool import TemporalPool

    encoder  = build_encoder(cfg, encoder_type)
    feat_dim = get_encoder_feat_dim(cfg, encoder_type)
    c_bilstm = cfg["bilstm"]

    seq_model = BiLSTM_CTC(
        input_size      = feat_dim,
        hidden_size     = c_bilstm["hidden_size"],
        num_layers      = c_bilstm["num_layers"],
        num_classes     = num_classes,
        dropout         = c_bilstm["dropout"],
        projection_size = c_bilstm["projection_size"],
        blank_idx       = cfg["cslr"]["ctc_blank_idx"],
    )

    is_temporal = (encoder_type == "swin")
    use_pool    = (encoder_type == "resnet34")
    temp_pool   = TemporalPool(num_pool_layers=2) if use_pool else None

    class CSLRModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = encoder
            self.seq_model = seq_model
            self.is_temporal = is_temporal
            if temp_pool is not None:
                self.temp_pool = temp_pool

        def forward(self, frames, frame_lens):
            feats = self.encoder(frames)                    # (B, T/T', D)

            T_in  = frames.shape[1]
            T_out = feats.shape[1]

            if hasattr(self, "temp_pool"):
                feats = self.temp_pool(feats)
                lens  = self.temp_pool.adjust_lengths(frame_lens, T_in)
            elif T_out != T_in:
                # Swin or any encoder that changes temporal dim
                ratio = T_out / T_in
                lens  = (frame_lens.float() * ratio).long().clamp(min=1, max=T_out)
            else:
                lens = frame_lens.clamp(max=T_out)

            log_probs, hidden = self.seq_model(feats, lens)
            return log_probs, hidden, lens

    return CSLRModel()


def build_slt_model(cfg, gloss_vocab_size: int, text_vocab_size: int,
                    cslr_model: nn.Module) -> nn.Module:
    from models.late_fusion import LateFusion, GlossEmbedding
    from models.translator  import SLTTransformer

    seq_model = cslr_model.seq_model
    proj_dim  = seq_model.hidden_out_dim

    fc          = cfg["fusion"]
    gloss_embed = GlossEmbedding(gloss_vocab_size, fc["gloss_embed_dim"])
    late_fusion = LateFusion(
        visual_dim      = proj_dim,
        gloss_embed_dim = fc["gloss_embed_dim"],
        fused_dim       = fc["fused_dim"],
        mode            = fc["mode"],
        dropout         = fc["dropout"],
        nhead           = fc.get("nhead", 8),
    )
    fused_dim = fc["fused_dim"]

    c = cfg["transformer_2d"]
    translator = SLTTransformer(
        src_dim            = fused_dim,
        tgt_vocab_size     = text_vocab_size,
        d_model            = c["d_model"],
        nhead              = c["nhead"],
        num_encoder_layers = c["num_encoder_layers"],
        num_decoder_layers = c["num_decoder_layers"],
        dim_feedforward    = c["dim_feedforward"],
        dropout            = c["dropout"],
        max_seq_len        = c["max_seq_len"],
    )

    blank_idx = cfg["cslr"]["ctc_blank_idx"]

    class CSLTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cslr        = cslr_model
            self.gloss_embed = gloss_embed
            self.late_fusion = late_fusion
            self.translator  = translator

        def forward(self, frames, frame_lens, gloss, gloss_lens, tgt, tgt_lens):
            with torch.no_grad():
                _, visual_hidden, adj_lens = self.cslr(frames, frame_lens)
            gloss_emb = self.gloss_embed(gloss)
            fused     = self.late_fusion(visual_hidden, gloss_emb)
            return self.translator(fused, tgt, adj_lens, tgt_lens)

        @torch.no_grad()
        def translate(self, frames, frame_lens, bos_idx, eos_idx):
            log_probs, visual_hidden, adj_lens = self.cslr(frames, frame_lens)
            T_out = log_probs.size(0)
            pred_gloss_ids = batch_ctc_decode(
                log_probs, adj_lens.clamp(max=T_out),
                blank_idx=blank_idx, mode="greedy",
            )
            device  = frames.device
            max_g   = max((len(g) for g in pred_gloss_ids), default=1)
            max_g   = max(max_g, 1)
            eff_gloss = torch.zeros(frames.size(0), max_g, dtype=torch.long, device=device)
            for b, pred in enumerate(pred_gloss_ids):
                if pred:
                    t = torch.tensor(pred[:max_g], dtype=torch.long, device=device)
                    eff_gloss[b, :len(t)] = t
            gloss_emb = self.gloss_embed(eff_gloss)
            fused     = self.late_fusion(visual_hidden, gloss_emb)
            return self.translator.greedy_decode(fused, bos_idx, eos_idx, adj_lens)

    return CSLTModel()


def build_dual_slt_model(
    cfg, gloss_vocab_size: int, text_vocab_size: int,
    primary_cslr: nn.Module, secondary_cslr: nn.Module,
) -> nn.Module:
    """
    Build SLT model with dual-encoder fusion.
    Primary = Swin CSLR, Secondary = ResNet34 CSLR.
    Both CSLR models are frozen; their hidden states are fused via
    DualEncoderFusion before being passed through LateFusion + Translator.
    """
    from models.late_fusion          import LateFusion, GlossEmbedding
    from models.translator           import SLTTransformer
    from models.dual_encoder_fusion  import DualEncoderFusion

    # Hidden dims from the two CSLR seq_models
    dim1 = primary_cslr.seq_model.hidden_out_dim
    dim2 = secondary_cslr.seq_model.hidden_out_dim

    dc = cfg.get("dual_fusion", {})
    dual_fuse = DualEncoderFusion(
        dim1      = dim1,
        dim2      = dim2,
        fused_dim = dc.get("fused_dim", 512),
        mode      = dc.get("fusion_mode", "gate"),
        dropout   = dc.get("dropout", 0.15),
    )
    dual_out_dim = dc.get("fused_dim", 512)

    fc          = cfg["fusion"]
    gloss_embed = GlossEmbedding(gloss_vocab_size, fc["gloss_embed_dim"])
    late_fusion = LateFusion(
        visual_dim      = dual_out_dim,
        gloss_embed_dim = fc["gloss_embed_dim"],
        fused_dim       = fc["fused_dim"],
        mode            = fc["mode"],
        dropout         = fc["dropout"],
        nhead           = fc.get("nhead", 8),
    )
    fused_dim = fc["fused_dim"]

    c = cfg["transformer_2d"]
    translator = SLTTransformer(
        src_dim            = fused_dim,
        tgt_vocab_size     = text_vocab_size,
        d_model            = c["d_model"],
        nhead              = c["nhead"],
        num_encoder_layers = c["num_encoder_layers"],
        num_decoder_layers = c["num_decoder_layers"],
        dim_feedforward    = c["dim_feedforward"],
        dropout            = c["dropout"],
        max_seq_len        = c["max_seq_len"],
    )

    blank_idx = cfg["cslr"]["ctc_blank_idx"]

    class DualCSLTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cslr1       = primary_cslr      # Swin (frozen)
            self.cslr2       = secondary_cslr    # ResNet34 (frozen)
            self.dual_fuse   = dual_fuse
            self.gloss_embed = gloss_embed
            self.late_fusion = late_fusion
            self.translator  = translator

        def _get_dual_hidden(self, frames, frame_lens):
            """Extract and fuse hidden states from both CSLR encoders."""
            with torch.no_grad():
                _, h1, lens1 = self.cslr1(frames, frame_lens)
                _, h2, lens2 = self.cslr2(frames, frame_lens)
            fused_vis, fused_lens = self.dual_fuse(h1, h2, lens1, lens2)
            return fused_vis, fused_lens

        def _get_primary_gloss(self, frames, frame_lens):
            """Get gloss predictions from primary CSLR for translate()."""
            with torch.no_grad():
                log_probs, _, adj_lens = self.cslr1(frames, frame_lens)
            T_out = log_probs.size(0)
            return batch_ctc_decode(
                log_probs, adj_lens.clamp(max=T_out),
                blank_idx=blank_idx, mode="greedy",
            )

        def forward(self, frames, frame_lens, gloss, gloss_lens, tgt, tgt_lens):
            fused_vis, fused_lens = self._get_dual_hidden(frames, frame_lens)
            gloss_emb = self.gloss_embed(gloss)
            fused     = self.late_fusion(fused_vis, gloss_emb)
            return self.translator(fused, tgt, fused_lens, tgt_lens)

        @torch.no_grad()
        def translate(self, frames, frame_lens, bos_idx, eos_idx):
            fused_vis, fused_lens = self._get_dual_hidden(frames, frame_lens)
            pred_gloss_ids = self._get_primary_gloss(frames, frame_lens)

            device = frames.device
            max_g  = max((len(g) for g in pred_gloss_ids), default=1)
            max_g  = max(max_g, 1)
            eff_gloss = torch.zeros(frames.size(0), max_g, dtype=torch.long, device=device)
            for b, pred in enumerate(pred_gloss_ids):
                if pred:
                    t = torch.tensor(pred[:max_g], dtype=torch.long, device=device)
                    eff_gloss[b, :len(t)] = t

            gloss_emb = self.gloss_embed(eff_gloss)
            fused     = self.late_fusion(fused_vis, gloss_emb)
            return self.translator.greedy_decode(fused, bos_idx, eos_idx, fused_lens)

    return DualCSLTModel()


# ------------------------------------------------------------------------------
# LR Schedulers
# ------------------------------------------------------------------------------

def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int,
                       eta_min: float = 0.0) -> LambdaLR:
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * prog)))
    return LambdaLR(optimizer, lr_lambda)

def step_scheduler(optimizer, milestones, gamma=0.2) -> MultiStepLR:
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


# ------------------------------------------------------------------------------
# Optimizer builder
# ------------------------------------------------------------------------------

def build_cslr_optimizer(model, cfg, encoder_type: str):
    c       = cfg["cslr"]
    base_lr = c["learning_rate"]
    scale   = c.get("encoder_lr_scale", 0.05)
    wd      = c.get("weight_decay", 1e-4)

    enc = model.encoder
    if hasattr(enc, "patch_embed"):
        # VideoSwin (new structure): pretrained components get scaled LR
        backbone_params = (list(enc.patch_embed.parameters()) +
                           list(enc.stages.parameters()) +
                           list(enc.swin_norm.parameters()))
        proj_params     = (list(enc.proj.parameters()) +
                           list(enc.out_norm.parameters()) +
                           list(enc.temporal_pool.parameters()))
        enc_groups = [
            {"params": backbone_params, "lr": base_lr * scale,       "name": "enc_backbone"},
            {"params": proj_params,     "lr": base_lr * scale * 5.0, "name": "enc_proj"},
        ]
    elif hasattr(enc, "backbone"):
        # Legacy VideoSwin: backbone (pretrained) gets tiny LR
        backbone_params = list(enc.backbone.parameters())
        proj_params     = (list(enc.proj.parameters()) +
                           (list(enc.out_norm.parameters()) if hasattr(enc, "out_norm") else []))
        enc_groups = [
            {"params": backbone_params, "lr": base_lr * scale,       "name": "enc_backbone"},
            {"params": proj_params,     "lr": base_lr * scale * 5.0, "name": "enc_proj"},
        ]
    else:
        # ResNet: uniform scale
        enc_groups = [{"params": list(enc.parameters()), "lr": base_lr * scale, "name": "encoder"}]

    seq_group = {"params": list(model.seq_model.parameters()), "lr": base_lr, "name": "seq_model"}
    groups = enc_groups + [seq_group]

    sched_type = c.get("lr_scheduler", "cosine")
    if sched_type == "step":
        return Adam(groups, weight_decay=wd)
    else:
        return AdamW(groups, weight_decay=wd)


# ------------------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------------------

def make_loader(split: str, return_translation: bool, cfg: dict,
                gloss_vocab, text_vocab, encoder_type: str, seed: int = 42):
    dc    = cfg["data"]
    cslrc = cfg["cslr"]

    ts_cfg = dc.get("augmentation", {}).get("temporal_scale", {})
    temporal_scale = (ts_cfg.get("min_scale", 0.8), ts_cfg.get("max_scale", 1.2)) \
                     if split == "train" else None

    clip_aug = (dc["img_height"], dc["img_width"]) if split == "train" else None

    ds = PhoenixDataset(
        split                = split,
        gloss_vocab          = gloss_vocab,
        text_vocab           = text_vocab,
        max_frames           = dc["max_frames"],
        temporal_stride      = dc["temporal_stride"],
        transform            = build_transforms(split, dc["img_height"], dc["img_width"]),
        return_translation   = return_translation,
        temporal_scale_range = temporal_scale,
        clip_aug_crop        = clip_aug,
    )

    bs      = cslrc["batch_size"] if not return_translation else cfg["slt"]["batch_size"]
    g       = torch.Generator(); g.manual_seed(seed)
    nw      = dc["num_workers"]
    use_pin = dc.get("pin_memory", False)   # False on cluster -- spawn+pin ? /dev/shm crash

    return DataLoader(
        ds,
        batch_size         = bs,
        shuffle            = (split == "train"),
        num_workers        = nw,
        pin_memory         = use_pin,
        prefetch_factor    = dc.get("prefetch_factor", 2) if nw > 0 else None,
        collate_fn         = collate_fn,
        drop_last          = (split == "train"),
        generator          = g if split == "train" else None,
        worker_init_fn     = _worker_init,
        persistent_workers = (nw > 0),
        # multiprocessing_context not set -> Linux uses "fork"
        # "fork" does not use /dev/shm -> no shared memory crash
    )


# ------------------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------------------

def find_latest_ckpt(ckpt_dir: Path):
    # First try numbered checkpoints
    numbered = sorted(
        [p for p in ckpt_dir.glob("ckpt_epoch_*.pth") if p.stem != "ckpt_epoch_latest"],
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    # Also check the every-epoch 'latest' checkpoint
    latest = ckpt_dir / "ckpt_epoch_latest.pth"
    if latest.exists():
        ck_latest = torch.load(latest, map_location="cpu", weights_only=False)
        ep_latest = ck_latest["epoch"]
        # Use 'latest' if it is ahead of numbered checkpoints
        if not numbered or ep_latest > int(numbered[-1].stem.split("_")[-1]):
            return latest, ep_latest
    if numbered:
        last = numbered[-1]
        epoch = int(last.stem.split("_")[-1])
        return last, epoch
    return None, 0


# ------------------------------------------------------------------------------
# CSLR Training
# ------------------------------------------------------------------------------

def train_cslr(
    cfg:          dict,
    encoder_type: str,
    device:       torch.device,
    gloss_vocab,
    ckpt_dir:     Path,
    seed:         int = 42,
) -> tuple:
    """
    Train CSLR (encoder + BiLSTM + CTC).
    Returns (best_wer, model).
    """
    print(f"\n{'-'*64}")
    print(f"CSLR  |  encoder={encoder_type}  |  BiLSTM + CTC")
    print(f"{'-'*64}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir = ckpt_dir / "best"
    best_dir.mkdir(exist_ok=True)

    c = cfg["cslr"]

    train_loader = make_loader("train", False, cfg, gloss_vocab, None, encoder_type, seed)
    dev_loader   = make_loader("dev",   False, cfg, gloss_vocab, None, encoder_type, seed)

    num_classes = len(gloss_vocab)
    model = build_cslr_model(cfg, num_classes, encoder_type).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params total     : {n_total:,}")
    print(f"  Params trainable : {n_train:,}")

    criterion = CTCCriterion(blank_idx=c["ctc_blank_idx"])
    optimizer = build_cslr_optimizer(model, cfg, encoder_type)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" and c.get("use_amp", True) else None

    sched_type   = c.get("lr_scheduler", "cosine")
    num_epochs   = c["num_epochs"]
    grad_acc     = c.get("grad_accumulation_steps", 4)
    warmup_ep    = c.get("warmup_epochs", 5)
    steps_per_ep = len(train_loader)

    if sched_type == "cosine":
        total_steps  = num_epochs * steps_per_ep
        warmup_steps = warmup_ep * steps_per_ep
        base_lr      = c["learning_rate"]
        eta_ratio    = c.get("eta_min", 1e-6) / base_lr
        scheduler    = cosine_with_warmup(optimizer, warmup_steps, total_steps, eta_ratio)
        step_per_batch = True
    else:
        milestones = c.get("lr_decay_epochs", [30, 38])
        gamma      = c.get("lr_decay_rate", 0.2)
        if warmup_ep > 0:
            # Linear warmup for warmup_ep epochs, then MultiStepLR
            # FIX: MultiStepLR internal counter starts from 0 when SequentialLR
            # hands off at epoch `warmup_ep`, so milestones must be shifted by
            # -warmup_ep to match the global epoch numbers in the config.
            from torch.optim.lr_scheduler import SequentialLR, LinearLR
            adjusted_milestones = [max(1, m - warmup_ep) for m in milestones]
            warmup_sched = LinearLR(optimizer, start_factor=0.01,
                                    end_factor=1.0, total_iters=warmup_ep)
            step_sched   = MultiStepLR(optimizer, milestones=adjusted_milestones, gamma=gamma)
            scheduler    = SequentialLR(optimizer,
                                        schedulers=[warmup_sched, step_sched],
                                        milestones=[warmup_ep])
        else:
            scheduler = step_scheduler(optimizer, milestones, gamma)
        step_per_batch = False

    # -- Resume if checkpoint exists -------------------------------------------
    start_epoch = 0
    best_wer    = float("inf")
    no_improve  = 0
    best_count  = 0

    last_ckpt, last_ep = find_latest_ckpt(ckpt_dir)
    if last_ckpt is not None:
        print(f"  Resuming from {last_ckpt}")
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("optimizer") is not None:
            optimizer.load_state_dict(ck["optimizer"])
        else:
            print("  [Warm-start] No optimizer state — starting optimizer fresh")
        if ck.get("scheduler") is not None:
            scheduler.load_state_dict(ck["scheduler"])
        else:
            print("  [Warm-start] No scheduler state — starting scheduler fresh")
        best_wer    = ck.get("best_wer", best_wer)
        no_improve  = ck.get("no_improve", 0)
        best_count  = ck.get("best_count", 0)
        start_epoch = last_ep
        print(f"  Resumed at epoch {start_epoch}, best WER so far = {best_wer*100:.2f}%")

    # -- Freeze backbone option ------------------------------------------------
    fb_cfg        = c.get("freeze_backbone", {})
    do_freeze     = fb_cfg.get("enabled", False)
    freeze_epochs = fb_cfg.get("freeze_epochs", 5)
    if do_freeze and start_epoch < freeze_epochs:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"  Backbone frozen for {freeze_epochs} epochs")

    # -- Early stopping config -------------------------------------------------
    es_cfg      = c.get("early_stopping", {})
    do_es       = es_cfg.get("enabled", False)
    es_patience = es_cfg.get("patience", 12)

    print(f"\n  Epochs={num_epochs} | EffBatch={c['batch_size']*grad_acc} | "
          f"LR={c['learning_rate']:.0e} | Scheduler={sched_type} | "
          f"enc_lr_scale={c.get('encoder_lr_scale',0.05)}")

    global_step = start_epoch * steps_per_ep

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        # -- Unfreeze backbone at freeze_epochs --------------------------------
        if do_freeze and epoch == freeze_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
            # Rebuild optimizer so backbone params get proper LR groups
            optimizer = build_cslr_optimizer(model, cfg, encoder_type)
            if sched_type == "cosine":
                remaining    = (num_epochs - epoch) * steps_per_ep
                warmup_rem   = 0   # already past warmup
                eta_ratio    = c.get("eta_min", 1e-6) / c["learning_rate"]
                scheduler    = cosine_with_warmup(optimizer, warmup_rem, remaining + global_step,
                                                  eta_ratio)
                # Fast-forward scheduler to current step
                for _ in range(global_step):
                    scheduler.step()
            else:
                # Warmup already passed; rebuild a plain MultiStepLR with the
                # original (unadjusted) milestones and fast-forward to current
                # epoch so future decay steps fire at the correct global epochs.
                scheduler = step_scheduler(optimizer, c.get("lr_decay_epochs", [30, 38]),
                                           c.get("lr_decay_rate", 0.2))
                for _ in range(epoch):
                    scheduler.step()
            print(f"  [Epoch {epoch+1}] Backbone UNFROZEN")

        # -- Train epoch -------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0; n_batches = 0

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"  CSLR Ep{epoch+1:3d}/{num_epochs}", leave=False)
        ):
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                log_probs, _, adj_lens = model(frames, frame_lens)
                loss = criterion(log_probs, gloss, adj_lens, gloss_lens) / grad_acc

            epoch_loss += loss.item() * grad_acc
            n_batches  += 1

            if scaler:
                scaler.scale(loss).backward()
                if (step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"])
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"])
                    optimizer.step(); optimizer.zero_grad()

            if step_per_batch:
                scheduler.step()
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if not step_per_batch:
            scheduler.step()

        # -- Evaluate ----------------------------------------------------------
        model.eval()
        hyp_g, ref_g = [], []
        with torch.no_grad():
            for batch in dev_loader:
                frames     = batch["frames"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                gloss      = batch["gloss"].to(device)
                gloss_lens = batch["gloss_lens"].to(device)

                log_probs, _, adj_lens = model(frames, frame_lens)
                preds = batch_ctc_decode(log_probs, adj_lens,
                                         blank_idx=c["ctc_blank_idx"])
                for b, pred in enumerate(preds):
                    hyp_g.append(gloss_vocab.decode(pred))
                    ref_g.append(gloss_vocab.decode(
                        gloss[b, :gloss_lens[b].item()].tolist(), skip_special=False))

        wer     = compute_wer(hyp_g, ref_g)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[-1]["lr"]   # seq_model LR

        print(f"  Ep {epoch+1:3d}/{num_epochs} | loss={avg_loss:.4f} | "
              f"WER={wer*100:.2f}% | lr={current_lr:.1e} | {elapsed:.0f}s"
              + (" ? BEST" if wer < best_wer else f" (no improve {no_improve+1}/{es_patience})"))

        if wer < best_wer:
            best_wer   = wer
            no_improve = 0
            best_count += 1
            torch.save({
                "epoch":    epoch + 1,
                "model":    model.state_dict(),
                "best_wer": best_wer,
            }, best_dir / f"best_{best_count:03d}_ep{epoch+1}_wer{wer*100:.2f}.pth")
            # Also save as canonical best
            torch.save({
                "epoch": epoch + 1, "model": model.state_dict(),
                "best_wer": best_wer, "encoder_type": encoder_type,
            }, ckpt_dir.parent / "best_cslr.pth")
        else:
            no_improve += 1

        # -- Resume checkpoint (every epoch, overwrite to save disk) -----------
        resume_path = ckpt_dir / "ckpt_epoch_latest.pth"
        torch.save({
            "epoch": epoch + 1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_wer": best_wer, "no_improve": no_improve, "best_count": best_count,
        }, resume_path)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            import shutil
            shutil.copy2(resume_path, ckpt_dir / f"ckpt_epoch_{epoch+1}.pth")

        if do_es and no_improve >= es_patience:
            print(f"\n  Early stopping at epoch {epoch+1}. Best WER = {best_wer*100:.2f}%")
            break

    print(f"\n  ? CSLR done. Best WER = {best_wer*100:.2f}%")

    # Restore best model
    best_ckpt = ckpt_dir.parent / "best_cslr.pth"
    if best_ckpt.exists():
        saved = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])

    return best_wer, model


# ------------------------------------------------------------------------------
# SLT Training
# ------------------------------------------------------------------------------

def train_slt(
    cfg:          dict,
    cslr_model:   nn.Module,
    encoder_type: str,
    device:       torch.device,
    gloss_vocab,
    text_vocab,
    ckpt_dir:     Path,
    seed:         int = 42,
    secondary_cslr: Optional[nn.Module] = None,
) -> float:
    """
    Train SLT (LateFusion + Transformer-2D).
    Returns best_bleu4.
    """
    print(f"\n{'-'*64}")
    print(f"SLT  |  LateFusion (attention) ? Transformer-2D decoder")
    print(f"{'-'*64}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir = ckpt_dir / "best"
    best_dir.mkdir(exist_ok=True)

    # Freeze CSLR backbone(s) completely
    for p in cslr_model.parameters():
        p.requires_grad = False
    if secondary_cslr is not None:
        for p in secondary_cslr.parameters():
            p.requires_grad = False

    # Build model: dual-encoder or single-encoder
    dc = cfg.get("dual_fusion", {})
    use_dual = dc.get("enabled", False) and secondary_cslr is not None
    if use_dual:
        print("  [Dual-Encoder] Fusing primary + secondary CSLR hidden states")
        model = build_dual_slt_model(
            cfg, len(gloss_vocab), len(text_vocab), cslr_model, secondary_cslr
        ).to(device)
    else:
        model = build_slt_model(cfg, len(gloss_vocab), len(text_vocab), cslr_model).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frz   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    fusion_label = "DualEncoderFusion + LateFusion + Translator" if use_dual else "LateFusion + Translator"
    backbone_label = "CSLR backbones (Swin + ResNet34)" if use_dual else "CSLR backbone"
    print(f"  Params trainable : {n_train:,}  ({fusion_label})")
    print(f"  Params frozen    : {n_frz:,}   ({backbone_label})")

    train_loader = make_loader("train", True, cfg, gloss_vocab, text_vocab, encoder_type, seed)
    dev_loader   = make_loader("dev",   True, cfg, gloss_vocab, text_vocab, encoder_type, seed)

    sc         = cfg["slt"]
    pad_idx    = text_vocab.token2idx[text_vocab.PAD]
    bos_idx    = text_vocab.token2idx[text_vocab.BOS]
    eos_idx    = text_vocab.token2idx[text_vocab.EOS]

    criterion = nn.CrossEntropyLoss(
        label_smoothing = sc.get("label_smoothing", 0.1),
        ignore_index    = pad_idx,
    )
    trainable   = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(trainable, lr=sc["learning_rate"], weight_decay=sc["weight_decay"])

    num_epochs   = sc["num_epochs"]
    steps_per_ep = len(train_loader)
    warmup_steps = sc.get("warmup_steps", 500)
    total_steps  = num_epochs * steps_per_ep
    eta_ratio    = sc.get("eta_min", 1e-8) / sc["learning_rate"]
    scheduler    = cosine_with_warmup(optimizer, warmup_steps, total_steps, eta_ratio)

    scaler   = torch.amp.GradScaler("cuda") if device.type == "cuda" and sc.get("use_amp", True) else None
    grad_acc = sc.get("grad_accumulation_steps", 2)

    # Resume
    start_epoch = 0; best_bleu = -1; no_improve = 0; best_count = 0
    last_ckpt, last_ep = find_latest_ckpt(ckpt_dir)
    if last_ckpt is not None:
        print(f"  Resuming from {last_ckpt}")
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        best_bleu   = ck.get("best_bleu", best_bleu)
        no_improve  = ck.get("no_improve", 0)
        best_count  = ck.get("best_count", 0)
        start_epoch = last_ep

    es_cfg      = sc.get("early_stopping", {})
    do_es       = es_cfg.get("enabled", False)
    es_patience = es_cfg.get("patience", 8)

    print(f"\n  Epochs={num_epochs} | EffBatch={sc['batch_size']*grad_acc} | "
          f"LR={sc['learning_rate']:.0e} | warmup={warmup_steps} steps | "
          f"fusion={cfg['fusion']['mode']}")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"  SLT  Ep{epoch+1:3d}/{num_epochs}", leave=False)
        ):
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)
            tgt        = batch["translation"].to(device)
            tgt_lens   = batch["translation_lens"].to(device)

            tgt_in      = tgt[:, :-1]
            tgt_out     = tgt[:, 1:]
            tgt_lens_in = (tgt_lens - 1).clamp(min=1)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(frames, frame_lens, gloss, gloss_lens, tgt_in, tgt_lens_in)
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B*T, V), tgt_out.reshape(B*T)) / grad_acc

            if scaler:
                scaler.scale(loss).backward()
                if (step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, sc["gradient_clip"])
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, sc["gradient_clip"])
                    optimizer.step(); optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item() * grad_acc

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        hyp_s, ref_s = [], []
        with torch.no_grad():
            for batch in dev_loader:
                frames     = batch["frames"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                tgt        = batch["translation"]
                pred_ids   = model.translate(frames, frame_lens, bos_idx, eos_idx)
                for b in range(pred_ids.size(0)):
                    hyp_s.append(" ".join(text_vocab.decode(pred_ids[b].tolist())))
                    ref_s.append(" ".join(text_vocab.decode(tgt[b].tolist())))

        bleu    = compute_bleu(hyp_s, ref_s)["bleu"]
        rouge   = compute_rouge(hyp_s, ref_s)
        meteor  = compute_meteor(hyp_s, ref_s)
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"  Ep {epoch+1:3d}/{num_epochs} | loss={avg_loss:.4f} | "
              f"BLEU-4={bleu:.2f} | ROUGE-L={rouge['rougeL']:.4f} | "
              f"METEOR={meteor:.4f} | lr={lr_now:.1e} | {elapsed:.0f}s"
              + (" ? BEST" if bleu > best_bleu else f" (no improve {no_improve+1}/{es_patience})"))

        if bleu > best_bleu:
            best_bleu  = bleu
            no_improve = 0
            best_count += 1
            torch.save({"epoch": epoch+1, "model": model.state_dict(),
                        "best_bleu": best_bleu, "bleu": bleu,
                        "rougeL": rouge["rougeL"], "meteor": meteor},
                       best_dir / f"best_{best_count:03d}_ep{epoch+1}_bleu{bleu:.2f}.pth")
            torch.save({"epoch": epoch+1, "model": model.state_dict(),
                        "best_bleu": best_bleu, "bleu": bleu,
                        "rougeL": rouge["rougeL"], "meteor": meteor,
                        "gloss_vocab": gloss_vocab, "text_vocab": text_vocab},
                       ckpt_dir.parent / "best_slt.pth")
        else:
            no_improve += 1

        # Save resume checkpoint every epoch (overwrite previous to save disk)
        resume_path = ckpt_dir / "ckpt_epoch_latest.pth"
        torch.save({"epoch": epoch+1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_bleu": best_bleu, "no_improve": no_improve, "best_count": best_count},
                   resume_path)
        # Also keep a numbered copy every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            import shutil
            shutil.copy2(resume_path, ckpt_dir / f"ckpt_epoch_{epoch+1}.pth")

        if do_es and no_improve >= es_patience:
            print(f"\n  Early stopping at epoch {epoch+1}. Best BLEU-4 = {best_bleu:.2f}")
            break

    print(f"\n  ? SLT done. Best BLEU-4 = {best_bleu:.2f}")
    return best_bleu


# ------------------------------------------------------------------------------
# Final evaluation on test set
# ------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test(cfg, cslr_ckpt_path: Path, slt_ckpt_path: Path,
                  encoder_type: str, device: torch.device):
    print(f"\n{'-'*64}")
    print("TEST SET EVALUATION")
    print(f"{'-'*64}")

    slt_ck  = torch.load(slt_ckpt_path, map_location="cpu", weights_only=False)
    gloss_v = slt_ck["gloss_vocab"]
    text_v  = slt_ck["text_vocab"]

    cslr_model = build_cslr_model(cfg, len(gloss_v), encoder_type)
    cslr_ck    = torch.load(cslr_ckpt_path, map_location="cpu", weights_only=False)
    cslr_model.load_state_dict(cslr_ck["model"])

    model = build_slt_model(cfg, len(gloss_v), len(text_v), cslr_model)
    model.load_state_dict(slt_ck["model"])
    model.to(device).eval()

    dc = cfg["data"]
    test_ds = PhoenixDataset(
        split="test", gloss_vocab=gloss_v, text_vocab=text_v,
        max_frames=dc["max_frames"], temporal_stride=dc["temporal_stride"],
        transform=build_transforms("test", dc["img_height"], dc["img_width"]),
        return_translation=True,
    )
    loader = DataLoader(test_ds, batch_size=cfg["slt"]["batch_size"],
                        collate_fn=collate_fn, num_workers=dc["num_workers"])

    bos_idx = text_v.token2idx[text_v.BOS]
    eos_idx = text_v.token2idx[text_v.EOS]
    blank   = cfg["cslr"]["ctc_blank_idx"]

    hyp_g, ref_g, hyp_s, ref_s = [], [], [], []

    for batch in tqdm(loader, desc="[Test]"):
        frames     = batch["frames"].to(device)
        frame_lens = batch["frame_lens"].to(device)
        gloss      = batch["gloss"].to(device)
        gloss_lens = batch["gloss_lens"].to(device)
        tgt        = batch["translation"]

        log_probs, _, adj_lens = model.cslr(frames, frame_lens)
        preds = batch_ctc_decode(log_probs, adj_lens.clamp(max=log_probs.size(0)),
                                 blank_idx=blank)
        for b, pred in enumerate(preds):
            hyp_g.append(gloss_v.decode(pred))
            ref_g.append(gloss_v.decode(gloss[b, :gloss_lens[b].item()].tolist(), skip_special=False))

        pred_ids = model.translate(frames, frame_lens, bos_idx, eos_idx)
        for b in range(pred_ids.size(0)):
            hyp_s.append(" ".join(text_v.decode(pred_ids[b].tolist())))
            ref_s.append(" ".join(text_v.decode(tgt[b].tolist())))

    wer   = compute_wer(hyp_g, ref_g)
    bleu  = compute_bleu(hyp_s, ref_s)
    rouge = compute_rouge(hyp_s, ref_s)
    met   = compute_meteor(hyp_s, ref_s)

    print(f"\n  +-----------------------------------------------------+")
    print(f"  |  CSLR WER        : {wer*100:6.2f}%                      |")
    print(f"  |  BLEU-4          : {bleu['bleu']:6.2f}                       |")
    print(f"  |  BLEU-1/2/3      : {bleu['bleu1']:.2f} / {bleu['bleu2']:.2f} / {bleu.get('bleu3',0):.2f}         |")
    print(f"  |  ROUGE-1/2/L     : {rouge['rouge1']:.4f} / {rouge['rouge2']:.4f} / {rouge['rougeL']:.4f} |")
    print(f"  |  METEOR          : {met:.4f}                       |")
    print(f"  +-----------------------------------------------------+")

    print("\n  Sample predictions (dev set):")
    for i in range(min(3, len(hyp_s))):
        print(f"    REF: {ref_s[i]}")
        print(f"    HYP: {hyp_s[i]}")
        print()

    results = {
        "encoder": encoder_type, "WER": round(wer*100, 2),
        "BLEU-4": bleu["bleu"], "BLEU-1": bleu["bleu1"], "BLEU-2": bleu["bleu2"],
        "ROUGE-L": rouge["rougeL"], "METEOR": met,
    }
    out_path = slt_ckpt_path.parent / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved ? {out_path}")
    return results


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Final CSLR->SLT Pipeline")
    parser.add_argument("--config",    required=True,
                        help="Path to config yaml (config_resnet34.yaml or config_swin.yaml)")
    parser.add_argument("--encoder",   required=True, choices=["resnet34", "swin"],
                        help="Visual encoder type")
    parser.add_argument("--stage",     default="all", choices=["all", "cslr", "slt", "eval"],
                        help="Which stage to run")
    parser.add_argument("--cslr_ckpt", default=None,
                        help="Pre-trained CSLR checkpoint (for SLT-only or eval)")
    parser.add_argument("--dual_cslr_ckpt", default=None,
                        help="Secondary CSLR checkpoint for dual-encoder fusion (e.g., ResNet34 ckpt when primary is Swin)")
    parser.add_argument("--slt_ckpt",  default=None,
                        help="Pre-trained SLT checkpoint (for eval-only)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    _ckpt_raw = Path(cfg["paths"]["checkpoint_dir"])
    if _ckpt_raw.is_absolute():
        ckpt_root = _ckpt_raw
    else:
        # Resolve relative to project root (parent of training/)
        project_root = Path(__file__).resolve().parent.parent
        ckpt_root = project_root / _ckpt_raw
    ckpt_root = ckpt_root.resolve()
    print(f"Checkpoints: {ckpt_root}")
    cslr_dir  = ckpt_root / "cslr"
    slt_dir   = ckpt_root / "slt"

    gloss_vocab, text_vocab = build_vocabularies()
    print(f"Vocab  : gloss={len(gloss_vocab)} | text={len(text_vocab)}")

    best_cslr_ckpt = ckpt_root / "best_cslr.pth"
    best_slt_ckpt  = ckpt_root / "best_slt.pth"

    # Override with CLI arguments
    if args.cslr_ckpt:
        best_cslr_ckpt = Path(args.cslr_ckpt)
    if args.slt_ckpt:
        best_slt_ckpt = Path(args.slt_ckpt)

    # Resolve secondary CSLR checkpoint for dual-encoder fusion
    dc = cfg.get("dual_fusion", {})
    dual_cslr_path = None
    if args.dual_cslr_ckpt:
        dual_cslr_path = Path(args.dual_cslr_ckpt)
    elif dc.get("enabled", False) and dc.get("resnet34_cslr_ckpt"):
        dual_cslr_path = Path(dc["resnet34_cslr_ckpt"])

    t_start = time.time()

    # -- CSLR Stage -------------------------------------------------------------
    cslr_model = None
    if args.stage in ("all", "cslr"):
        best_wer, cslr_model = train_cslr(
            cfg, args.encoder, device, gloss_vocab, cslr_dir, seed
        )
        torch.cuda.empty_cache(); gc.collect()

    # -- SLT Stage --------------------------------------------------------------
    if args.stage in ("all", "slt"):
        if cslr_model is None:
            # Load from checkpoint
            assert best_cslr_ckpt.exists(), f"CSLR checkpoint not found: {best_cslr_ckpt}"
            print(f"\nLoading CSLR from {best_cslr_ckpt}")
            ck = torch.load(best_cslr_ckpt, map_location="cpu", weights_only=False)
            cslr_model = build_cslr_model(cfg, len(gloss_vocab), args.encoder)
            cslr_model.load_state_dict(ck["model"])
            cslr_model.to(device)

        # Load secondary CSLR for dual-encoder fusion if configured
        secondary_cslr = None
        if dual_cslr_path is not None and dual_cslr_path.exists():
            # Determine secondary encoder type (opposite of primary)
            sec_encoder = "resnet34" if args.encoder == "swin" else "swin"
            print(f"\nLoading secondary CSLR ({sec_encoder}) from {dual_cslr_path}")
            ck2 = torch.load(dual_cslr_path, map_location="cpu", weights_only=False)
            secondary_cslr = build_cslr_model(cfg, len(gloss_vocab), sec_encoder)
            secondary_cslr.load_state_dict(ck2["model"])
            secondary_cslr.to(device)
        elif dual_cslr_path is not None:
            print(f"\n[WARNING] Dual CSLR checkpoint not found: {dual_cslr_path}")
            print(f"          Falling back to single-encoder SLT.")

        best_bleu = train_slt(
            cfg, cslr_model, args.encoder, device,
            gloss_vocab, text_vocab, slt_dir, seed,
            secondary_cslr=secondary_cslr,
        )
        torch.cuda.empty_cache(); gc.collect()

    # -- Evaluation Stage -------------------------------------------------------
    if args.stage == "eval" or (args.stage == "all" and best_slt_ckpt.exists()):
        evaluate_test(cfg, best_cslr_ckpt, best_slt_ckpt, args.encoder, device)

    total_time = (time.time() - t_start) / 3600
    print(f"\n{'-'*64}")
    print(f"Pipeline complete.  Total time: {total_time:.2f}h")
    print(f"{'-'*64}")


if __name__ == "__main__":
    main()