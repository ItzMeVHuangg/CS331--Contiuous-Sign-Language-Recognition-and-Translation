# -*- coding: utf-8 -*-
"""
train_dual.py
=============
Training pipeline for Dual-Stream CSLR (ResNet34 + VideoSwin) → SLT.

Usage examples
──────────────
# CSLR only (freeze stream1, train stream2 + fusion)
python training/train_dual.py \
    --config   configs/config.yaml \
    --stage    cslr \
    --s1_ckpt  checkpoints/ablation/cslr_variant_A.pth \
    --seed     42

# CSLR + SLT full pipeline
python training/train_dual.py \
    --config   configs/config.yaml \
    --stage    all \
    --s1_ckpt  checkpoints/ablation/cslr_variant_A.pth \
    --seed     42

# SLT only (reuse saved dual CSLR checkpoint)
python training/train_dual.py \
    --config    configs/config.yaml \
    --stage     slt \
    --dual_ckpt checkpoints/dual/best_cslr.pth \
    --seed      42

Phases
──────
CSLR Phase A (freeze_epochs): stream1 frozen, train stream2 + fusion
CSLR Phase B (remaining):     unfreeze stream1, joint fine-tune
SLT stage:                    freeze entire CSLR, train LateFusion + Transformer
"""

import sys
import gc
import math
import random
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset            import build_vocabularies, PhoenixDataset, collate_fn, build_transforms
from models.bilstm_ctc       import CTCCriterion
from models.late_fusion      import LateFusion, GlossEmbedding
from models.translator       import SLTTransformer
from models.dual_stream_cslr import DualStreamCSLR, DualCSLTModel
from utils.ctc_decoder       import batch_ctc_decode
from utils.metrics           import compute_wer, compute_bleu, compute_rouge, compute_meteor

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(split, return_translation, cfg, gloss_vocab, text_vocab, seed=42):
    dc    = cfg["data"]
    ts    = dc.get("augmentation", {}).get("temporal_scale", {})
    t_range = (ts["min_scale"], ts["max_scale"]) if (
        split == "train" and ts) else None
    clip_crop = (dc["img_height"], dc["img_width"]) if split == "train" else None

    ds = PhoenixDataset(
        split                = split,
        gloss_vocab          = gloss_vocab,
        text_vocab           = text_vocab,
        max_frames           = dc["max_frames"],
        temporal_stride      = dc["temporal_stride"],
        transform            = build_transforms(split, dc["img_height"], dc["img_width"]),
        return_translation   = return_translation,
        temporal_scale_range = t_range,
        clip_aug_crop        = clip_crop,
    )

    bs = cfg["slt"]["batch_size"] if return_translation else cfg["cslr"]["batch_size"]
    nw = dc["num_workers"]
    g  = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size         = bs,
        shuffle            = (split == "train"),
        num_workers        = nw,
        pin_memory         = False,
        prefetch_factor    = dc.get("prefetch_factor", 2) if nw > 0 else None,
        collate_fn         = collate_fn,
        drop_last          = (split == "train"),
        generator          = g if split == "train" else None,
        worker_init_fn     = _worker_init_fn,
        persistent_workers = (nw > 0),
    )


# ══════════════════════════════════════════════════════════════════════════════
# LR schedule
# ══════════════════════════════════════════════════════════════════════════════

def cosine_with_warmup(optimizer, warmup_steps, total_steps, eta_min=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * prog)))
    return LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer builder
# ══════════════════════════════════════════════════════════════════════════════

def build_cslr_optimizer(model: DualStreamCSLR, cfg: dict, phase: str = "A"):
    """
    Phase A (stream1 frozen):
        - swin backbone    : base_lr * swin_scale
        - swin proj/norm   : base_lr * swin_scale * 5
        - bilstm2          : base_lr
        - cross-attn/fusion: base_lr

    Phase B (all unfrozen):
        - resnet34         : base_lr * resnet_scale
        - bilstm1          : base_lr * resnet_scale * 2
        - swin backbone    : base_lr * swin_scale
        - bilstm2          : base_lr
        - cross-attn/fusion: base_lr
    """
    c         = cfg["cslr"]
    base_lr   = c["learning_rate"]
    wd        = c.get("weight_decay", 5e-4)
    r_scale   = c.get("encoder_lr_scale", 0.1)      # ResNet34 scale
    s_scale   = c.get("encoder_lr_scale", 0.1) * 0.5  # Swin scale (smaller, pretrained 3D)

    swin      = model.swin_encoder
    swin_bb   = (list(swin.patch_embed.parameters()) +
                 list(swin.stages.parameters())      +
                 list(swin.swin_norm.parameters()))
    swin_proj = list(swin.proj.parameters()) + list(swin.out_norm.parameters())

    param_groups = [
        {"params": swin_bb,   "lr": base_lr * s_scale,       "name": "swin_backbone"},
        {"params": swin_proj, "lr": base_lr * s_scale * 5.0, "name": "swin_proj"},
        {"params": list(model.bilstm2.parameters()),
         "lr": base_lr, "name": "bilstm2"},
        {"params": list(model.cross_attn.parameters()) +
                   list(model.fusion_norm.parameters()) +
                   list(model.fused_ctc_head.parameters()),
         "lr": base_lr, "name": "fusion"},
    ]

    if phase == "B":
        cnn = model.cnn_encoder
        param_groups += [
            {"params": list(cnn.feature_extractor.parameters()),
             "lr": base_lr * r_scale, "name": "resnet34_bb"},
            {"params": list(cnn.proj.parameters()) + list(cnn.out_norm.parameters()),
             "lr": base_lr * r_scale * 2.0, "name": "resnet34_proj"},
            {"params": list(model.bilstm1.parameters()),
             "lr": base_lr * r_scale * 2.0, "name": "bilstm1"},
        ]

    return AdamW(param_groups, weight_decay=wd)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_latest_ckpt(ckpt_dir: Path):
    ckpts = list(ckpt_dir.glob("checkpoint_*.pth"))
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda p: int(p.stem.split("_")[-1]))
    last  = ckpts[-1]
    epoch = int(last.stem.split("_")[-1])
    return last, epoch


# ══════════════════════════════════════════════════════════════════════════════
# CSLR training loop
# ══════════════════════════════════════════════════════════════════════════════

def run_cslr(
    model:        DualStreamCSLR,
    train_loader: DataLoader,
    dev_loader:   DataLoader,
    cfg:          dict,
    device:       torch.device,
    gloss_vocab,
    ckpt_dir:     Path,
    best_dir:     Path,
):
    """
    Two-phase training:
      Phase A: stream1 frozen, train stream2 + fusion
      Phase B: all unfrozen, joint fine-tune

    Loss = λ1*CTC1 + λ2*CTC2 + λ3*CTC_fused
    """
    c          = cfg["cslr"]
    num_epochs = c["num_epochs"]
    grad_acc   = c.get("grad_accumulation_steps", 1)
    grad_clip  = c.get("gradient_clip", 5.0)
    blank_idx  = c.get("ctc_blank_idx", 0)
    use_amp    = c.get("use_amp", True) and device.type == "cuda"

    # Loss weights: lower λ2 because stream-2 is noisier (43% vs 24% WER)
    lw = cfg.get("dual_cslr", {})
    lambda1     = lw.get("lambda1", 0.3)
    lambda2     = lw.get("lambda2", 0.1)
    lambda_fuse = lw.get("lambda_fuse", 0.6)
    freeze_epochs = lw.get("freeze_epochs", 15)

    criterion   = CTCCriterion(blank_idx=blank_idx)
    scaler      = torch.amp.GradScaler("cuda") if use_amp else None

    # Resume
    last_ckpt, start_epoch = _find_latest_ckpt(ckpt_dir)
    if last_ckpt:
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_wer = ckpt.get("best_wer", float("inf"))
        print(f"[DualCSLR] Resumed from {last_ckpt} (epoch {start_epoch})")
    else:
        best_wer    = float("inf")
        start_epoch = 0

    # Determine starting phase
    phase = "A" if start_epoch < freeze_epochs else "B"
    if phase == "A":
        model.freeze_stream1()

    optimizer = build_cslr_optimizer(model, cfg, phase=phase)

    n_steps   = num_epochs * len(train_loader)
    warmup_s  = c.get("warmup_epochs", 3) * len(train_loader)
    scheduler = cosine_with_warmup(
        optimizer, warmup_s, n_steps, eta_min=c.get("eta_min", 1e-6))

    # Advance scheduler to resume point
    if start_epoch > 0:
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()

    es_cfg      = c.get("early_stopping", {})
    es_patience = es_cfg.get("patience", 15) if es_cfg.get("enabled", False) else 9999
    no_improve  = 0
    best_count  = 0

    print(f"\n[DualCSLR] Starting epoch {start_epoch+1}/{num_epochs}")
    print(f"  Freeze epochs={freeze_epochs} | λ1={lambda1} λ2={lambda2} λf={lambda_fuse}")
    print(f"  Loss weights: stream1×{lambda1}, stream2×{lambda2}, fused×{lambda_fuse}")

    for epoch in range(start_epoch, num_epochs):

        # ── Phase transition ────────────────────────────────────────────────
        if epoch == freeze_epochs and phase == "A":
            phase = "B"
            model.unfreeze_stream1()
            optimizer = build_cslr_optimizer(model, cfg, phase="B")
            # Restart cosine schedule from current position
            remaining = (num_epochs - epoch) * len(train_loader)
            scheduler = cosine_with_warmup(optimizer, 0, remaining,
                                           eta_min=c.get("eta_min", 1e-6))
            if scaler:
                scaler = torch.amp.GradScaler("cuda")

        # ── Train epoch ─────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches  = 0

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"[DualCSLR Ph-{phase}] E{epoch+1}/{num_epochs}", leave=False)
        ):
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                lp_fused, _, lens1, lp1, lp2, lens2 = model(frames, frame_lens)

                # Three CTC losses
                L1    = criterion(lp1, gloss, lens1, gloss_lens)
                L2    = criterion(lp2, gloss, lens2, gloss_lens)
                L_fuse = criterion(lp_fused, gloss, lens1, gloss_lens)

                loss = (lambda1 * L1 + lambda2 * L2 + lambda_fuse * L_fuse) / grad_acc

            if scaler:
                scaler.scale(loss).backward()
                if (step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()
            epoch_loss += loss.item() * grad_acc
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ── Eval on dev ─────────────────────────────────────────────────────
        model.eval()
        hyp_g, ref_g = [], []
        with torch.no_grad():
            for batch in dev_loader:
                frames     = batch["frames"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                gloss_gt   = batch["gloss"].to(device)
                gloss_lens = batch["gloss_lens"].to(device)

                lp_fused, _, lens1, _, _, _ = model(frames, frame_lens)
                T_out = lp_fused.size(0)
                preds = batch_ctc_decode(
                    lp_fused, lens1.clamp(max=T_out),
                    blank_idx=blank_idx, mode="greedy")

                for b, pred in enumerate(preds):
                    hyp_g.append(gloss_vocab.decode(pred))
                    ref_g.append(gloss_vocab.decode(
                        gloss_gt[b, :gloss_lens[b].item()].tolist(),
                        skip_special=False))

        wer     = compute_wer(hyp_g, ref_g)
        cur_lr  = scheduler.get_last_lr()[0]
        is_best = wer < best_wer
        tag     = "★ BEST" if is_best else f"(no improve {no_improve+1}/{es_patience})"

        print(f"  [DualCSLR] E{epoch+1}/{num_epochs} | "
              f"loss={avg_loss:.4f} | WER={wer*100:.2f}% | "
              f"lr={cur_lr:.2e} | phase={phase} | {tag}")

        if is_best:
            best_wer   = wer
            no_improve = 0
            best_count += 1
            torch.save({
                "epoch": epoch + 1, "model": model.state_dict(), "best_wer": best_wer
            }, best_dir / f"best_dual_cslr_{best_count}.pth")
        else:
            no_improve += 1

        if no_improve >= es_patience:
            print(f"  [DualCSLR] Early stopping at epoch {epoch+1}. "
                  f"Best WER = {best_wer*100:.2f}%")
            break

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":     epoch + 1,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_wer":  best_wer,
                "phase":     phase,
            }, ckpt_dir / f"checkpoint_{epoch+1}.pth")

    print(f"\n[DualCSLR] Done. Best WER = {best_wer*100:.2f}%")
    return best_wer, model


# ══════════════════════════════════════════════════════════════════════════════
# SLT training loop  (reuses existing logic from MBtrain_ablation.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_slt(
    slt_model:    DualCSLTModel,
    train_loader: DataLoader,
    dev_loader:   DataLoader,
    cfg:          dict,
    device:       torch.device,
    text_vocab,
    gloss_vocab,
    ckpt_dir:     Path,
    best_dir:     Path,
):
    pad_idx = text_vocab.token2idx[text_vocab.PAD]
    bos_idx = text_vocab.token2idx[text_vocab.BOS]
    eos_idx = text_vocab.token2idx[text_vocab.EOS]
    blank_idx = cfg["cslr"]["ctc_blank_idx"]

    c          = cfg["slt"]
    num_epochs = c["num_epochs"]
    grad_acc   = c.get("grad_accumulation_steps", 4)
    grad_clip  = c.get("gradient_clip", 1.0)
    use_amp    = c.get("use_amp", True) and device.type == "cuda"
    use_oracle = cfg.get("eval", {}).get("use_oracle_gloss_in_eval", False)

    criterion = nn.CrossEntropyLoss(
        label_smoothing = c.get("label_smoothing", 0.1),
        ignore_index    = pad_idx,
    )
    trainable  = [p for p in slt_model.parameters() if p.requires_grad]
    optimizer  = AdamW(trainable, lr=c["learning_rate"],
                       weight_decay=c.get("weight_decay", 1e-4))
    n_steps    = num_epochs * len(train_loader)
    warmup_s   = c.get("warmup_steps", 500)
    scheduler  = cosine_with_warmup(
        optimizer, warmup_s, n_steps, eta_min=c.get("eta_min", 1e-7))
    scaler     = torch.amp.GradScaler("cuda") if use_amp else None

    last_ckpt, start_epoch = _find_latest_ckpt(ckpt_dir)
    best_bleu  = -1.0
    best_count = 0
    if last_ckpt:
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        slt_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_bleu   = ckpt.get("best_bleu", best_bleu)
        start_epoch = ckpt.get("epoch", start_epoch)
        print(f"[DualSLT] Resumed from {last_ckpt}")

    es_cfg      = c.get("early_stopping", {})
    es_patience = es_cfg.get("patience", 8) if es_cfg.get("enabled", True) else 9999
    no_improve  = 0

    print(f"\n[DualSLT] Training SLT for {num_epochs} epochs | "
          f"trainable params: {sum(p.numel() for p in trainable):,}")

    for epoch in range(start_epoch, num_epochs):
        slt_model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches  = 0

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"[DualSLT] E{epoch+1}/{num_epochs}", leave=False)
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

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits   = slt_model(frames, frame_lens, gloss, gloss_lens,
                                     tgt_in, tgt_lens_in)
                B, T, V  = logits.shape
                loss     = criterion(logits.reshape(B * T, V),
                                     tgt_out.reshape(B * T)) / grad_acc

            if scaler:
                scaler.scale(loss).backward()
                if (step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()
            epoch_loss += loss.item() * grad_acc
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ── Eval ────────────────────────────────────────────────────────────
        slt_model.eval()
        hyp_s, ref_s = [], []
        with torch.no_grad():
            for batch in dev_loader:
                frames     = batch["frames"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                gloss      = batch["gloss"].to(device)
                gloss_lens = batch["gloss_lens"].to(device)
                tgt        = batch["translation"]

                pred_ids = slt_model.translate(
                    frames, frame_lens,
                    gloss      = gloss if use_oracle else None,
                    gloss_lens = gloss_lens if use_oracle else None,
                    bos_idx    = bos_idx,
                    eos_idx    = eos_idx,
                    use_oracle_gloss = use_oracle,
                    blank_idx  = blank_idx,
                )
                for b in range(pred_ids.size(0)):
                    hyp_s.append(" ".join(text_vocab.decode(pred_ids[b].tolist())))
                    ref_s.append(" ".join(text_vocab.decode(tgt[b].tolist())))

        bleu_score = compute_bleu(hyp_s, ref_s)["bleu"]
        rouge_l    = compute_rouge(hyp_s, ref_s).get("rougeL", 0.0)
        meteor     = compute_meteor(hyp_s, ref_s)
        cur_lr     = scheduler.get_last_lr()[0]
        is_best    = bleu_score > best_bleu
        tag        = "★ BEST" if is_best else f"(no improve {no_improve+1}/{es_patience})"

        print(f"  [DualSLT] Ep {epoch+1:>2}/{num_epochs} | "
              f"loss={avg_loss:.4f} | BLEU-4={bleu_score:.2f} | "
              f"ROUGE-L={rouge_l:.4f} | METEOR={meteor:.4f} | "
              f"lr={cur_lr:.2e} | {tag}")

        if is_best:
            best_bleu  = bleu_score
            no_improve = 0
            best_count += 1
            torch.save({
                "epoch": epoch + 1, "model": slt_model.state_dict(),
                "best_bleu": best_bleu
            }, best_dir / f"best_dual_slt_{best_count}.pth")
        else:
            no_improve += 1

        if no_improve >= es_patience:
            print(f"  [DualSLT] Early stopping at epoch {epoch+1}. "
                  f"Best BLEU-4 = {best_bleu:.2f}")
            break

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":     epoch + 1,
                "model":     slt_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_bleu": best_bleu,
            }, ckpt_dir / f"checkpoint_{epoch+1}.pth")

    print(f"\n[DualSLT] Done. Best BLEU-4 = {best_bleu:.2f}")
    return best_bleu


# ══════════════════════════════════════════════════════════════════════════════
# Model builders
# ══════════════════════════════════════════════════════════════════════════════

def build_dual_cslr(cfg, num_classes) -> DualStreamCSLR:
    c  = cfg.get("dual_cslr", {})
    vs = cfg.get("video_swin", {})
    bl = cfg.get("bilstm", {})
    return DualStreamCSLR(
        num_classes       = num_classes,
        cnn_out_features  = cfg["cnn"]["out_features"],
        swin_out_features = vs.get("out_features", 512),
        bilstm_hidden     = bl.get("hidden_size", 512),
        bilstm_layers     = bl.get("num_layers", 2),
        bilstm_dropout    = bl.get("dropout", 0.3),
        projection_size   = bl.get("projection_size", 256),
        fusion_nhead      = c.get("fusion_nhead", 8),
        fusion_dropout    = c.get("fusion_dropout", 0.2),
        swin_backbone     = vs.get("backbone", "swin3d_t"),
        swin_pretrained   = vs.get("pretrained", True),
        swin_clip_len     = vs.get("clip_len", 16),
        swin_clip_stride  = vs.get("clip_stride", 8),
        swin_max_clips    = vs.get("max_clips_per_fwd", 4),
        blank_idx         = cfg["cslr"]["ctc_blank_idx"],
    )


def build_dual_slt(cfg, dual_cslr, gloss_vocab_size, text_vocab_size) -> DualCSLTModel:
    fc = cfg["fusion"]
    proj_dim = cfg["bilstm"]["projection_size"]

    gloss_embed = GlossEmbedding(gloss_vocab_size, fc["gloss_embed_dim"])
    late_fusion = LateFusion(
        visual_dim      = proj_dim,
        gloss_embed_dim = fc["gloss_embed_dim"],
        fused_dim       = fc["fused_dim"],
        mode            = fc["mode"],
        dropout         = fc["dropout"],
        nhead           = fc.get("nhead", 8),
    )
    t = cfg["transformer_2d"]
    translator = SLTTransformer(
        src_dim            = fc["fused_dim"],
        tgt_vocab_size     = text_vocab_size,
        d_model            = t["d_model"],
        nhead              = t["nhead"],
        num_encoder_layers = t["num_encoder_layers"],
        num_decoder_layers = t["num_decoder_layers"],
        dim_feedforward    = t["dim_feedforward"],
        dropout            = t["dropout"],
        max_seq_len        = t["max_seq_len"],
    )
    return DualCSLTModel(dual_cslr, gloss_embed, late_fusion, translator)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dual-Stream CSLR + SLT")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--stage",     choices=["cslr", "slt", "all"], default="all")
    parser.add_argument("--s1_ckpt",   default=None,
                        help="Path to cslr_variant_A.pth (ResNet34+BiLSTM checkpoint, Variant A)")
    # [FIX-2] New argument: load Variant G (VideoSwin+BiLSTM) weights into stream2
    # instead of starting stream2 from raw ImageNet-pretrained Swin weights.
    # Usage: --s2_ckpt .../checkpoints/ablation/Variant_G_path/best_path/best_path_33.pth
    parser.add_argument("--s2_ckpt",   default=None,
                        help="Path to cslr_variant_G.pth (VideoSwin+BiLSTM checkpoint, Variant G). "
                             "Loads trained Swin+BiLSTM2 weights into stream2 instead of "
                             "starting from scratch.")
    parser.add_argument("--dual_ckpt", default=None,
                        help="Pretrained dual CSLR checkpoint (for --stage slt)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    cfg["seed"] = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    gloss_vocab, text_vocab = build_vocabularies()
    print(f"Vocab  : gloss={len(gloss_vocab)} | text={len(text_vocab)}")

    # ── Directories ─────────────────────────────────────────────────────────
    base_ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "dual_stream"
    cslr_ckpt_dir  = base_ckpt / "cslr" / "checkpoints"
    cslr_best_dir  = base_ckpt / "cslr" / "best"
    slt_ckpt_dir   = base_ckpt / "slt"  / "checkpoints"
    slt_best_dir   = base_ckpt / "slt"  / "best"
    for d in [cslr_ckpt_dir, cslr_best_dir, slt_ckpt_dir, slt_best_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── CSLR stage ──────────────────────────────────────────────────────────
    if args.stage in ("cslr", "all"):
        print("\n" + "="*60)
        print(" Dual-Stream CSLR: ResNet34-2D + Video Swin-T")
        print("="*60)

        model = build_dual_cslr(cfg, num_classes=len(gloss_vocab)).to(device)

        # Load stream-1 pretrained weights
        if args.s1_ckpt:
            print(f"\nLoading stream1 from: {args.s1_ckpt}")
            model.load_stream1_from_ablation_ckpt(args.s1_ckpt, device=str(device))
        else:
            print("\n[WARN] --s1_ckpt not provided. Stream1 starts from ImageNet weights only.")

        # [FIX-2] Load stream-2 pretrained weights from Variant G checkpoint.
        # Without this, stream2 (VideoSwin+BiLSTM2) always starts from raw
        # ImageNet-pretrained Swin weights — wasting Variant G's training.
        if args.s2_ckpt:
            print(f"\nLoading stream2 (Swin+BiLSTM2) from: {args.s2_ckpt}")
            model.load_stream2_from_ablation_ckpt(args.s2_ckpt, device=str(device))
        else:
            print("\n[WARN] --s2_ckpt not provided. Stream2 starts from ImageNet Swin weights only.")
            print("       Provide --s2_ckpt pointing to Variant G checkpoint to skip retraining.")

        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Params total    : {n_total:,}")
        print(f"  Params trainable: {n_train:,}  (after stream1 load, before freeze)")

        train_loader = make_loader("train", False, cfg, gloss_vocab, text_vocab, seed=args.seed)
        dev_loader   = make_loader("dev",   False, cfg, gloss_vocab, text_vocab, seed=args.seed)

        best_wer, model = run_cslr(
            model, train_loader, dev_loader,
            cfg, device, gloss_vocab,
            cslr_ckpt_dir, cslr_best_dir,
        )

        # Save final dual CSLR checkpoint
        final_ckpt_path = base_ckpt / "best_dual_cslr_final.pth"
        torch.save({"model": model.state_dict(), "best_wer": best_wer}, final_ckpt_path)
        print(f"\n  Dual CSLR final saved → {final_ckpt_path}")
        print(f"  Best WER = {best_wer*100:.2f}%")

        del train_loader, dev_loader
        torch.cuda.empty_cache()
        gc.collect()

    # ── SLT stage ───────────────────────────────────────────────────────────
    if args.stage in ("slt", "all"):
        print("\n" + "="*60)
        print(" Dual-Stream SLT: LateFusion + Transformer-2D")
        print("="*60)

        # Build or load dual CSLR
        dual_cslr = build_dual_cslr(cfg, num_classes=len(gloss_vocab)).to(device)

        cslr_ckpt_to_load = args.dual_ckpt or str(base_ckpt / "best_dual_cslr_final.pth")
        if Path(cslr_ckpt_to_load).exists():
            ckpt = torch.load(cslr_ckpt_to_load, map_location=device, weights_only=False)
            state = ckpt["model"] if "model" in ckpt else ckpt
            dual_cslr.load_state_dict(state, strict=True)
            wer_info = f"{ckpt.get('best_wer', 0)*100:.2f}% WER" if "best_wer" in ckpt else "?"
            print(f"  Loaded dual CSLR from: {cslr_ckpt_to_load} ({wer_info})")
        else:
            print(f"  [WARN] No dual CSLR checkpoint found at {cslr_ckpt_to_load}.")
            print(f"  Starting SLT with randomly initialized CSLR (not recommended).")

        # Freeze CSLR backbone for SLT training
        for p in dual_cslr.parameters():
            p.requires_grad = False

        slt_model = build_dual_slt(
            cfg, dual_cslr, len(gloss_vocab), len(text_vocab)).to(device)

        n_train = sum(p.numel() for p in slt_model.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in slt_model.parameters() if not p.requires_grad)
        print(f"\n  Params trainable : {n_train:,}  (LateFusion + Translator)")
        print(f"  Params frozen    : {n_frozen:,}  (Dual CSLR backbone)")

        train_loader_slt = make_loader("train", True, cfg, gloss_vocab, text_vocab, seed=args.seed)
        dev_loader_slt   = make_loader("dev",   True, cfg, gloss_vocab, text_vocab, seed=args.seed)

        best_bleu = run_slt(
            slt_model, train_loader_slt, dev_loader_slt,
            cfg, device, text_vocab, gloss_vocab,
            slt_ckpt_dir, slt_best_dir,
        )

        print(f"\n  ★ SLT done. Best BLEU-4 = {best_bleu:.2f}")

    print("\n" + "="*60)
    print(" Pipeline complete.")
    print("="*60)


import gc
if __name__ == "__main__":
    main()