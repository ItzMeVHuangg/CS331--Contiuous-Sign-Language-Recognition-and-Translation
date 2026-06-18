"""
Ablation Variants:
    CSLR:
        A  ResNet18-2D   + BiLSTM           [baseline]
        B  ResNet18-2D   + Transformer CTC
        C  R3D-18        + BiLSTM
        D  R3D-18        + Transformer CTC
        E  MediaPipe     + BiLSTM
        F  MediaPipe     + Transformer CTC
        G  Video Swin    + BiLSTM
        H  Video Swin    + Transformer CTC
        I  ResNet18-2D   + Conformer CTC

    SLT:
        Y  best_encoder → Transformer-2D   [SLT baseline]
"""

import sys
import gc
import json
import random
import argparse
import yaml
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset       import build_vocabularies, PhoenixDataset, collate_fn, build_transforms
from models.bilstm_ctc  import BiLSTM_CTC, CTCCriterion
from utils.ctc_decoder  import batch_ctc_decode
from utils.metrics      import compute_wer, compute_bleu, compute_rouge, compute_meteor



import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed: int = 42):
    """Set seed cho toàn bộ stack: Python, NumPy, PyTorch, CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int):
    """DataLoader worker seed để reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MediapipeDataset(Dataset):

    def __init__(self, phoenix_ds: PhoenixDataset, kpts_root: str, keypoint_dim: int = 225):
        self.ds           = phoenix_ds
        self.kpts_root    = Path(kpts_root)
        self.keypoint_dim = keypoint_dim
        self.max_frames   = phoenix_ds.max_frames

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        import numpy as np
        item    = self.ds[idx]
        vid_id  = item["video_id"]
        split   = self.ds.split
        npy_path = self.kpts_root / split / f"{vid_id}.npy"

        if npy_path.exists():
            kpts = np.load(str(npy_path)).astype("float32")
            T_actual = min(kpts.shape[0], self.max_frames)
            kpts = kpts[:T_actual]

            if T_actual < self.max_frames:
                pad  = torch.zeros(self.max_frames - T_actual, self.keypoint_dim)
                kpts = torch.cat([torch.from_numpy(kpts), pad], dim=0)
            else:
                kpts = torch.from_numpy(kpts)

            item["frames"]    = kpts
            item["frame_len"] = torch.tensor(T_actual, dtype=torch.long)

        return item


def _build_encoder(cfg: dict, encoder_type: str) -> nn.Module:
    if encoder_type == "cnn_2d":
        from models.cnn_encoder import CNNEncoder
        return CNNEncoder(
            backbone     = cfg["cnn"]["backbone"],
            pretrained   = cfg["cnn"]["pretrained"],
            out_features = cfg["cnn"]["out_features"],
            freeze_bn    = cfg["cnn"]["freeze_bn"],
        )

    elif encoder_type == "cnn_3d":
        from models.cnn_encoder_3d import CNNEncoder3D
        c = cfg["cnn_3d"]
        return CNNEncoder3D(
            backbone     = c.get("backbone", "r3d_18"),
            pretrained   = c.get("pretrained", True),
            out_features = c.get("out_features", 512),
            clip_len     = c.get("clip_len", 16),
            clip_stride  = c.get("clip_stride", 8),
        )

    elif encoder_type == "video_swin":
        from models.video_swin_encoder import VideoSwinEncoder
        c = cfg.get("video_swin", {})
        return VideoSwinEncoder(
            backbone          = c.get("backbone", "swin3d_t"),
            pretrained        = c.get("pretrained", True),
            out_features      = c.get("out_features", 512),
            clip_len          = c.get("clip_len", 16),
            clip_stride       = c.get("clip_stride", 8),
            max_clips_per_fwd = c.get("max_clips_per_fwd", 8),
        )

    elif encoder_type == "mediapipe":
        from models.mediapipe_encoder import MediapipeEncoder
        c = cfg.get("mediapipe", {})
        return MediapipeEncoder(
            keypoint_dim   = c.get("keypoint_dim", 225),
            hidden_dim     = c.get("hidden_dim", 256),
            out_features   = c.get("out_features", 512),
            num_tcn_layers = c.get("num_tcn_layers", 4),
            tcn_kernel     = c.get("tcn_kernel", 3),
            dropout        = c.get("dropout", 0.2),
        )

    else:
        raise ValueError(f"Unknown encoder_type: '{encoder_type}'")


def _build_seq_model(cfg: dict, seq_type: str,
                     input_size: int, num_classes: int) -> nn.Module:
    if seq_type == "bilstm":
        c = cfg["bilstm"]
        return BiLSTM_CTC(
            input_size      = input_size,
            hidden_size     = c["hidden_size"],
            num_layers      = c["num_layers"],
            num_classes     = num_classes,
            dropout         = c["dropout"],
            projection_size = c["projection_size"],
            blank_idx       = cfg["cslr"]["ctc_blank_idx"],
        )

    elif seq_type == "transformer":
        from models.transformer_ctc import TransformerCTC
        c = cfg["transformer_ctc"]

        return TransformerCTC(
            input_size      = input_size,
            d_model         = c.get("d_model", 512),
            nhead           = c.get("nhead", 8),
            num_layers      = c.get("num_layers", 4),
            num_classes     = num_classes,
            dim_feedforward = c.get("dim_feedforward", 2048),
            dropout         = c.get("dropout", 0.3),
            projection_size = c.get("projection_size", 256),
            blank_idx       = cfg["cslr"]["ctc_blank_idx"],
        )

    elif seq_type == "conformer":
        from models.conformer_ctc import ConformerCTC
        c = cfg.get("conformer", {})
        return ConformerCTC(
            input_size          = input_size,
            d_model             = c.get("d_model", 512),
            nhead               = c.get("nhead", 8),
            num_layers          = c.get("num_layers", 6),
            num_classes         = num_classes,
            conv_kernel_size    = c.get("conv_kernel_size", 31),
            ff_expansion_factor = c.get("ff_expansion_factor", 4),
            attn_dropout        = c.get("attn_dropout", 0.1),
            conv_dropout        = c.get("conv_dropout", 0.1),
            ff_dropout          = c.get("ff_dropout", 0.1),
            projection_size     = c.get("projection_size", 256),
            blank_idx           = cfg["cslr"]["ctc_blank_idx"],
        )

    else:
        raise ValueError(f"Unknown seq_model type: '{seq_type}'")


def _get_encoder_out_features(cfg: dict, encoder_type: str) -> int:
    if encoder_type == "video_swin":
        return cfg.get("video_swin", {}).get("out_features", 512)
    elif encoder_type == "cnn_3d":
        return cfg.get("cnn_3d", {}).get("out_features", 512)
    elif encoder_type == "mediapipe":
        return cfg.get("mediapipe", {}).get("out_features", 512)
    else:
        return cfg["cnn"]["out_features"]


def _is_temporal_encoder(encoder_type: str) -> bool:
    return encoder_type in ("cnn_3d", "video_swin")


def build_cslr_model(cfg, num_classes, encoder_type, seq_model_type):
    from models.temporal_pool import TemporalPool

    encoder   = _build_encoder(cfg, encoder_type)
    feat_dim  = _get_encoder_out_features(cfg, encoder_type)
    seq_model = _build_seq_model(cfg, seq_model_type, feat_dim, num_classes)
    is_temp   = _is_temporal_encoder(encoder_type)

    use_temp_pool = (encoder_type == "cnn_2d")
    temp_pool     = TemporalPool(num_pool_layers=2) if use_temp_pool else None

    class CSLRModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder     = encoder
            self.seq_model   = seq_model
            self.is_temporal = is_temp
            if temp_pool is not None:
                self.temp_pool = temp_pool

        def forward(self, frames, frame_lens):
            if encoder_type == "cnn_2d":
                feats = self.encoder(frames, frame_lens)
            else:
                feats = self.encoder(frames)

            T_in  = frames.shape[1]
            T_out = feats.shape[1]

            if hasattr(self, "temp_pool"):
                # 2D CNN: temporal pooling
                feats = self.temp_pool(feats)               # (B, T//4, feat_dim)
                lens  = self.temp_pool.adjust_lengths(frame_lens, T_in)
            elif encoder_type == "video_swin":
                # VideoSwin
                lens = self.encoder.scale_lengths(frame_lens, T_in)
            elif self.is_temporal and T_out < frame_lens.max().item():
                # 3D CNN
                scale = T_out / frame_lens.float().max().item()
                lens  = (frame_lens.float() * scale).long().clamp(min=1, max=T_out)
            else:
                lens = frame_lens.clamp(max=T_out)

            log_probs, hidden = self.seq_model(feats, lens)
            return log_probs, hidden, lens

    return CSLRModel()


def build_slt_model(cfg, gloss_vocab_size, text_vocab_size, cslr_model, slt_type):
    from models.late_fusion import LateFusion, GlossEmbedding

    seq_model = cslr_model.seq_model
    if hasattr(seq_model, "hidden_out_dim"):
        proj_dim = seq_model.hidden_out_dim
    elif hasattr(seq_model, "projection_size") and seq_model.projection_size > 0:
        proj_dim = seq_model.projection_size
    else:
        proj_dim = cfg["bilstm"]["projection_size"]

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

    if slt_type == "transformer_2d":
        from models.translator import SLTTransformer
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
    else:
        raise ValueError(f"Unknown slt_type: '{slt_type}'")

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
        def translate(self, frames, frame_lens,
                      gloss=None, gloss_lens=None,
                      bos_idx=None, eos_idx=None,
                      use_oracle_gloss: bool = False,
                      blank_idx: int = 0):
            """
            [F1] FIX Oracle Gloss Leakage:
            - use_oracle_gloss=True  → dùng ground truth gloss (chỉ để phân tích upper bound)
            - use_oracle_gloss=False → dùng predicted gloss từ CTC decode (inference thực tế)
            """
            _, visual_hidden, adj_lens = self.cslr(frames, frame_lens)

            if use_oracle_gloss:
                effective_gloss = gloss
            else:
                log_probs, _, adj_lens_cslr = self.cslr(frames, frame_lens)
                T_out = log_probs.size(0)
                pred_gloss_ids = batch_ctc_decode(
                    log_probs,
                    adj_lens_cslr.clamp(max=T_out),
                    blank_idx=blank_idx,
                    mode="greedy",
                )
                max_g = max(len(g) for g in pred_gloss_ids) if pred_gloss_ids else 1
                max_g = max(max_g, 1)
                device = frames.device
                effective_gloss = torch.zeros(
                    frames.size(0), max_g, dtype=torch.long, device=device
                )
                for b, pred in enumerate(pred_gloss_ids):
                    if len(pred) > 0:
                        t = torch.tensor(pred[:max_g], dtype=torch.long, device=device)
                        effective_gloss[b, :len(t)] = t

            gloss_emb = self.gloss_embed(effective_gloss)
            fused     = self.late_fusion(visual_hidden, gloss_emb)

            if hasattr(self.translator, "beam_decode"):
                return self.translator.beam_decode(fused, bos_idx, eos_idx, adj_lens)
            return self.translator.greedy_decode(fused, bos_idx, eos_idx, adj_lens)

    return CSLTModel()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(eta_min, cosine)
    return LambdaLR(optimizer, lr_lambda)


def get_step_decay_scheduler(optimizer, decay_epochs, decay_rate=0.2):
    """Paper-style: multiply LR by decay_rate at each milestone epoch."""
    return MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_rate)


def _build_cslr_optimizer(model, cfg):
    c       = cfg["cslr"]
    base_lr = c["learning_rate"]
    scale   = c.get("encoder_lr_scale", 0.1)   # pretrained backbone LR = base_lr * 0.1
    sched   = c.get("lr_scheduler", "cosine")

    encoder_mod = model.encoder
    if hasattr(encoder_mod, "backbone"):
        # VideoSwin hoặc tương tự: backbone pretrained + projection học mới
        backbone_params  = list(encoder_mod.backbone.parameters())
        proj_norm_params = (
            list(encoder_mod.proj.parameters()) +
            list(encoder_mod.out_norm.parameters())
            if hasattr(encoder_mod, "out_norm") else
            list(encoder_mod.proj.parameters())
        )
        encoder_param_groups = [
            {"params": backbone_params,  "lr": base_lr * scale,        "name": "enc_backbone"},
            {"params": proj_norm_params, "lr": base_lr * scale * 5.0,  "name": "enc_proj"},
        ]
    else:
        # CNN: toàn bộ encoder dùng cùng LR scale
        encoder_param_groups = [
            {"params": list(encoder_mod.parameters()), "lr": base_lr * scale, "name": "encoder"},
        ]

    seqmodel_params = list(model.seq_model.parameters())
    param_groups = encoder_param_groups + [
        {"params": seqmodel_params, "lr": base_lr, "name": "seq_model"},
    ]

    # Paper dùng Adam cho baseline ResNet34+BiLSTM, AdamW cho các variant khác
    if sched == "step":
        optimizer = Adam(param_groups, weight_decay=c.get("weight_decay", 5e-4))
    else:
        optimizer = AdamW(param_groups, weight_decay=c.get("weight_decay", 1e-4))
    return optimizer


def _prepare_variant_dirs(base_dir, variant_key):
    variant_dir = base_dir / f"Variant_{variant_key}_path"
    ckpt_dir    = variant_dir / "checkpoint"
    best_dir    = variant_dir / "best_path"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    return variant_dir, ckpt_dir, best_dir


def _find_latest_checkpoint(ckpt_dir):
    ckpts = list(ckpt_dir.glob("checkpoint_*.pth"))
    if not ckpts:
        return None, 0

    def get_epoch(p):
        return int(p.stem.split("_")[-1])

    ckpts = sorted(ckpts, key=get_epoch)
    last_ckpt = ckpts[-1]
    return last_ckpt, get_epoch(last_ckpt)

def run_cslr_epochs(model, train_loader, dev_loader, cfg, device,
                    gloss_vocab, num_epochs, variant_name,
                    ckpt_dir, best_dir):

    c         = cfg["cslr"]
    criterion = CTCCriterion(blank_idx=c["ctc_blank_idx"])
    optimizer = _build_cslr_optimizer(model, cfg)
    scaler    = torch.amp.GradScaler("cuda") if (device.type == "cuda" and
                                                   c.get("use_amp", True)) else None

    sched_type   = c.get("lr_scheduler", "cosine")
    num_epochs_s = num_epochs

    if sched_type == "step":
        decay_epochs = c.get("lr_decay_epochs", [40, 50])
        decay_rate   = c.get("lr_decay_rate", 0.2)
        scheduler    = get_step_decay_scheduler(optimizer, decay_epochs, decay_rate)
        step_per_batch = False          # step scheduler is per-epoch
    else:
        n_steps      = num_epochs * len(train_loader)
        warmup_steps = c.get("warmup_epochs", 5) * len(train_loader)
        scheduler    = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, n_steps, eta_min=c.get("eta_min", 1e-6))
        step_per_batch = True           # cosine scheduler is per-step

    grad_acc    = c.get("grad_accumulation_steps", 1)
    start_epoch = 0
    best_wer    = float("inf")
    best_count  = 0

    last_ckpt, last_epoch = _find_latest_checkpoint(ckpt_dir)
    if last_ckpt is not None:
        print(f"  [{variant_name}] Resume from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_wer = ckpt.get("best_wer", best_wer)
        start_epoch = last_epoch

    fb_cfg        = c.get("freeze_backbone", {})
    do_freeze     = fb_cfg.get("enabled", False)
    freeze_epochs = fb_cfg.get("freeze_epochs", 10)
    if do_freeze:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"  [{variant_name}] Encoder frozen for {freeze_epochs} epochs")

    es_cfg      = c.get("early_stopping", {})
    do_es       = es_cfg.get("enabled", False)
    es_patience = es_cfg.get("patience", 15)
    no_improve  = 0

    for epoch in range(start_epoch, num_epochs):
        if do_freeze and epoch == freeze_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = _build_cslr_optimizer(model, cfg)
            if sched_type == "step":
                decay_ep   = c.get("lr_decay_epochs", [40, 50])
                decay_rate = c.get("lr_decay_rate", 0.2)
                scheduler  = get_step_decay_scheduler(optimizer, decay_ep, decay_rate)
                for _ in range(epoch):
                    scheduler.step()
            else:
                remaining  = (num_epochs - epoch) * len(train_loader)
                scheduler  = get_cosine_schedule_with_warmup(
                    optimizer, 0, remaining, eta_min=c.get("eta_min", 1e-6))
            print(f"  [{variant_name}] Encoder UNFROZEN at epoch {epoch+1}")

        model.train()
        optimizer.zero_grad()
        epoch_loss  = 0.0
        num_batches = 0

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"[{variant_name}] CSLR E{epoch+1}", leave=False)
        ):
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                log_probs, _, adj_lens = model(frames, frame_lens)
                loss = criterion(log_probs, gloss, adj_lens, gloss_lens) / grad_acc

            epoch_loss  += loss.item() * grad_acc
            num_batches += 1

            if scaler:
                scaler.scale(loss).backward()
                if (step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"])
                    optimizer.step()
                    optimizer.zero_grad()

            if step_per_batch:
                scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)

        # Step-decay scheduler advances per epoch
        if not step_per_batch:
            scheduler.step()

        # Evaluation
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

        wer = compute_wer(hyp_g, ref_g)
        print(f"  [{variant_name}] CSLR E{epoch+1}/{num_epochs} | "
              f"Loss={avg_loss:.4f} | WER={wer*100:.2f}%")

        if wer < best_wer:
            best_wer = wer
            best_count += 1
            no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "best_wer": best_wer
            }, best_dir / f"best_path_{best_count}.pth")

        else:
            no_improve += 1

        if do_es and no_improve >= es_patience:
            print(f"  [{variant_name}] Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_wer": best_wer
            }, ckpt_dir / f"checkpoint_{epoch+1}.pth")

    return best_wer, model


def run_slt_epochs(model, train_loader, dev_loader, cfg, device,
                   text_vocab, num_epochs, variant_name,
                   gloss_vocab=None, blank_idx=0,
                   ckpt_dir=None, best_dir=None):

    pad_idx = text_vocab.token2idx[text_vocab.PAD]
    bos_idx = text_vocab.token2idx[text_vocab.BOS]
    eos_idx = text_vocab.token2idx[text_vocab.EOS]
    c       = cfg["slt"]

    criterion = nn.CrossEntropyLoss(
        label_smoothing=c.get("label_smoothing", 0.1),
        ignore_index=pad_idx,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=c["learning_rate"], weight_decay=c["weight_decay"])
    n_steps   = num_epochs * len(train_loader)
    warmup_s  = c.get("warmup_steps", 1000)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_s, n_steps, eta_min=c.get("eta_min", 1e-7))
    
    start_epoch = 0
    best_bleu = -1
    best_count = 0

    last_ckpt, last_epoch = _find_latest_checkpoint(ckpt_dir)
    if last_ckpt is not None:
        print(f"[{variant_name}] Resume from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_bleu = ckpt.get("best_bleu", best_bleu)
        start_epoch = last_epoch
    
    scaler    = torch.amp.GradScaler("cuda") if (device.type == "cuda" and
                                                   c.get("use_amp", True)) else None
    grad_acc  = c.get("grad_accumulation_steps", 2)

    use_oracle = cfg.get("eval", {}).get("use_oracle_gloss_in_eval", False)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"[{variant_name}] SLT E{epoch+1}", leave=False)
        ):
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)
            tgt        = batch["translation"].to(device)
            tgt_lens   = batch["translation_lens"].to(device)

            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_lens_in     = (tgt_lens - 1).clamp(min=1)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                # Training dùng oracle gloss (teacher forcing) — đây là hợp lệ
                logits = model(frames, frame_lens, gloss, gloss_lens,
                               tgt_in, tgt_lens_in)
                B, T, V = logits.shape
                loss    = criterion(logits.reshape(B * T, V),
                                    tgt_out.reshape(B * T)) / grad_acc

            if scaler:
                scaler.scale(loss).backward()
                if (step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, c["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, c["gradient_clip"])
                    optimizer.step()
                    optimizer.zero_grad()
            scheduler.step()

        # ── Evaluation ─────────────────────────────────────────────
        model.eval()
        hyp_s, ref_s = [], []
        with torch.no_grad():
            for batch in dev_loader:
                frames     = batch["frames"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                gloss      = batch["gloss"].to(device)
                gloss_lens = batch["gloss_lens"].to(device)
                tgt        = batch["translation"]

                # [F1] FIX: Dùng predicted gloss (không phải oracle)
                pred_ids = model.translate(
                    frames, frame_lens,
                    gloss=gloss if use_oracle else None,
                    gloss_lens=gloss_lens if use_oracle else None,
                    bos_idx=bos_idx, eos_idx=eos_idx,
                    use_oracle_gloss=use_oracle,
                    blank_idx=blank_idx,
                )
                for b in range(pred_ids.size(0)):
                    hyp_s.append(" ".join(text_vocab.decode(pred_ids[b].tolist())))
                    ref_s.append(" ".join(text_vocab.decode(tgt[b].tolist())))

        bleu = compute_bleu(hyp_s, ref_s)["bleu"]
        print(f"  [{variant_name}] SLT E{epoch+1}/{num_epochs} | "
              f"BLEU-4={bleu:.2f} {'[oracle gloss]' if use_oracle else '[predicted gloss]'}")
        if bleu > best_bleu:
            best_bleu = bleu
            best_count += 1
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "best_bleu": best_bleu
            }, best_dir / f"best_path_{best_count}.pth")

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_bleu": best_bleu
            }, ckpt_dir / f"checkpoint_{epoch+1}.pth")

    return best_bleu


CSLR_VARIANTS = {
    "A": {"encoder": "cnn_2d",    "seq_model": "bilstm",
          "desc": "ResNet34-2D + BiLSTM"},
    "B": {"encoder": "cnn_2d",    "seq_model": "transformer",
          "desc": "ResNet18-2D + Transformer CTC"},
    "C": {"encoder": "cnn_3d",    "seq_model": "bilstm",
          "desc": "R3D-18 + BiLSTM"},
    "D": {"encoder": "cnn_3d",    "seq_model": "transformer",
          "desc": "R3D-18 + Transformer CTC"},
    "E": {"encoder": "mediapipe", "seq_model": "bilstm",
          "desc": "MediaPipe Keypoints + BiLSTM"},
    "F": {"encoder": "mediapipe", "seq_model": "transformer",
          "desc": "MediaPipe Keypoints + Transformer CTC"},
    "G": {"encoder": "video_swin", "seq_model": "bilstm",
          "desc": "Video Swin Transformer + BiLSTM"},
    "H": {"encoder": "video_swin", "seq_model": "transformer",
          "desc": "Video Swin Transformer + Transformer CTC"},
    "I": {"encoder": "cnn_2d", "seq_model": "conformer",
            "desc": "ResNet18-2D + Conformer CTC"},
}

SLT_VARIANTS = {
    "Y": {"slt_type": "transformer_2d",
          "desc": "best_CSLR → Transformer-2D [SLT baseline]"},
}

ALL_VARIANTS = {**CSLR_VARIANTS, **SLT_VARIANTS}


def make_loader(split, return_translation, cfg, gloss_vocab, text_vocab,
                encoder_type="cnn_2d", seed=42):
    dc    = cfg["data"]
    cslrc = cfg["cslr"]

    # Paper: temporal scale augmentation ±20% for train, read from config
    ts_cfg = dc.get("augmentation", {}).get("temporal_scale", {})
    temporal_scale_range = None
    if split == "train" and ts_cfg and encoder_type != "mediapipe":
        temporal_scale_range = (
            ts_cfg.get("min_scale", 0.8),
            ts_cfg.get("max_scale", 1.2),
        )

    clip_aug_crop = None
    if split == "train" and encoder_type != "mediapipe":
        clip_aug_crop = (dc["img_height"], dc["img_width"])

    base_ds = PhoenixDataset(
        split                = split,
        gloss_vocab          = gloss_vocab,
        text_vocab           = text_vocab,
        max_frames           = dc["max_frames"],
        temporal_stride      = dc["temporal_stride"],
        transform            = build_transforms(split, dc["img_height"], dc["img_width"]),
        return_translation   = return_translation,
        temporal_scale_range = temporal_scale_range,
        clip_aug_crop        = clip_aug_crop,
    )

    if encoder_type == "mediapipe":
        kpts_root = cfg["paths"]["mediapipe_kpts_root"]
        kd        = cfg["mediapipe"]["keypoint_dim"]
        ds        = MediapipeDataset(base_ds, kpts_root, keypoint_dim=kd)
        cf        = collate_fn
    else:
        ds = base_ds
        cf = collate_fn

    batch_size = cslrc["batch_size"] if not return_translation \
                 else cfg["slt"]["batch_size"]

    # [F3] Generator để reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    nw = dc["num_workers"]
    return DataLoader(
        ds,
        batch_size              = batch_size,
        shuffle                 = (split == "train"),
        num_workers             = nw,
        pin_memory              = False,
        prefetch_factor         = dc.get("prefetch_factor", 2) if nw > 0 else None,
        collate_fn              = cf,
        drop_last               = (split == "train"),
        generator               = g if split == "train" else None,
        worker_init_fn          = _worker_init_fn,
        persistent_workers      = (nw > 0),         
        multiprocessing_context = None,  
    )


def run_ablation(cfg, variants_to_run, best_encoder=None, best_seq=None,
                 cslr_ckpt=None, init_cslr_ckpt=None):
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    gloss_vocab, text_vocab = build_vocabularies()
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "ablation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cslr_results = {}
    slt_results  = {}
    num_cslr_ep  = cfg["cslr"]["num_epochs"]
    num_slt_ep   = cfg["slt"]["num_epochs"]

    cslr_keys = [k for k in variants_to_run if k in CSLR_VARIANTS]
    slt_keys  = [k for k in variants_to_run if k in SLT_VARIANTS]

    # ── Stage 1: CSLR ────────────────────────────────────────────────────────
    for key in cslr_keys:
        v = CSLR_VARIANTS[key]
        print(f"\n{'='*64}")
        print(f"CSLR Variant {key}: {v['desc']}")
        print(f"{'='*64}")

        train_loader = make_loader("train", False, cfg, gloss_vocab, text_vocab,
                                   encoder_type=v["encoder"], seed=seed)
        dev_loader   = make_loader("dev",   False, cfg, gloss_vocab, text_vocab,
                                   encoder_type=v["encoder"], seed=seed)

        cslr_model = build_cslr_model(
            cfg, len(gloss_vocab),
            encoder_type   = v["encoder"],
            seq_model_type = v["seq_model"],
        ).to(device)

        if init_cslr_ckpt:
            init_obj = torch.load(init_cslr_ckpt, map_location="cpu", weights_only=False)
            init_state = init_obj.get("model", init_obj) if isinstance(init_obj, dict) else init_obj
            ckpt_variant = str(init_obj.get("variant", "")).upper() if isinstance(init_obj, dict) else ""

            if ckpt_variant and ckpt_variant != key:
                print(f"  [Variant-{key}] Skip init checkpoint (ckpt variant={ckpt_variant}).")
            else:
                missing, unexpected = cslr_model.load_state_dict(init_state, strict=False)
                print(f"  [Variant-{key}] Initialized from checkpoint: {init_cslr_ckpt}")
                if missing:
                    print(f"    missing keys    : {len(missing)}")
                if unexpected:
                    print(f"    unexpected keys : {len(unexpected)}")

        n_total     = sum(p.numel() for p in cslr_model.parameters())
        n_trainable = sum(p.numel() for p in cslr_model.parameters() if p.requires_grad)
        print(f"  Params total    : {n_total:,}")
        print(f"  Params trainable: {n_trainable:,}")

        variant_dir, ckpt_subdir, best_subdir = _prepare_variant_dirs(ckpt_dir, key)

        best_wer, cslr_model = run_cslr_epochs(
            cslr_model, train_loader, dev_loader,
            cfg, device, gloss_vocab, num_cslr_ep,
            f"Variant-{key}",
            ckpt_subdir, best_subdir
        )

        torch.save({
            "variant":     key,
            "encoder":     v["encoder"],
            "seq_model":   v["seq_model"],
            "model":       cslr_model.state_dict(),
            "wer":         best_wer,
            "gloss_vocab": gloss_vocab,
            "text_vocab":  text_vocab,
            "seed":        seed,
        }, ckpt_dir / f"cslr_variant_{key}.pth")

        cslr_results[key] = {
            "description": v["desc"],
            "encoder":     v["encoder"],
            "seq_model":   v["seq_model"],
            "best_wer":    round(best_wer * 100, 2),
        }
        print(f"\n[Variant {key}] Best WER = {best_wer*100:.2f}%")

        del cslr_model, train_loader, dev_loader
        torch.cuda.empty_cache()
        gc.collect()

    # ── Stage 2: SLT ─────────────────────────────────────────────────────────
    if slt_keys:
        if cslr_ckpt:
            ckpt     = torch.load(cslr_ckpt, map_location="cpu")
            enc_type = ckpt["encoder"]
            seq_type = ckpt["seq_model"]
        elif best_encoder and best_seq:
            enc_type, seq_type, ckpt = best_encoder, best_seq, None
        elif cslr_results:
            best_key = min(cslr_results, key=lambda k: cslr_results[k]["best_wer"])
            enc_type = cslr_results[best_key]["encoder"]
            seq_type = cslr_results[best_key]["seq_model"]
            ckpt     = None
        else:
            raise ValueError("SLT variants cần --cslr_ckpt hoặc đã chạy CSLR trước.")

        base_cslr = build_cslr_model(cfg, len(gloss_vocab), enc_type, seq_type)
        if ckpt:
            base_cslr.load_state_dict(ckpt["model"])
        elif cslr_results:
            saved = torch.load(ckpt_dir / f"cslr_variant_{best_key}.pth", map_location="cpu")
            base_cslr.load_state_dict(saved["model"])

        base_cslr = base_cslr.to(device)
        for p in base_cslr.parameters():
            p.requires_grad = False

        train_loader_slt = make_loader("train", True, cfg, gloss_vocab, text_vocab,
                                       encoder_type=enc_type, seed=seed)
        dev_loader_slt   = make_loader("dev",   True, cfg, gloss_vocab, text_vocab,
                                       encoder_type=enc_type, seed=seed)

        for key in slt_keys:
            v = SLT_VARIANTS[key]
            print(f"\n{'='*64}")
            print(f"SLT Variant {key}: {v['desc']}")
            print(f"{'='*64}")

            slt_model = build_slt_model(
                cfg, len(gloss_vocab), len(text_vocab),
                base_cslr, slt_type=v["slt_type"],
            ).to(device)

            n_trainable = sum(p.numel() for p in slt_model.parameters() if p.requires_grad)
            print(f"  Params trainable: {n_trainable:,}")

            variant_dir, ckpt_subdir, best_subdir = _prepare_variant_dirs(ckpt_dir, key)

            best_bleu = run_slt_epochs(
                slt_model, train_loader_slt, dev_loader_slt,
                cfg, device, text_vocab, num_slt_ep,
                f"Variant-{key}",
                gloss_vocab=gloss_vocab,
                blank_idx=cfg["cslr"]["ctc_blank_idx"],
                ckpt_dir=ckpt_subdir,
                best_dir=best_subdir
            )

            # Extended metrics
            dev_eval_loader = make_loader("dev", True, cfg, gloss_vocab, text_vocab,
                                          encoder_type=enc_type, seed=seed)
            hyp_s, ref_s = [], []
            bos_idx = text_vocab.token2idx[text_vocab.BOS]
            eos_idx = text_vocab.token2idx[text_vocab.EOS]
            use_oracle = cfg.get("eval", {}).get("use_oracle_gloss_in_eval", False)

            slt_model.eval()
            with torch.no_grad():
                for batch in dev_eval_loader:
                    frames     = batch["frames"].to(device)
                    frame_lens = batch["frame_lens"].to(device)
                    gloss      = batch["gloss"].to(device)
                    gloss_lens = batch["gloss_lens"].to(device)
                    tgt        = batch["translation"]
                    pred_ids   = slt_model.translate(
                        frames, frame_lens,
                        gloss=gloss if use_oracle else None,
                        gloss_lens=gloss_lens if use_oracle else None,
                        bos_idx=bos_idx, eos_idx=eos_idx,
                        use_oracle_gloss=use_oracle,
                        blank_idx=cfg["cslr"]["ctc_blank_idx"],
                    )
                    for b in range(pred_ids.size(0)):
                        hyp_s.append(" ".join(text_vocab.decode(pred_ids[b].tolist())))
                        ref_s.append(" ".join(text_vocab.decode(tgt[b].tolist())))

            bleu_scores = compute_bleu(hyp_s, ref_s)
            rouge       = compute_rouge(hyp_s, ref_s)
            meteor      = compute_meteor(hyp_s, ref_s)

            torch.save({
                "variant":   key, "slt_type": v["slt_type"],
                "encoder":   enc_type, "seq_model": seq_type,
                "model":     slt_model.state_dict(), "bleu": best_bleu,
            }, ckpt_dir / f"slt_variant_{key}.pth")

            slt_results[key] = {
                "description": v["desc"],
                "slt_type":    v["slt_type"],
                "encoder":     enc_type,
                "seq_model":   seq_type,
                "best_bleu4":  round(bleu_scores["bleu"], 2),
                "bleu1":       round(bleu_scores["bleu1"], 2),
                "bleu2":       round(bleu_scores["bleu2"], 2),
                "bleu3":       round(bleu_scores.get("bleu3", 0), 2),
                # [F2] FIX: "rouge-l" → "rougeL"
                "rouge1":      round(rouge.get("rouge1", 0) * 100, 2),
                "rouge2":      round(rouge.get("rouge2", 0) * 100, 2),
                "rougeL":      round(rouge.get("rougeL", 0) * 100, 2),
                "meteor":      round(meteor * 100, 2),
                "oracle_gloss_used": use_oracle,
            }

            del slt_model
            torch.cuda.empty_cache()
            gc.collect()

        del base_cslr, train_loader_slt, dev_loader_slt
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    enc_labels = {
        "cnn_2d":     "ResNet34-2D",
        "cnn_3d":     "R3D-18",
        "mediapipe":  "MediaPipe-TCN",
        "video_swin": "Video Swin-T",
    }
    seq_labels = {
        "bilstm": "BiLSTM",
        "transformer": "Transformer CTC",
        "conformer": "Conformer CTC",
    }

    if cslr_results:
        print(f"\n{'='*72}")
        print("CSLR ABLATION RESULTS")
        print(f"{'='*72}")
        print(f"{'Var':<5} {'Encoder':<18} {'Seq Model':<22} {'WER (%)':>8}")
        print("-" * 72)
        for k, r in cslr_results.items():
            print(f"  {k:<3} {enc_labels.get(r['encoder'], r['encoder']):<18}"
                  f" {seq_labels.get(r['seq_model'], r['seq_model']):<22}"
                  f" {r['best_wer']:>8.2f}")
        print(f"{'='*72}")

    if slt_results:
        print(f"\n{'='*90}")
        print("SLT ABLATION RESULTS")
        print(f"{'='*90}")
        print(f"{'Var':<5} {'SLT Decoder':<22} {'BLEU-4':>7} {'BLEU-1':>7} "
              f"{'ROUGE-L':>8} {'METEOR':>8} {'Oracle':>7}")
        print("-" * 90)
        for k, r in slt_results.items():
            print(f"  {k:<3} {r['slt_type']:<22}"
                  f" {r['best_bleu4']:>7.2f} {r['bleu1']:>7.2f}"
                  f" {r['rougeL']:>8.2f} {r['meteor']:>8.2f}"
                  f" {'Yes' if r['oracle_gloss_used'] else 'No':>7}")
        print(f"{'='*90}")

    all_results = {"cslr": cslr_results, "slt": slt_results, "seed": seed}
    results_path = ckpt_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results → {results_path}")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSLT Ablation Study v2")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--variant",      nargs="+", default=["all_cslr"])
    parser.add_argument("--best_encoder", default=None)
    parser.add_argument("--best_seq",     default=None)
    parser.add_argument("--cslr_ckpt",   default=None)
    parser.add_argument("--init_cslr_ckpt", default=None,
                        help="Optional CSLR checkpoint to initialize model weights before training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    requested = [v.lower() for v in args.variant]
    if "all" in requested:
        variants = list(ALL_VARIANTS.keys())
    elif "all_cslr" in requested:
        variants = list(CSLR_VARIANTS.keys())
    elif "all_slt" in requested:
        variants = list(SLT_VARIANTS.keys())
    else:
        variants = [v.upper() for v in args.variant]
        unknown  = [v for v in variants if v not in ALL_VARIANTS]
        if unknown:
            parser.error(f"Unknown variants: {unknown}.")

    run_ablation(cfg, variants,
                 best_encoder=args.best_encoder,
                 best_seq=args.best_seq,
                 cslr_ckpt=args.cslr_ckpt,
                 init_cslr_ckpt=args.init_cslr_ckpt)
