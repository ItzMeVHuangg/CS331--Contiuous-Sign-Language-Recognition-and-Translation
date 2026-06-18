import random
import sys
import argparse
import math
import time
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset   import PhoenixDataset, collate_fn, build_vocabularies, build_transforms
from models.cnn_encoder import CNNEncoder
from models.bilstm_ctc  import BiLSTM_CTC, CTCCriterion
from utils.ctc_decoder  import batch_ctc_decode
from utils.metrics      import compute_wer

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class CSLRModel(nn.Module):

    def __init__(self, cfg: dict, num_classes: int):
        super().__init__()

        # Visual encoder — ResNet18, output: (B, T, cnn_out_features)
        self.cnn = CNNEncoder(
            backbone     = cfg["cnn"]["backbone"],
            pretrained   = cfg["cnn"]["pretrained"],
            out_features = cfg["cnn"]["out_features"],
            freeze_bn    = cfg["cnn"]["freeze_bn"],
        )

        # [F1] FIX: Đọc từ cfg["bilstm"], không phải cfg["cslr"]
        bilstm_cfg = cfg["bilstm"]
        self.bilstm_ctc = BiLSTM_CTC(
            input_size      = cfg["cnn"]["out_features"],   # (B, T, 512)
            hidden_size     = bilstm_cfg["hidden_size"],    # 512
            num_layers      = bilstm_cfg["num_layers"],     # 2
            num_classes     = num_classes,
            dropout         = bilstm_cfg["dropout"],        # 0.3
            projection_size = bilstm_cfg["projection_size"],# 256
            blank_idx       = cfg["cslr"]["ctc_blank_idx"], # 0
        )

    def forward(
        self,
        frames:     torch.Tensor,   # (B, T, C, H, W)
        frame_lens: torch.Tensor,   # (B,)
    ):

        feats = self.cnn(frames)                           # (B, T, 512)
        log_probs, hidden = self.bilstm_ctc(feats, frame_lens)
        return log_probs, hidden

def get_cosine_schedule_with_warmup(
    optimizer:           torch.optim.Optimizer,
    num_warmup_steps:    int,
    num_training_steps:  int,
    eta_min_ratio:       float = 0.0,   # eta_min = eta_min_ratio × base_lr
) -> LambdaLR:

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup: 0 → 1.0
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay: 1.0 → eta_min_ratio
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(eta_min_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)

def build_optimizer(model: CSLRModel, cfg: dict) -> AdamW:

    c        = cfg["cslr"]
    base_lr  = c["learning_rate"]
    scale    = c.get("encoder_lr_scale", 0.1)

    encoder_params  = list(model.cnn.parameters())
    seqmodel_params = list(model.bilstm_ctc.parameters())

    param_groups = [
        {"params": encoder_params,  "lr": base_lr * scale, "name": "cnn_encoder"},
        {"params": seqmodel_params, "lr": base_lr,          "name": "bilstm_ctc"},
    ]
    return AdamW(param_groups, weight_decay=c["weight_decay"])


def train_one_epoch(
    model:       CSLRModel,
    loader:      DataLoader,
    optimizer:   AdamW,
    scheduler:   LambdaLR,
    criterion:   CTCCriterion,
    scaler,                     # torch.amp.GradScaler | None
    device:      torch.device,
    cfg:         dict,
    epoch:       int,
    writer:      SummaryWriter,
    global_step: int,
) -> tuple:
    """
    Returns: (avg_loss, global_step)

    [F2] LR được quản lý hoàn toàn bởi scheduler (step-level).
         Không còn manual warmup_lr() — tránh LR spike / double-apply.
    [F3] Hỗ trợ gradient accumulation: loss chia cho grad_acc trước backward.
    """
    model.train()

    c        = cfg["cslr"]
    grad_acc = c.get("grad_accumulation_steps", 1)
    epoch_loss  = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"[CSLR Train] Epoch {epoch+1}", leave=False)
    for step, batch in enumerate(pbar):
        frames     = batch["frames"].to(device)     # (B, T, C, H, W)
        frame_lens = batch["frame_lens"].to(device) # (B,)
        gloss      = batch["gloss"].to(device)      # (B, G)
        gloss_lens = batch["gloss_lens"].to(device) # (B,)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            log_probs, _ = model(frames, frame_lens)  # (T, B, C)
            T_out      = log_probs.size(0)
            input_lens = frame_lens.clamp(max=T_out)  # CTC: input_len ≤ T_out
            loss = criterion(log_probs, gloss, input_lens, gloss_lens)
            # [F3] Chia loss cho grad_acc (effective batch = grad_acc × batch_size)
            loss_scaled = loss / grad_acc

        if scaler is not None:
            scaler.scale(loss_scaled).backward()
            if (step + 1) % grad_acc == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), c["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_scaled.backward()
            if (step + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), c["gradient_clip"])
                optimizer.step()
                optimizer.zero_grad()

        # [F2] Scheduler step ở mức step (không phải epoch)
        scheduler.step()

        epoch_loss  += loss.item()  # Log loss gốc (chưa chia grad_acc)
        num_batches += 1
        global_step += 1

        if writer and global_step % 10 == 0:
            writer.add_scalar("cslr/train_loss", loss.item(), global_step)
            writer.add_scalar("cslr/lr_encoder",
                              optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("cslr/lr_seq_model",
                              optimizer.param_groups[1]["lr"], global_step)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{optimizer.param_groups[1]['lr']:.2e}")

    return epoch_loss / max(num_batches, 1), global_step


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model:       CSLRModel,
    loader:      DataLoader,
    criterion:   CTCCriterion,
    device:      torch.device,
    gloss_vocab,
    cfg:         dict,
) -> tuple:
    """
    Returns: (avg_ctc_loss, wer)
    """
    model.eval()
    total_loss = 0.0
    all_hyp, all_ref = [], []

    for batch in tqdm(loader, desc="[CSLR Eval]", leave=False):
        frames     = batch["frames"].to(device)
        frame_lens = batch["frame_lens"].to(device)
        gloss      = batch["gloss"].to(device)
        gloss_lens = batch["gloss_lens"].to(device)

        log_probs, _ = model(frames, frame_lens)    # (T, B, C)
        T_out      = log_probs.size(0)
        input_lens = frame_lens.clamp(max=T_out)
        loss       = criterion(log_probs, gloss, input_lens, gloss_lens)
        total_loss += loss.item()

        # Greedy CTC decode
        preds = batch_ctc_decode(
            log_probs, input_lens,
            blank_idx = cfg["cslr"]["ctc_blank_idx"],
            mode      = "greedy",
        )
        for b_idx, pred_ids in enumerate(preds):
            hyp_tokens = gloss_vocab.decode(pred_ids)
            ref_tokens = gloss_vocab.decode(
                gloss[b_idx, :gloss_lens[b_idx].item()].tolist(),
                skip_special=False,
            )
            all_hyp.append(hyp_tokens)
            all_ref.append(ref_tokens)

    avg_loss = total_loss / len(loader)
    wer      = compute_wer(all_hyp, all_ref)
    return avg_loss, wer


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train_cslr(cfg: dict, trial_params: dict = None) -> float:
    """
    Main CSLR training function (ResNet18-2D + BiLSTM baseline).

    Args:
        cfg         : config dict (từ config.yaml)
        trial_params: dict hyperparams từ Optuna (override cfg["cslr"])

    Returns:
        best_wer (float) — để Optuna minimize
    """
    # HPO override
    if trial_params:
        cfg["cslr"].update(trial_params)

    # [F6] Set seed
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CSLR] Device : {device}")
    if device.type == "cuda":
        print(f"[CSLR] GPU    : {torch.cuda.get_device_name(0)}")
        print(f"[CSLR] VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Vocabularies ──────────────────────────────────────────────────────────
    gloss_vocab, text_vocab = build_vocabularies()
    print(f"[Vocab] Gloss: {len(gloss_vocab)} | Text: {len(text_vocab)}")

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    dc = cfg["data"]
    train_ds = PhoenixDataset(
        split              = "train",
        gloss_vocab        = gloss_vocab,
        text_vocab         = text_vocab,
        max_frames         = dc["max_frames"],
        temporal_stride    = dc["temporal_stride"],
        transform          = build_transforms("train", dc["img_height"], dc["img_width"]),
        return_translation = False,
    )
    dev_ds = PhoenixDataset(
        split              = "dev",
        gloss_vocab        = gloss_vocab,
        text_vocab         = text_vocab,
        max_frames         = dc["max_frames"],
        temporal_stride    = dc["temporal_stride"],
        transform          = build_transforms("dev", dc["img_height"], dc["img_width"]),
        return_translation = False,
    )

    batch_size = cfg["cslr"]["batch_size"]

    # [F6] Generator để reproducible shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers    = dc["num_workers"],
        pin_memory     = dc.get("pin_memory", True),
        prefetch_factor= dc.get("prefetch_factor", 2) if dc["num_workers"] > 0 else None,
        collate_fn     = collate_fn,
        drop_last      = True,
        generator      = g,
        worker_init_fn = _worker_init_fn,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers    = dc["num_workers"],
        pin_memory     = dc.get("pin_memory", True),
        prefetch_factor= dc.get("prefetch_factor", 2) if dc["num_workers"] > 0 else None,
        collate_fn     = collate_fn,
        worker_init_fn = _worker_init_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CSLRModel(cfg, num_classes=len(gloss_vocab)).to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params total    : {total_p:,}")
    print(f"[Model] Params trainable: {trainable_p:,}")
    print(f"[Model] CNN backbone    : {cfg['cnn']['backbone']}")
    print(f"[Model] BiLSTM hidden   : {cfg['bilstm']['hidden_size']} × 2 = "
          f"{cfg['bilstm']['hidden_size']*2}")
    print(f"[Model] Projection dim  : {cfg['bilstm']['projection_size']}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = CTCCriterion(blank_idx=cfg["cslr"]["ctc_blank_idx"])

    # [F4] Differential LR optimizer
    optimizer = build_optimizer(model, cfg)

    # [F2] Unified cosine-with-warmup scheduler (step-level)
    c            = cfg["cslr"]
    num_epochs   = c["num_epochs"]
    steps_per_ep = len(train_loader)
    grad_acc     = c.get("grad_accumulation_steps", 1)
    warmup_steps = c.get("warmup_epochs", 5) * steps_per_ep
    total_steps  = num_epochs * steps_per_ep
    # eta_min_ratio: dùng eta_min tuyệt đối → chuyển về ratio so với base_lr
    eta_min_ratio = c.get("eta_min", 1e-6) / c["learning_rate"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, eta_min_ratio
    )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Checkpoint & TensorBoard ──────────────────────────────────────────────
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "cslr"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg["paths"]["log_dir"]) / "cslr"))
    best_ckpt_path = ckpt_dir / "best_cslr.pth"

    # ── Early Stopping state ──────────────────────────────────────────────────
    # [F5]
    es_cfg      = c.get("early_stopping", {})
    do_es       = es_cfg.get("enabled", False)
    es_patience = es_cfg.get("patience", 15)
    no_improve  = 0

    best_wer    = float("inf")
    global_step = 0

    print(f"\n[CSLR] Training for {num_epochs} epochs | "
          f"Effective batch = {batch_size * grad_acc} | "
          f"Warmup = {c.get('warmup_epochs', 5)} epochs")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device, cfg, epoch, writer, global_step,
        )

        val_loss, val_wer = evaluate(model, dev_loader, criterion, device, gloss_vocab, cfg)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"WER={val_wer*100:.2f}% | "
            f"lr={optimizer.param_groups[1]['lr']:.2e} | "
            f"{elapsed:.0f}s"
        )

        writer.add_scalar("cslr/val_loss", val_loss, epoch)
        writer.add_scalar("cslr/val_wer",  val_wer,  epoch)

        # ── Save best checkpoint ─────────────────────────────────────────────
        if val_wer < best_wer:
            best_wer   = val_wer
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "wer":         best_wer,
                "gloss_vocab": gloss_vocab,
                "text_vocab":  text_vocab,
                "cfg":         cfg,
                "seed":        seed,
            }, best_ckpt_path)
            print(f"  ✓ Best checkpoint saved  (WER={best_wer*100:.2f}%)")
        else:
            no_improve += 1
            print(f"  ✗ No improvement ({no_improve}/{es_patience})")

        # [F5] Early stopping
        if do_es and no_improve >= es_patience:
            print(f"\n[CSLR] Early stopping triggered at epoch {epoch+1}.")
            break

    writer.close()

    # [F7] Restore best checkpoint sau khi training xong
    if best_ckpt_path.exists():
        saved = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        print(f"[CSLR] Best model restored (WER={best_wer*100:.2f}%)")

    print(f"\n[CSLR] Training complete. Best WER = {best_wer*100:.2f}%")
    return best_wer


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSLR Baseline (ResNet18 + BiLSTM)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cslr(cfg)