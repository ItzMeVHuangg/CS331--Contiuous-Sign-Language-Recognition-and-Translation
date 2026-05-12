
import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import (
    PhoenixDataset, collate_fn, build_vocabularies, build_transforms
)
from models.cnn_encoder import CNNEncoder
from models.bilstm_ctc   import BiLSTM_CTC, CTCCriterion
from utils.ctc_decoder   import batch_ctc_decode
from utils.metrics       import compute_wer


# ──────────────────────────────────────────────────────────────────────────────
# Full CSLR Model
# ──────────────────────────────────────────────────────────────────────────────

class CSLRModel(nn.Module):
    def __init__(self, cfg: dict, num_classes: int):
        super().__init__()
        self.cnn = CNNEncoder(
            backbone     = cfg["cnn"]["backbone"],
            pretrained   = cfg["cnn"]["pretrained"],
            out_features = cfg["cnn"]["out_features"],
            freeze_bn    = cfg["cnn"]["freeze_bn"],
        )
        self.bilstm_ctc = BiLSTM_CTC(
            input_size      = cfg["cnn"]["out_features"],
            hidden_size     = cfg["cslr"]["hidden_size"],
            num_layers      = cfg["cslr"]["num_layers"],
            num_classes     = num_classes,
            dropout         = cfg["cslr"]["dropout"],
            projection_size = cfg["cslr"]["projection_size"],
            blank_idx       = cfg["cslr"]["ctc_blank_idx"],
        )

    def forward(self, frames, frame_lens):
        """
        frames    : (B, T, C, H, W)
        frame_lens: (B,)
        Returns   : log_probs (T, B, C), hidden (B, T, proj_dim)
        """
        feats = self.cnn(frames)                          # (B, T, feat_dim)
        log_probs, hidden = self.bilstm_ctc(feats, frame_lens)
        return log_probs, hidden


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_lr_scheduler(optimizer, cfg, num_steps_per_epoch):
    name = cfg["cslr"]["lr_scheduler"]
    if name == "cosine":
        T_max = (cfg["cslr"]["num_epochs"] - cfg["cslr"]["warmup_epochs"]) * num_steps_per_epoch
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def warmup_lr(optimizer, step, warmup_steps, base_lr):
    lr = base_lr * min(1.0, step / max(warmup_steps, 1))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ──────────────────────────────────────────────────────────────────────────────
# Train / Eval loops
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, cfg, epoch, writer, global_step
):
    model.train()
    total_loss = 0.0
    warmup_steps = cfg["cslr"]["warmup_epochs"] * len(loader)
    base_lr = cfg["cslr"]["learning_rate"]

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch+1}", leave=False)
    for batch in pbar:
        frames     = batch["frames"].to(device)          # (B, T, C, H, W)
        frame_lens = batch["frame_lens"].to(device)
        gloss      = batch["gloss"].to(device)           # (B, G)
        gloss_lens = batch["gloss_lens"].to(device)

        # Warmup LR
        if global_step < warmup_steps:
            warmup_lr(optimizer, global_step, warmup_steps, base_lr)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            log_probs, _ = model(frames, frame_lens)     # (T, B, C)
            T_out = log_probs.size(0)
            # CTC requires input_lengths ≤ T_out
            input_lens = frame_lens.clamp(max=T_out)
            loss = criterion(log_probs, gloss, input_lens, gloss_lens)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["cslr"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["cslr"]["gradient_clip"])
            optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if writer and global_step % 10 == 0:
            writer.add_scalar("train/ctc_loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, gloss_vocab, cfg):
    model.eval()
    total_loss = 0.0
    all_hyp, all_ref = [], []

    for batch in tqdm(loader, desc="[Eval]", leave=False):
        frames     = batch["frames"].to(device)
        frame_lens = batch["frame_lens"].to(device)
        gloss      = batch["gloss"].to(device)
        gloss_lens = batch["gloss_lens"].to(device)

        log_probs, _ = model(frames, frame_lens)
        T_out = log_probs.size(0)
        input_lens = frame_lens.clamp(max=T_out)
        loss = criterion(log_probs, gloss, input_lens, gloss_lens)
        total_loss += loss.item()

        # Decode
        preds = batch_ctc_decode(log_probs, input_lens,
                                  blank_idx=cfg["cslr"]["ctc_blank_idx"],
                                  mode="greedy")
        for b_idx, pred_ids in enumerate(preds):
            hyp_tokens = gloss_vocab.decode(pred_ids)
            ref_tokens = gloss_vocab.decode(
                gloss[b_idx, :gloss_lens[b_idx].item()].tolist(), skip_special=False
            )
            all_hyp.append(hyp_tokens)
            all_ref.append(ref_tokens)

    avg_loss = total_loss / len(loader)
    wer = compute_wer(all_hyp, all_ref)
    return avg_loss, wer


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train_cslr(cfg: dict, trial_params: dict = None):
    """
    Main CSLR training function. 
    trial_params: dict of hyperparameters (used by Optuna tuner to override cfg).
    Returns best WER (for HPO minimization).
    """
    # Override cfg with trial params if provided (HPO)
    if trial_params:
        cfg["cslr"].update(trial_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Vocabularies ────────────────────────────────────────────────
    gloss_vocab, text_vocab = build_vocabularies()

    # ── Datasets ────────────────────────────────────────────────────
    train_ds = PhoenixDataset(
        split="train",
        frames_root=cfg["paths"]["frames_root"],
        annot_root =cfg["paths"]["annot_root"],
        gloss_vocab=gloss_vocab,
        text_vocab =text_vocab,
        max_frames =cfg["data"]["max_frames"],
        temporal_stride=cfg["data"]["temporal_stride"],
        transform  =build_transforms("train", cfg["data"]["img_height"], cfg["data"]["img_width"]),
        return_translation=False,
    )
    dev_ds = PhoenixDataset(
        split="dev",
        frames_root=cfg["paths"]["frames_root"],
        annot_root =cfg["paths"]["annot_root"],
        gloss_vocab=gloss_vocab,
        text_vocab =text_vocab,
        max_frames =cfg["data"]["max_frames"],
        temporal_stride=cfg["data"]["temporal_stride"],
        transform  =build_transforms("dev", cfg["data"]["img_height"], cfg["data"]["img_width"]),
        return_translation=False,
    )

    batch_size = cfg["cslr"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=cfg["data"]["num_workers"],
                               pin_memory=cfg["data"]["pin_memory"],
                               collate_fn=collate_fn, drop_last=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=cfg["data"]["num_workers"],
                               pin_memory=cfg["data"]["pin_memory"],
                               collate_fn=collate_fn)

    # ── Model ───────────────────────────────────────────────────────
    model = CSLRModel(cfg, num_classes=len(gloss_vocab)).to(device)
    print(f"[Model] CSLR params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────
    criterion = CTCCriterion(blank_idx=cfg["cslr"]["ctc_blank_idx"])
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["cslr"]["learning_rate"],
        weight_decay=cfg["cslr"]["weight_decay"],
    )
    scheduler  = get_lr_scheduler(optimizer, cfg, len(train_loader))
    scaler     = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Checkpoint dir & Logger ─────────────────────────────────────
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "cslr"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg["paths"]["log_dir"]) / "cslr"))

    # ── Training loop ───────────────────────────────────────────────
    best_wer = float("inf")
    global_step = 0

    for epoch in range(cfg["cslr"]["num_epochs"]):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, cfg, epoch, writer, global_step
        )

        val_loss, val_wer = evaluate(model, dev_loader, criterion, device, gloss_vocab, cfg)

        # Scheduler step
        if cfg["cslr"]["lr_scheduler"] == "plateau":
            scheduler.step(val_wer)
        elif epoch >= cfg["cslr"]["warmup_epochs"]:
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch+1:3d}/{cfg['cslr']['num_epochs']}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_WER={val_wer*100:.2f}% | {elapsed:.1f}s"
        )

        writer.add_scalar("val/ctc_loss", val_loss, epoch)
        writer.add_scalar("val/wer",      val_wer,  epoch)

        # Save best checkpoint
        if val_wer < best_wer:
            best_wer = val_wer
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "wer":         best_wer,
                "gloss_vocab": gloss_vocab,
                "text_vocab":  text_vocab,
                "cfg":         cfg,
            }, ckpt_dir / "best_cslr.pth")
            print(f"  ✓ Saved best CSLR checkpoint (WER={best_wer*100:.2f}%)")

    writer.close()
    print(f"\n[CSLR] Training complete. Best WER = {best_wer*100:.2f}%")
    return best_wer


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cslr(cfg)