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

from data.dataset       import PhoenixDataset, collate_fn, build_transforms
from models.late_fusion import LateFusion, GlossEmbedding
from models.translator  import SLTTransformer
from utils.metrics      import compute_bleu, compute_rouge, compute_meteor
from utils.ctc_decoder  import batch_ctc_decode
from training.train_cslr import CSLRModel


# ══════════════════════════════════════════════════════════════════════════════
# [F6] Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# CSLT Model — Frozen CSLR + LateFusion + SLT Transformer
# ══════════════════════════════════════════════════════════════════════════════

class CSLTModel(nn.Module):

    def __init__(self, cfg: dict, gloss_vocab_size: int, text_vocab_size: int):
        super().__init__()

        # [F1] FIX: Đọc proj_dim từ cfg["bilstm"], không phải cfg["cslr"]
        bilstm_cfg    = cfg["bilstm"]
        proj_dim      = bilstm_cfg["projection_size"]   # 256

        # [F1] FIX: Đọc SLT decoder config từ cfg["transformer_2d"]
        t2d_cfg   = cfg["transformer_2d"]
        fused_dim = cfg["fusion"]["fused_dim"]          # 512

        # ── Sub-modules ──────────────────────────────────────────────────────

        # CSLR: CNN + BiLSTM (sẽ bị freeze sau khi load weights)
        self.cslr = CSLRModel(cfg, num_classes=gloss_vocab_size)

        # Gloss embedding: (B, G) → (B, G, gloss_embed_dim)
        self.gloss_embed = GlossEmbedding(
            vocab_size  = gloss_vocab_size,
            embed_dim   = cfg["fusion"]["gloss_embed_dim"],   # 256
        )

        # Late fusion: visual × gloss → fused sequence
        self.late_fusion = LateFusion(
            visual_dim      = proj_dim,                       # 256
            gloss_embed_dim = cfg["fusion"]["gloss_embed_dim"],# 256
            fused_dim       = fused_dim,                      # 512
            mode            = cfg["fusion"]["mode"],          # "attention"
            dropout         = cfg["fusion"]["dropout"],       # 0.2
            nhead           = cfg["fusion"].get("nhead", 8),  # 8
        )

        # [F1] FIX: Đọc từ cfg["transformer_2d"]
        self.translator = SLTTransformer(
            src_dim            = fused_dim,
            tgt_vocab_size     = text_vocab_size,
            d_model            = t2d_cfg["d_model"],            # 512
            nhead              = t2d_cfg["nhead"],              # 8
            num_encoder_layers = t2d_cfg["num_encoder_layers"], # 4
            num_decoder_layers = t2d_cfg["num_decoder_layers"], # 4
            dim_feedforward    = t2d_cfg["dim_feedforward"],    # 2048
            dropout            = t2d_cfg["dropout"],            # 0.1
            max_seq_len        = t2d_cfg["max_seq_len"],        # 128
        )

        # Lưu blank_idx để dùng trong translate()
        self._blank_idx = cfg["cslr"]["ctc_blank_idx"]

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        frames:     torch.Tensor,   # (B, T, C, H, W)
        frame_lens: torch.Tensor,   # (B,)
        gloss:      torch.Tensor,   # (B, G) — oracle gloss cho teacher-forcing train
        gloss_lens: torch.Tensor,   # (B,)
        tgt:        torch.Tensor,   # (B, S) — target translation (teacher-forced)
        tgt_lens:   torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Teacher-forcing forward (chỉ dùng khi training).
        Dùng oracle gloss trong training là hợp lệ — đây là teacher forcing chuẩn.

        Returns: logits (B, S, vocab_size)
        """
        # 1. CSLR: trích xuất visual features (frozen)
        with torch.no_grad():
            _, visual_hidden = self.cslr(frames, frame_lens)  # (B, T, proj_dim)

        # 2. Embed oracle gloss sequence
        gloss_emb = self.gloss_embed(gloss)                    # (B, G, gloss_embed_dim)

        # 3. Late fusion
        fused = self.late_fusion(visual_hidden, gloss_emb)     # (B, T, fused_dim)

        # [F8] FIX: Dùng frame_lens.clamp(max=T_v) làm src_lengths
        T_v       = visual_hidden.size(1)
        adj_lens  = frame_lens.clamp(max=T_v)                  # (B,)

        # 4. Transformer encoder-decoder (teacher forcing)
        logits = self.translator(fused, tgt, adj_lens, tgt_lens)  # (B, S, vocab)
        return logits

    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def translate(
        self,
        frames:           torch.Tensor,             # (B, T, C, H, W)
        frame_lens:       torch.Tensor,             # (B,)
        bos_idx:          int,
        eos_idx:          int,
        use_oracle_gloss: bool          = False,
        oracle_gloss:     torch.Tensor  = None,     # (B, G) — chỉ cần nếu use_oracle_gloss=True
    ) -> torch.Tensor:

        # Step 1: CSLR forward — lấy log_probs và visual_hidden
        log_probs, visual_hidden = self.cslr(frames, frame_lens)
        # log_probs    : (T, B, num_classes)
        # visual_hidden: (B, T, proj_dim)

        T_v       = visual_hidden.size(1)
        adj_lens  = frame_lens.clamp(max=T_v)         # (B,) [F8]

        # Step 2: Lấy gloss sequence (predicted hoặc oracle)
        if use_oracle_gloss:
            # Upper-bound: biết trước ground truth
            assert oracle_gloss is not None, \
                "oracle_gloss phải được truyền vào khi use_oracle_gloss=True"
            effective_gloss = oracle_gloss
        else:
            # [F2] FIX: Decode predicted gloss từ CTC (inference thực tế)
            T_out          = log_probs.size(0)
            pred_gloss_ids = batch_ctc_decode(
                log_probs,
                adj_lens.clamp(max=T_out),
                blank_idx = self._blank_idx,
                mode      = "greedy",
            )
            # Pad predicted glosses thành batch tensor
            device  = frames.device
            max_g   = max(len(g) for g in pred_gloss_ids) if pred_gloss_ids else 1
            max_g   = max(max_g, 1)
            effective_gloss = torch.zeros(
                frames.size(0), max_g, dtype=torch.long, device=device
            )
            for b, pred in enumerate(pred_gloss_ids):
                if pred:
                    t = torch.tensor(pred[:max_g], dtype=torch.long, device=device)
                    effective_gloss[b, :len(t)] = t

        # Step 3: Gloss embedding → Late fusion
        gloss_emb = self.gloss_embed(effective_gloss)             # (B, G, gloss_embed_dim)
        fused     = self.late_fusion(visual_hidden, gloss_emb)    # (B, T, fused_dim)

        # Step 4: Greedy decode
        tokens = self.translator.greedy_decode(
            fused, bos_idx, eos_idx,
            src_lengths = adj_lens,    # [F8] FIX: Truyền adj_lens, không phải frame_lens
        )
        return tokens   # (B, max_len)


# ══════════════════════════════════════════════════════════════════════════════
# Label-smoothed cross-entropy
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothedCE(nn.Module):

    def __init__(self, label_smoothing: float = 0.1, pad_idx: int = 1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing = label_smoothing,
            ignore_index    = pad_idx,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, V = logits.shape
        return self.criterion(logits.reshape(B * T, V), targets.reshape(B * T))


# ══════════════════════════════════════════════════════════════════════════════
# [F4] Unified cosine-with-warmup LR schedule
# ══════════════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer:          torch.optim.Optimizer,
    num_warmup_steps:   int,
    num_training_steps: int,
    eta_min_ratio:      float = 0.0,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(eta_min_ratio, cosine)
    return LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train_slt(cfg: dict, cslr_ckpt_path: str) -> float:

    # [F6] Set seed
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SLT] Device: {device}")

    # ── Load vocabularies từ CSLR checkpoint ─────────────────────────────────
    ckpt        = torch.load(cslr_ckpt_path, map_location="cpu", weights_only=False)
    gloss_vocab = ckpt["gloss_vocab"]
    text_vocab  = ckpt["text_vocab"]
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
        return_translation = True,
    )
    dev_ds = PhoenixDataset(
        split              = "dev",
        gloss_vocab        = gloss_vocab,
        text_vocab         = text_vocab,
        max_frames         = dc["max_frames"],
        temporal_stride    = dc["temporal_stride"],
        transform          = build_transforms("dev", dc["img_height"], dc["img_width"]),
        return_translation = True,
    )

    sc = cfg["slt"]
    bs = sc["batch_size"]

    # [F6] Reproducible DataLoaders
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers     = dc["num_workers"],
        pin_memory      = dc.get("pin_memory", True),
        prefetch_factor = dc.get("prefetch_factor", 2) if dc["num_workers"] > 0 else None,
        collate_fn      = collate_fn,
        drop_last       = True,
        generator       = g,
        worker_init_fn  = _worker_init_fn,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=bs, shuffle=False,
        num_workers     = dc["num_workers"],
        pin_memory      = dc.get("pin_memory", True),
        prefetch_factor = dc.get("prefetch_factor", 2) if dc["num_workers"] > 0 else None,
        collate_fn      = collate_fn,
        worker_init_fn  = _worker_init_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CSLTModel(cfg, len(gloss_vocab), len(text_vocab)).to(device)

    # Load CSLR weights và freeze
    model.cslr.load_state_dict(ckpt["model"])
    for p in model.cslr.parameters():
        p.requires_grad = False

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p    = total_p - trainable_p
    print(f"[Model] Params total    : {total_p:,}")
    print(f"[Model] Params trainable: {trainable_p:,}")
    print(f"[Model] Params frozen   : {frozen_p:,}  (CSLR backbone)")

    # ── Loss ──────────────────────────────────────────────────────────────────
    pad_idx   = text_vocab.token2idx[text_vocab.PAD]
    criterion = LabelSmoothedCE(
        label_smoothing = sc["label_smoothing"],
        pad_idx         = pad_idx,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr           = sc["learning_rate"],
        weight_decay = sc["weight_decay"],
    )

    # [F4] Cosine-with-warmup (step-level)
    num_epochs   = sc["num_epochs"]
    steps_per_ep = len(train_loader)
    grad_acc     = sc.get("grad_accumulation_steps", 4)
    warmup_steps = sc.get("warmup_steps", 1000)
    total_steps  = num_epochs * steps_per_ep
    eta_min_ratio = sc.get("eta_min", 1e-7) / sc["learning_rate"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, eta_min_ratio
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Checkpoint & TensorBoard ──────────────────────────────────────────────
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "slt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg["paths"]["log_dir"]) / "slt"))
    best_ckpt_path = ckpt_dir / "best_slt.pth"

    # ── Special tokens ────────────────────────────────────────────────────────
    bos_idx = text_vocab.token2idx[text_vocab.BOS]
    eos_idx = text_vocab.token2idx[text_vocab.EOS]

    # ── Early Stopping state ──────────────────────────────────────────────────
    # [F5]
    es_cfg      = sc.get("early_stopping", {})
    do_es       = es_cfg.get("enabled", False)
    es_patience = es_cfg.get("patience", 8)
    no_improve  = 0

    best_bleu   = -1.0
    global_step = 0

    # [F2] Có report both oracle và predicted BLEU cho transparency
    use_oracle = cfg.get("eval", {}).get("use_oracle_gloss_in_eval", False)
    if use_oracle:
        print("[SLT] ⚠️  WARNING: use_oracle_gloss_in_eval=True — "
              "Kết quả này chỉ dùng để phân tích upper bound, KHÔNG dùng để report paper!")

    print(f"\n[SLT] Training for {num_epochs} epochs | "
          f"Effective batch = {bs * grad_acc} | "
          f"Fusion mode = {cfg['fusion']['mode']} | "
          f"Eval gloss = {'oracle' if use_oracle else 'predicted (CTC)'}")

    # ══════════════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(num_epochs):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"[SLT Train] Epoch {epoch+1}", leave=False)
        for step, batch in enumerate(pbar):
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)          # Oracle gloss (OK cho training)
            gloss_lens = batch["gloss_lens"].to(device)
            tgt        = batch["translation"].to(device)
            tgt_lens   = batch["translation_lens"].to(device)

            # Teacher forcing: encoder thấy tgt[:-1], predict tgt[1:]
            tgt_in      = tgt[:, :-1]
            tgt_out     = tgt[:, 1:]
            tgt_lens_in = (tgt_lens - 1).clamp(min=1)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(
                    frames, frame_lens,
                    gloss, gloss_lens,
                    tgt_in, tgt_lens_in,
                )
                # [F3] Chia loss cho grad_acc
                loss = criterion(logits, tgt_out) / grad_acc

            if scaler is not None:
                scaler.scale(loss).backward()
                if (step + 1) % grad_acc == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, sc["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, sc["gradient_clip"])
                    optimizer.step()
                    optimizer.zero_grad()

            # [F4] Step-level scheduler
            scheduler.step()

            total_loss  += loss.item() * grad_acc   # Log loss gốc
            global_step += 1

            if global_step % 20 == 0:
                writer.add_scalar("slt/train_loss", loss.item() * grad_acc, global_step)
                writer.add_scalar("slt/lr", optimizer.param_groups[0]["lr"], global_step)

            pbar.set_postfix(loss=f"{loss.item() * grad_acc:.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # ── Evaluate ───────────────────────────────────────────────────────
        model.eval()
        hyp_sents, ref_sents = [], []

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="[SLT Eval]", leave=False):
                frames     = batch["frames"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                gloss      = batch["gloss"].to(device)
                tgt        = batch["translation"]

                # [F2] FIX: Dùng predicted gloss (không oracle) cho eval chính thức
                pred_ids = model.translate(
                    frames, frame_lens,
                    bos_idx          = bos_idx,
                    eos_idx          = eos_idx,
                    use_oracle_gloss = use_oracle,
                    oracle_gloss     = gloss if use_oracle else None,
                )
                for b in range(pred_ids.size(0)):
                    hyp_sents.append(" ".join(text_vocab.decode(pred_ids[b].tolist())))
                    ref_sents.append(" ".join(text_vocab.decode(tgt[b].tolist())))

        bleu   = compute_bleu(hyp_sents, ref_sents)
        rouge  = compute_rouge(hyp_sents, ref_sents)
        meteor = compute_meteor(hyp_sents, ref_sents)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}] "
            f"loss={avg_train_loss:.4f} | "
            f"BLEU-4={bleu['bleu']:.2f} | "
            f"ROUGE-L={rouge['rougeL']:.4f} | "
            f"METEOR={meteor:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.0f}s"
        )

        writer.add_scalar("slt/bleu4",  bleu["bleu"],    epoch)
        writer.add_scalar("slt/bleu1",  bleu["bleu1"],   epoch)
        writer.add_scalar("slt/rougeL", rouge["rougeL"], epoch)
        writer.add_scalar("slt/meteor", meteor,          epoch)

        # ── Save best checkpoint ────────────────────────────────────────────
        if bleu["bleu"] > best_bleu:
            best_bleu  = bleu["bleu"]
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "bleu":        best_bleu,
                "bleu_scores": bleu,
                "rouge":       rouge,
                "meteor":      meteor,
                "gloss_vocab": gloss_vocab,
                "text_vocab":  text_vocab,
                "cfg":         cfg,
                "seed":        seed,
                "oracle_gloss_in_eval": use_oracle,
            }, best_ckpt_path)
            print(f"  ✓ Best checkpoint saved  (BLEU-4={best_bleu:.2f})")
        else:
            no_improve += 1
            print(f"  ✗ No improvement ({no_improve}/{es_patience})")

        # [F5] Early stopping
        if do_es and no_improve >= es_patience:
            print(f"\n[SLT] Early stopping triggered at epoch {epoch+1}.")
            break

    writer.close()

    # [F7] Restore best checkpoint
    if best_ckpt_path.exists():
        saved = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        print(f"[SLT] Best model restored (BLEU-4={best_bleu:.2f})")

    print(f"\n[SLT] Training complete. Best BLEU-4 = {best_bleu:.2f}")
    print(f"[SLT] Eval mode: {'ORACLE gloss (upper bound)' if use_oracle else 'PREDICTED gloss (real inference)'}")
    return best_bleu


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SLT Stage (LateFusion + Transformer-2D)")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--cslr_ckpt", default="checkpoints/cslr/best_cslr.pth")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_slt(cfg, args.cslr_ckpt)