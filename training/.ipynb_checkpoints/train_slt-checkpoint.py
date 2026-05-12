
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import PhoenixDataset, collate_fn, build_transforms
from models.cnn_encoder import CNNEncoder
from models.bilstm_ctc  import BiLSTM_CTC
from models.late_fusion  import LateFusion, GlossEmbedding
from models.translator   import SLTTransformer
from utils.metrics       import compute_bleu, compute_rouge, compute_meteor
from training.train_cslr import CSLRModel


class CSLTModel(nn.Module):
    def __init__(self, cfg: dict, gloss_vocab_size: int, text_vocab_size: int):
        super().__init__()
        cnn_feat = cfg["cnn"]["out_features"]
        bilstm_h = cfg["cslr"]["hidden_size"]
        proj_dim = cfg["cslr"]["projection_size"] or bilstm_h * 2
        fused_dim = cfg["fusion"]["fused_dim"]

        # Frozen CSLR encoder (CNN + BiLSTM)
        self.cslr = CSLRModel(cfg, num_classes=gloss_vocab_size)

        # Gloss embedding (for late fusion)
        self.gloss_embed = GlossEmbedding(
            vocab_size  = gloss_vocab_size,
            embed_dim   = cfg["fusion"]["gloss_embed_dim"],
        )

        # Late fusion
        self.late_fusion = LateFusion(
            visual_dim      = proj_dim,
            gloss_embed_dim = cfg["fusion"]["gloss_embed_dim"],
            fused_dim       = fused_dim,
            mode            = cfg["fusion"]["mode"],
            dropout         = cfg["fusion"]["dropout"],
        )

        # Transformer translator
        self.translator = SLTTransformer(
            src_dim            = fused_dim,
            tgt_vocab_size     = text_vocab_size,
            d_model            = cfg["slt"]["d_model"],
            nhead              = cfg["slt"]["nhead"],
            num_encoder_layers = cfg["slt"]["num_encoder_layers"],
            num_decoder_layers = cfg["slt"]["num_decoder_layers"],
            dim_feedforward    = cfg["slt"]["dim_feedforward"],
            dropout            = cfg["slt"]["dropout"],
            max_seq_len        = cfg["slt"]["max_seq_len"],
        )

    def forward(
        self,
        frames,       # (B, T, C, H, W)
        frame_lens,   # (B,)
        gloss,        # (B, G) — teacher-forced gloss ids
        gloss_lens,   # (B,)
        tgt,          # (B, S) — teacher-forced translation ids
        tgt_lens,     # (B,)
    ):
        # 1. Extract visual features from frozen CSLR
        with torch.no_grad():
            _, visual_hidden = self.cslr(frames, frame_lens)   # (B, T, proj_dim)

        # 2. Embed gloss sequence
        gloss_emb = self.gloss_embed(gloss)                    # (B, G, gloss_embed_dim)

        # 3. Late fusion
        fused = self.late_fusion(visual_hidden, gloss_emb)     # (B, T, fused_dim)

        # 4. Transformer: teacher-forcing
        logits = self.translator(fused, tgt, frame_lens, tgt_lens)  # (B, S, vocab)
        return logits

    @torch.no_grad()
    def translate(self, frames, frame_lens, gloss, gloss_lens, bos_idx, eos_idx):
        """Autoregressive translation at inference."""
        _, visual_hidden = self.cslr(frames, frame_lens)
        gloss_emb = self.gloss_embed(gloss)
        fused = self.late_fusion(visual_hidden, gloss_emb)
        tokens = self.translator.greedy_decode(fused, bos_idx, eos_idx, frame_lens)
        return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Label-smoothed cross-entropy
# ──────────────────────────────────────────────────────────────────────────────

class LabelSmoothedCE(nn.Module):
    def __init__(self, vocab_size: int, label_smoothing: float = 0.1, pad_idx: int = 1):
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=pad_idx,
        )

    def forward(self, logits, targets):
        """
        logits : (B, T, vocab)
        targets: (B, T)
        """
        B, T, V = logits.shape
        return self.criterion(logits.reshape(B * T, V), targets.reshape(B * T))


# ──────────────────────────────────────────────────────────────────────────────
# Train / Eval
# ──────────────────────────────────────────────────────────────────────────────

def train_slt(cfg: dict, cslr_ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabularies from CSLR checkpoint
    ckpt = torch.load(cslr_ckpt_path, map_location="cpu", weights_only=False)
    gloss_vocab = ckpt["gloss_vocab"]
    text_vocab  = ckpt["text_vocab"]
    print(f"[Vocab] Gloss: {len(gloss_vocab)} | Text: {len(text_vocab)}")

    # Datasets
    train_ds = PhoenixDataset(
        split="train", 
        
        gloss_vocab=gloss_vocab, text_vocab=text_vocab,
        max_frames=cfg["data"]["max_frames"],
        temporal_stride=cfg["data"]["temporal_stride"],
        transform=build_transforms("train", cfg["data"]["img_height"], cfg["data"]["img_width"]),
        return_translation=True,
    )
    dev_ds = PhoenixDataset(
        split="dev", 
        
        gloss_vocab=gloss_vocab, text_vocab=text_vocab,
        max_frames=cfg["data"]["max_frames"],
        temporal_stride=cfg["data"]["temporal_stride"],
        transform=build_transforms("dev", cfg["data"]["img_height"], cfg["data"]["img_width"]),
        return_translation=True,
    )

    bs = cfg["slt"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               num_workers=cfg["data"]["num_workers"],
                               collate_fn=collate_fn, drop_last=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=bs, shuffle=False,
                               num_workers=cfg["data"]["num_workers"],
                               collate_fn=collate_fn)

    # Model
    model = CSLTModel(cfg, len(gloss_vocab), len(text_vocab)).to(device)

    # Load CSLR weights and freeze
    model.cslr.load_state_dict(ckpt["model"])
    for p in model.cslr.parameters():
        p.requires_grad = False
    print("[Model] CSLR weights loaded & frozen.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {trainable:,}")

    # Loss / Optimizer
    pad_idx   = text_vocab.token2idx[text_vocab.PAD]
    criterion = LabelSmoothedCE(
        len(text_vocab),
        label_smoothing=cfg["slt"]["label_smoothing"],
        pad_idx=pad_idx,
    )
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["slt"]["learning_rate"],
        weight_decay=cfg["slt"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg["slt"]["num_epochs"] * len(train_loader), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "slt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg["paths"]["log_dir"]) / "slt"))

    bos_idx = text_vocab.token2idx[text_vocab.BOS]
    eos_idx = text_vocab.token2idx[text_vocab.EOS]
    best_bleu = -1.0
    global_step = 0

    for epoch in range(cfg["slt"]["num_epochs"]):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[SLT Train] Epoch {epoch+1}", leave=False)
        for batch in pbar:
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)
            tgt        = batch["translation"].to(device)
            tgt_lens   = batch["translation_lens"].to(device)

            # Teacher forcing: input = tgt[:-1], label = tgt[1:]
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_lens_in = (tgt_lens - 1).clamp(min=1)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(frames, frame_lens, gloss, gloss_lens, tgt_in, tgt_lens_in)
                loss   = criterion(logits, tgt_out)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["slt"]["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["slt"]["gradient_clip"])
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % 20 == 0:
                writer.add_scalar("slt/train_loss", loss.item(), global_step)

        # ── Evaluate ─────────────────────────────────────────────────
        model.eval()
        hyp_sents, ref_sents = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="[SLT Eval]", leave=False):
                frames     = batch["frames"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                gloss      = batch["gloss"].to(device)
                gloss_lens = batch["gloss_lens"].to(device)
                tgt        = batch["translation"]

                pred_ids = model.translate(
                    frames, frame_lens, gloss, gloss_lens, bos_idx, eos_idx
                )                                                # (B, max_len)
                for b in range(pred_ids.size(0)):
                    hyp_tokens = text_vocab.decode(pred_ids[b].tolist())
                    ref_tokens = text_vocab.decode(
                        tgt[b].tolist()
                    )
                    hyp_sents.append(" ".join(hyp_tokens))
                    ref_sents.append(" ".join(ref_tokens))

        bleu   = compute_bleu(hyp_sents, ref_sents)
        rouge  = compute_rouge(hyp_sents, ref_sents)
        meteor = compute_meteor(hyp_sents, ref_sents)

        print(
            f"Epoch [{epoch+1:3d}/{cfg['slt']['num_epochs']}] "
            f"loss={total_loss/len(train_loader):.4f} | "
            f"BLEU={bleu['bleu']:.2f} | ROUGE-L={rouge['rougeL']:.4f} | "
            f"METEOR={meteor:.4f}"
        )
        writer.add_scalar("slt/bleu",   bleu["bleu"],    epoch)
        writer.add_scalar("slt/rougeL", rouge["rougeL"], epoch)
        writer.add_scalar("slt/meteor", meteor,          epoch)

        if bleu["bleu"] > best_bleu:
            best_bleu = bleu["bleu"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "bleu":  best_bleu,
                "gloss_vocab": gloss_vocab,
                "text_vocab":  text_vocab,
                "cfg":         cfg,
            }, ckpt_dir / "best_slt.pth")
            print(f"  ✓ Saved best SLT checkpoint (BLEU={best_bleu:.2f})")

    writer.close()
    print(f"\n[SLT] Done. Best BLEU = {best_bleu:.2f}")
    return best_bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--cslr_ckpt", default="checkpoints/cslr/best_cslr.pth")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_slt(cfg, args.cslr_ckpt)