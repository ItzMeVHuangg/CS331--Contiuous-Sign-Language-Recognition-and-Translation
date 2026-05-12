

import sys
import json
import copy
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset      import build_vocabularies, PhoenixDataset, collate_fn, build_transforms
from models.bilstm_ctc import BiLSTM_CTC, CTCCriterion
from utils.ctc_decoder import batch_ctc_decode
from utils.metrics     import compute_wer, compute_bleu, compute_rouge, compute_meteor


# ──────────────────────────────────────────────────────────────────────────────
# Model factory — builds each variant
# ──────────────────────────────────────────────────────────────────────────────

def build_cslr_model(cfg: dict, num_classes: int, use_3d_cnn: bool) -> nn.Module:
    """Build CSLR model with either 2D or 3D CNN."""

    if use_3d_cnn:
        from models.cnn_encoder_3d import CNNEncoder3D
        cnn = CNNEncoder3D(
            backbone    = cfg.get("cnn_3d", {}).get("backbone", "r3d_18"),
            pretrained  = cfg["cnn"]["pretrained"],
            out_features= cfg["cnn"]["out_features"],
            clip_len    = cfg.get("cnn_3d", {}).get("clip_len", 16),
        )
    else:
        from models.cnn_encoder import CNNEncoder
        cnn = CNNEncoder(
            backbone    = cfg["cnn"]["backbone"],
            pretrained  = cfg["cnn"]["pretrained"],
            out_features= cfg["cnn"]["out_features"],
            freeze_bn   = cfg["cnn"]["freeze_bn"],
        )

    bilstm = BiLSTM_CTC(
        input_size      = cfg["cnn"]["out_features"],
        hidden_size     = cfg["cslr"]["hidden_size"],
        num_layers      = cfg["cslr"]["num_layers"],
        num_classes     = num_classes,
        dropout         = cfg["cslr"]["dropout"],
        projection_size = cfg["cslr"]["projection_size"],
        blank_idx       = cfg["cslr"]["ctc_blank_idx"],
    )

    class CSLRModel(nn.Module):
        def __init__(self, cnn, bilstm):
            super().__init__()
            self.cnn     = cnn
            self.bilstm  = bilstm

        def forward(self, frames, frame_lens):
            feats = self.cnn(frames)
            # 3D CNN returns fewer time steps → update frame_lens
            T_out = feats.shape[1]
            lens  = frame_lens.clamp(max=T_out)
            # Scale lens proportionally if 3D CNN reduced T
            if T_out < frame_lens.max().item():
                scale = T_out / frame_lens.float().max().item()
                lens  = (frame_lens.float() * scale).long().clamp(min=1, max=T_out)
            log_probs, hidden = self.bilstm(feats, lens)
            return log_probs, hidden, lens

    return CSLRModel(cnn, bilstm)


def build_slt_model(cfg, gloss_vocab_size, text_vocab_size,
                    cslr_model, use_3d_transformer: bool) -> nn.Module:
    """Build full CSLT model with either 2D or 3D Transformer."""

    from models.late_fusion import LateFusion, GlossEmbedding

    proj_dim  = cfg["cslr"]["projection_size"] or cfg["cslr"]["hidden_size"] * 2
    fused_dim = cfg["fusion"]["fused_dim"]

    gloss_embed  = GlossEmbedding(gloss_vocab_size, cfg["fusion"]["gloss_embed_dim"])
    late_fusion  = LateFusion(
        visual_dim      = proj_dim,
        gloss_embed_dim = cfg["fusion"]["gloss_embed_dim"],
        fused_dim       = fused_dim,
        mode            = cfg["fusion"]["mode"],
        dropout         = cfg["fusion"]["dropout"],
    )

    if use_3d_transformer:
        from models.translator_3d import SLTTransformer3D
        translator = SLTTransformer3D(
            src_dim            = fused_dim,
            tgt_vocab_size     = text_vocab_size,
            d_model            = cfg["slt"]["d_model"],
            nhead              = cfg["slt"]["nhead"],
            num_encoder_layers = cfg["slt"]["num_encoder_layers"],
            num_decoder_layers = cfg["slt"]["num_decoder_layers"],
            dim_feedforward    = cfg["slt"]["dim_feedforward"],
            dropout            = cfg["slt"]["dropout"],
            max_seq_len        = cfg["slt"].get("max_seq_len", 300),
            encoder_type       = "conv3d",
        )
    else:
        from models.translator import SLTTransformer
        translator = SLTTransformer(
            src_dim            = fused_dim,
            tgt_vocab_size     = text_vocab_size,
            d_model            = cfg["slt"]["d_model"],
            nhead              = cfg["slt"]["nhead"],
            num_encoder_layers = cfg["slt"]["num_encoder_layers"],
            num_decoder_layers = cfg["slt"]["num_decoder_layers"],
            dim_feedforward    = cfg["slt"]["dim_feedforward"],
            dropout            = cfg["slt"]["dropout"],
            max_seq_len        = cfg["slt"].get("max_seq_len", 300),
        )

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
        def translate(self, frames, frame_lens, gloss, gloss_lens, bos_idx, eos_idx):
            _, visual_hidden, adj_lens = self.cslr(frames, frame_lens)
            gloss_emb = self.gloss_embed(gloss)
            fused     = self.late_fusion(visual_hidden, gloss_emb)
            return self.translator.greedy_decode(fused, bos_idx, eos_idx, adj_lens)

    return CSLTModel()


# ──────────────────────────────────────────────────────────────────────────────
# Train + Eval helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_cslr_epochs(model, train_loader, dev_loader, cfg, device,
                    gloss_vocab, num_epochs, variant_name):
    criterion = CTCCriterion(blank_idx=cfg["cslr"]["ctc_blank_idx"])
    optimizer = AdamW(model.parameters(),
                      lr=cfg["cslr"]["learning_rate"],
                      weight_decay=cfg["cslr"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer,
                                   T_max=num_epochs * len(train_loader), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_wer = float("inf")
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"[{variant_name}] CSLR E{epoch+1}", leave=False):
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                log_probs, _, adj_lens = model(frames, frame_lens)
                loss = criterion(log_probs, gloss, adj_lens, gloss_lens)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["cslr"]["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["cslr"]["gradient_clip"])
                optimizer.step()
            scheduler.step()

        # Eval
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
                                          blank_idx=cfg["cslr"]["ctc_blank_idx"])
                for b, pred in enumerate(preds):
                    hyp_g.append(gloss_vocab.decode(pred))
                    ref_g.append(gloss_vocab.decode(
                        gloss[b, :gloss_lens[b].item()].tolist(), skip_special=False))
        wer = compute_wer(hyp_g, ref_g)
        print(f"  [{variant_name}] CSLR Epoch {epoch+1}/{num_epochs} | WER={wer*100:.2f}%")
        if wer < best_wer:
            best_wer = wer

    return best_wer, model


def run_slt_epochs(model, train_loader, dev_loader, cfg, device,
                   text_vocab, num_epochs, variant_name):
    pad_idx   = text_vocab.token2idx[text_vocab.PAD]
    bos_idx   = text_vocab.token2idx[text_vocab.BOS]
    eos_idx   = text_vocab.token2idx[text_vocab.EOS]
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["slt"]["label_smoothing"], ignore_index=pad_idx)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg["slt"]["learning_rate"],
                      weight_decay=cfg["slt"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer,
                                   T_max=num_epochs * len(train_loader), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_bleu = -1.0
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"[{variant_name}] SLT E{epoch+1}", leave=False):
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)
            tgt        = batch["translation"].to(device)
            tgt_lens   = batch["translation_lens"].to(device)

            tgt_in, tgt_out  = tgt[:, :-1], tgt[:, 1:]
            tgt_lens_in = (tgt_lens - 1).clamp(min=1)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = model(frames, frame_lens, gloss, gloss_lens, tgt_in, tgt_lens_in)
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

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

        # Eval
        model.eval()
        hyp_s, ref_s = [], []
        with torch.no_grad():
            for batch in dev_loader:
                frames     = batch["frames"].to(device)
                frame_lens = batch["frame_lens"].to(device)
                gloss      = batch["gloss"].to(device)
                gloss_lens = batch["gloss_lens"].to(device)
                tgt        = batch["translation"]
                pred_ids = model.translate(frames, frame_lens, gloss, gloss_lens, bos_idx, eos_idx)
                for b in range(pred_ids.size(0)):
                    hyp_s.append(" ".join(text_vocab.decode(pred_ids[b].tolist())))
                    ref_s.append(" ".join(text_vocab.decode(tgt[b].tolist())))

        bleu = compute_bleu(hyp_s, ref_s)["bleu"]
        print(f"  [{variant_name}] SLT Epoch {epoch+1}/{num_epochs} | BLEU={bleu:.2f}")
        if bleu > best_bleu:
            best_bleu = bleu

    return best_bleu


# ──────────────────────────────────────────────────────────────────────────────
# Ablation variants
# ──────────────────────────────────────────────────────────────────────────────

VARIANTS = {
    "A": {"use_3d_cnn": False, "use_3d_transformer": False,
          "desc": "CNN-2D (ResNet18) + Transformer-2D  [baseline]"},
    "B": {"use_3d_cnn": True,  "use_3d_transformer": False,
          "desc": "CNN-3D (R3D-18)  + Transformer-2D"},
    "C": {"use_3d_cnn": False, "use_3d_transformer": True,
          "desc": "CNN-2D (ResNet18) + Transformer-3D (Conv3D stem)"},
    "D": {"use_3d_cnn": True,  "use_3d_transformer": True,
          "desc": "CNN-3D (R3D-18)  + Transformer-3D (Conv3D stem)  [full 3D]"},
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(cfg: dict, variants_to_run: list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    gloss_vocab, text_vocab = build_vocabularies()

    # Shared dataloaders
    def make_loader(split, return_translation):
        ds = PhoenixDataset(
            split=split, gloss_vocab=gloss_vocab, text_vocab=text_vocab,
            max_frames=cfg["data"]["max_frames"],
            temporal_stride=cfg["data"]["temporal_stride"],
            transform=build_transforms(split, cfg["data"]["img_height"], cfg["data"]["img_width"]),
            return_translation=return_translation,
        )
        return DataLoader(ds, batch_size=cfg["cslr"]["batch_size"], shuffle=(split == "train"),
                          num_workers=cfg["data"]["num_workers"],
                          collate_fn=collate_fn, drop_last=(split == "train"))

    train_loader_cslr = make_loader("train", return_translation=False)
    dev_loader_cslr   = make_loader("dev",   return_translation=False)
    train_loader_slt  = make_loader("train", return_translation=True)
    dev_loader_slt    = make_loader("dev",   return_translation=True)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "ablation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    num_cslr_epochs = cfg["cslr"]["num_epochs"]
    num_slt_epochs  = cfg["slt"]["num_epochs"]

    for key in variants_to_run:
        v = VARIANTS[key]
        print(f"\n{'='*60}")
        print(f"Variant {key}: {v['desc']}")
        print(f"{'='*60}")

        # ── Stage 1: CSLR ───────────────────────────────────────────
        cslr_model = build_cslr_model(
            cfg, len(gloss_vocab), use_3d_cnn=v["use_3d_cnn"]
        ).to(device)

        n_params = sum(p.numel() for p in cslr_model.parameters() if p.requires_grad)
        print(f"CSLR params: {n_params:,}")

        best_wer, cslr_model = run_cslr_epochs(
            cslr_model, train_loader_cslr, dev_loader_cslr,
            cfg, device, gloss_vocab, num_cslr_epochs, f"Variant-{key}"
        )

        # Freeze CSLR for SLT stage
        for p in cslr_model.parameters():
            p.requires_grad = False

        # ── Stage 2: SLT ────────────────────────────────────────────
        slt_model = build_slt_model(
            cfg, len(gloss_vocab), len(text_vocab),
            cslr_model, use_3d_transformer=v["use_3d_transformer"]
        ).to(device)

        best_bleu = run_slt_epochs(
            slt_model, train_loader_slt, dev_loader_slt,
            cfg, device, text_vocab, num_slt_epochs, f"Variant-{key}"
        )

        results[key] = {
            "description": v["desc"],
            "use_3d_cnn":         v["use_3d_cnn"],
            "use_3d_transformer": v["use_3d_transformer"],
            "best_wer":  round(best_wer * 100, 2),
            "best_bleu": round(best_bleu, 2),
        }

        # Save checkpoint
        torch.save({
            "variant":    key,
            "model":      slt_model.state_dict(),
            "wer":        best_wer,
            "bleu":       best_bleu,
            "gloss_vocab": gloss_vocab,
            "text_vocab":  text_vocab,
        }, ckpt_dir / f"variant_{key}.pth")

        print(f"\n[Variant {key}] WER={best_wer*100:.2f}% | BLEU={best_bleu:.2f}")

    # ── Print summary table ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*60}")
    print(f"{'Variant':<10} {'CNN':<12} {'Transformer':<16} {'WER%':>6} {'BLEU-4':>8}")
    print("-" * 60)
    for key, r in results.items():
        cnn_label = "3D R3D-18" if r["use_3d_cnn"] else "2D ResNet18"
        tfm_label = "3D Conv3D" if r["use_3d_transformer"] else "2D Standard"
        print(f"  {key:<8} {cnn_label:<12} {tfm_label:<16} {r['best_wer']:>6.2f} {r['best_bleu']:>8.2f}")
    print(f"{'='*60}")

    # Save JSON
    results_path = ckpt_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--variant", default="all",
                        help="A | B | C | D | all")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant.upper()]
    run_ablation(cfg, variants)