"""
training/extract_swin_features.py
───────────────────────────────────
CHẠY MỘT LẦN trước khi train stream1/stream3.

Extract Swin-T pooled features → lưu disk dưới dạng .npy.
Stream1 sẽ load features này (chỉ train TAPE + CTC) → từ 4h/epoch → ~10 phút/epoch.

Usage:
    python training/extract_swin_features.py --config configs/config1.yaml

Output:
    <swin_cache_root>/train/<video_id>.npy   shape (T', 768) float32
    <swin_cache_root>/dev/<video_id>.npy
    <swin_cache_root>/test/<video_id>.npy

Thời gian: ~20-40 phút / split trên A100 (chỉ chạy 1 lần).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.models.video as tv_video
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import (
    PhoenixDataset,
    build_transforms,
    build_vocabularies,
    collate_fn,
)


# ─── Frozen Swin-T extractor ──────────────────────────────────────────────────

class SwinTExtractor(torch.nn.Module):
    """Frozen Swin-T: (B, T, C, H, W) → pooled (B, T', 768). Inference only."""

    OUT_DIM = 768  # Swin-T stage-4 output dim

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_video.Swin3D_T_Weights.DEFAULT if pretrained else None
        swin = tv_video.swin3d_t(weights=weights)

        self.patch_embed = swin.patch_embed
        self.pos_drop    = swin.pos_drop
        self.features    = swin.features  # Sequential of stages + merges
        self.norm        = swin.norm
        del swin

        for p in self.parameters():          # Freeze toàn bộ
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) → (B, T', 768) trên CPU."""
        x = frames.permute(0, 2, 1, 3, 4).contiguous()   # (B, C, T, H, W)
        x = self.patch_embed(x)                           # (B, T', H', W', 96) NHWC
        x = self.pos_drop(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)                                  # (B, T', Hp, Wp, 768)
        x = x.mean(dim=(2, 3))                            # (B, T', 768) spatial pool
        return x.cpu().float()


# ─── Per-split extraction ─────────────────────────────────────────────────────

def extract_split(split, cfg, model, cache_root, device, batch_size=4, num_workers=4):
    out_dir = cache_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    dc = cfg["data"]
    gloss_vocab, text_vocab = build_vocabularies()

    ds = PhoenixDataset(
        split=split,
        gloss_vocab=gloss_vocab,
        text_vocab=text_vocab,
        max_frames=dc["max_frames"],
        temporal_stride=dc["temporal_stride"],
        transform=build_transforms(split, dc["img_height"], dc["img_width"]),
        return_translation=False,
    )

    # Custom collate: giữ lại video_id + frames + frame_lens
    def extract_collate(batch):
        return {
            "video_id"  : [b["video_id"]  for b in batch],
            "frames"    : torch.stack([b["frames"]    for b in batch]),
            "frame_lens": torch.stack([b["frame_len"] for b in batch]),  # ← "frame_len" không có 's'
        }

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=extract_collate,
        persistent_workers=(num_workers > 0),
    )

    model.eval()
    n_skip = 0
    for batch in tqdm(loader, desc=f"  [{split}]"):
        frames     = batch["frames"].to(device)
        frame_lens = batch["frame_lens"]
        video_ids  = batch["video_id"]          # list[str] — hoạt động vì custom collate

        feats = model(frames)                   # (B, T', 768) on cpu

        T_in    = frames.shape[1]
        T_prime = feats.shape[1]
        scale   = T_prime / max(T_in, 1)

        for i, vid in enumerate(video_ids):
            out_path = out_dir / f"{vid}.npy"
            if out_path.exists():
                n_skip += 1
                continue
            actual = int(round(frame_lens[i].item() * scale))
            actual = max(1, min(actual, T_prime))
            np.save(str(out_path), feats[i, :actual].numpy())

    if n_skip:
        print(f"    {n_skip} files skipped (already exist)")
    print(f"    Saved to: {out_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--splits",  nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_root = Path(
        cfg["paths"].get("swin_cache_root",
            str(Path(cfg["paths"]["phoenix_root"]) / "swin_t_features"))
    )
    print(f"[extract] Output dir : {cache_root}")
    print(f"[extract] Splits     : {args.splits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] Device     : {device}")
    if torch.cuda.is_available():
        print(f"[extract] GPU        : {torch.cuda.get_device_name(0)}")

    model = SwinTExtractor(pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[extract] Swin-T params: {n_params:,} (frozen)")
    print()

    for split in args.splits:
        print(f"Extracting split: {split}")
        t0 = time.time()
        extract_split(
            split, cfg, model, cache_root, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        elapsed = (time.time() - t0) / 60
        print(f"  Done in {elapsed:.1f} min\n")

    print("[extract] All splits done. Features ready for stream1 training.")


if __name__ == "__main__":
    main()