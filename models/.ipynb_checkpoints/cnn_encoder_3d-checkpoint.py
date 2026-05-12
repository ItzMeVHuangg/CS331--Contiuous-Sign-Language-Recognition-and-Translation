

import torch
import torch.nn as nn
import torchvision.models.video as tv_video
from typing import Optional


class CNNEncoder3D(nn.Module):

    SUPPORTED = {
        "r3d_18":      (tv_video.r3d_18,      512),
        "r2plus1d_18": (tv_video.r2plus1d_18, 512),
        "mc3_18":      (tv_video.mc3_18,      512),
    }

    def __init__(
        self,
        backbone: str = "r3d_18",
        pretrained: bool = True,
        out_features: int = 512,
        clip_len: int = 16,
        clip_stride: Optional[int] = None,
    ):
        super().__init__()
        assert backbone in self.SUPPORTED, \
            f"backbone must be one of {list(self.SUPPORTED)}, got '{backbone}'"

        self.backbone_name = backbone
        self.clip_len      = clip_len
        self.clip_stride   = clip_stride if clip_stride is not None else clip_len

        builder, self.raw_dim = self.SUPPORTED[backbone]

        # ── Build 3D backbone ────────────────────────────────────────
        weights = "DEFAULT" if pretrained else None
        base = builder(weights=weights)

        # Remove the final FC layer, keep spatial-temporal pooling
        # Output after pool: (B, raw_dim, 1, 1, 1)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        # ── Optional projection ──────────────────────────────────────
        if out_features != self.raw_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.raw_dim, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
            )
        else:
            self.proj = nn.Identity()

        self.out_features = out_features

    # ------------------------------------------------------------------
    def _extract_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip: (B, C, clip_len, H, W)
        returns: (B, out_features)
        """
        feat = self.feature_extractor(clip)   # (B, raw_dim, 1, 1, 1)
        feat = feat.flatten(1)                # (B, raw_dim)
        feat = self.proj(feat)                # (B, out_features)
        return feat

    # ------------------------------------------------------------------
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, C, H, W)
        returns: (B, T', out_features)
                 where T' = number of clips = ceil(T / clip_stride)

        Each output vector represents one temporal clip (clip_len frames).
        BiLSTM downstream will model relationships between clips.
        """
        B, T, C, H, W = frames.shape

        # Rearrange to (B, C, T, H, W) for 3D conv
        x = frames.permute(0, 2, 1, 3, 4).contiguous()   # (B, C, T, H, W)

        clip_features = []
        positions = range(0, T, self.clip_stride)
        if len(positions) == 0:
            positions = [0]

        for start in positions:
            end  = start + self.clip_len
            clip = x[:, :, start:end]                     # (B, C, clip_len*, H, W)

            # Pad last clip if shorter than clip_len
            if clip.size(2) < self.clip_len:
                pad  = torch.zeros(
                    B, C, self.clip_len - clip.size(2), H, W,
                    device=frames.device, dtype=frames.dtype
                )
                clip = torch.cat([clip, pad], dim=2)      # (B, C, clip_len, H, W)

            feat = self._extract_clip(clip)               # (B, out_features)
            clip_features.append(feat.unsqueeze(1))       # (B, 1, out_features)

        # Stack all clip features → (B, T', out_features)
        return torch.cat(clip_features, dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 2, 64, 3, 112, 112

    for backbone in ["r3d_18", "r2plus1d_18", "mc3_18"]:
        model = CNNEncoder3D(
            backbone=backbone, pretrained=False,
            out_features=512, clip_len=16
        ).to(device)

        x = torch.randn(B, T, C, H, W, device=device)
        t0 = time.time()
        out = model(x)
        elapsed = time.time() - t0

        T_prime = out.shape[1]
        print(f"{backbone:15s} | input: {list(x.shape)} → output: {list(out.shape)} | {elapsed:.2f}s")
        del model