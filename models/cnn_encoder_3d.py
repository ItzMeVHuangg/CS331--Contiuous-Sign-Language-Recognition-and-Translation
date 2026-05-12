import torch
import torch.nn as nn
import torchvision.models.video as tv_video
from typing import Optional


class CNNEncoder3D(nn.Module):
    """
    3-D CNN clip encoder (R3D-18 / R2Plus1D-18) for CSLR.

    Interface:
        input : (B, T, C, H, W)
        output: (B, T', out_features)   T' = number of clips

    Fix log:
        - Default clip_stride changed to clip_len//2 (50% overlap) to avoid
          information loss at clip boundaries when stride == clip_len.
        - Added output LayerNorm for consistent scale with other encoders.
    """

    SUPPORTED = {
        "r3d_18":      (tv_video.r3d_18,      512),
        "r2plus1d_18": (tv_video.r2plus1d_18, 512),
        "mc3_18":      (tv_video.mc3_18,       512),
    }

    def __init__(
        self,
        backbone:     str   = "r3d_18",
        pretrained:   bool  = True,
        out_features: int   = 512,
        clip_len:     int   = 16,
        clip_stride:  Optional[int] = None,
    ):
        super().__init__()
        assert backbone in self.SUPPORTED, \
            f"backbone must be one of {list(self.SUPPORTED)}, got '{backbone}'"

        self.clip_len    = clip_len
        # 50% overlap by default — preserves boundary information
        self.clip_stride = clip_stride if clip_stride is not None else clip_len // 2

        builder, self.raw_dim = self.SUPPORTED[backbone]
        weights = "DEFAULT" if pretrained else None
        base    = builder(weights=weights)

        # Remove FC; output after pool: (B, raw_dim, 1, 1, 1)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        if out_features != self.raw_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.raw_dim, out_features),
                nn.GELU(),
                nn.Dropout(p=0.1),
            )
        else:
            self.proj = nn.Identity()

        self.out_norm     = nn.LayerNorm(out_features)
        self.out_features = out_features

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)

        positions = list(range(0, T, self.clip_stride))
        if not positions:
            positions = [0]
        num_clips = len(positions)

        clip_list = []
        for start in positions:
            end  = start + self.clip_len
            clip = x[:, :, start:end]
            if clip.size(2) < self.clip_len:
                pad  = torch.zeros(
                    B, C, self.clip_len - clip.size(2), H, W,
                    device=frames.device, dtype=frames.dtype,
                )
                clip = torch.cat([clip, pad], dim=2)
            clip_list.append(clip)                        # (B, C, clip_len, H, W)

        clip_batch = torch.cat(clip_list, dim=0)          # (B*num_clips, C, clip_len, H, W)
        feat = self.feature_extractor(clip_batch)         # (B*num_clips, raw_dim, 1, 1, 1)
        feat = feat.flatten(1)                            # (B*num_clips, raw_dim)
        feat = self.proj(feat)                            # (B*num_clips, out_features)

        out = feat.view(B, num_clips, self.out_features)  # (B, T', out_features)
        return self.out_norm(out)