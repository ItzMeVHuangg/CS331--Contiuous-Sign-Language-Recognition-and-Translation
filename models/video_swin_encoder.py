import math
import torch
import torch.nn as nn
import torchvision.models.video as tv_video
from typing import Optional


class VideoSwinEncoder(nn.Module):
    

    SUPPORTED = {
        "swin3d_t": (tv_video.swin3d_t, tv_video.Swin3D_T_Weights, 768),
        "swin3d_s": (tv_video.swin3d_s, tv_video.Swin3D_S_Weights, 768),
        "swin3d_b": (tv_video.swin3d_b, tv_video.Swin3D_B_Weights, 1024),
    }

    def __init__(
        self,
        backbone:          str            = "swin3d_t",
        pretrained:        bool           = True,
        out_features:      int            = 512,
        clip_len:          int            = 16,
        clip_stride:       Optional[int]  = None,
        max_clips_per_fwd: int            = 8,  
    ):
        super().__init__()
        assert backbone in self.SUPPORTED, \
            f"backbone phải là một trong {list(self.SUPPORTED)}, nhận '{backbone}'"

        self.clip_len          = clip_len
        self.clip_stride       = clip_stride if clip_stride is not None else clip_len // 2
        self.max_clips_per_fwd = max_clips_per_fwd

        builder, weights_cls, self.raw_dim = self.SUPPORTED[backbone]
        weights = weights_cls.DEFAULT if pretrained else None
        base    = builder(weights=weights)

        base.head     = nn.Identity()
        self.backbone = base

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

    def get_output_length(self, input_len: int) -> int:
  
        if input_len <= 0:
            return 1
        return max(1, math.ceil(max(0, input_len - self.clip_len) / self.clip_stride) + 1)

    def scale_lengths(self, frame_lens: torch.Tensor, T_frames: int) -> torch.Tensor:
      
        T_prime = self.get_output_length(T_frames)
        clip_lens = torch.tensor(
            [self.get_output_length(int(fl.item())) for fl in frame_lens],
            dtype=torch.long, device=frame_lens.device,
        )
        return clip_lens.clamp(min=1, max=T_prime)

    def _extract_clips(self, x: torch.Tensor) -> list:
        B, C, T, H, W = x.shape
        positions = list(range(0, T, self.clip_stride))
        if not positions:
            positions = [0]

        clips = []
        for start in positions:
            end  = start + self.clip_len
            clip = x[:, :, start:end]
            if clip.size(2) < self.clip_len:
                pad  = x.new_zeros(B, C, self.clip_len - clip.size(2), H, W)
                clip = torch.cat([clip, pad], dim=2)
            clips.append(clip)  # (B, C, clip_len, H, W)
        return clips

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        x     = frames.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        clips = self._extract_clips(x)                        # List[(B, C, clip_len, H, W)]
        num_clips = len(clips)

        all_feats = []
        for i in range(0, num_clips, self.max_clips_per_fwd):
            mini = clips[i : i + self.max_clips_per_fwd]       # List tối đa max_clips_per_fwd clips

            mini_batch = torch.cat(mini, dim=0)
            if self.training:
                mini_batch.requires_grad_(True)
                feats = torch.utils.checkpoint.checkpoint(self.backbone, mini_batch, use_reentrant=False)
            else:
                feats = self.backbone(mini_batch)              # (len_mini*B, raw_dim)
            feats      = self.proj(feats)                       # (len_mini*B, out_features)
            feats      = feats.view(len(mini), B, self.out_features)
            all_feats.append(feats)

        out = torch.cat(all_feats, dim=0).permute(1, 0, 2).contiguous()
        return self.out_norm(out)   