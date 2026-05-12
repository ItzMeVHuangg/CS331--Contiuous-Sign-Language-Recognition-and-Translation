"""
models/mediapipe_encoder.py
────────────────────────────
MediaPipe Keypoint Encoder for CSLR.

Pre-extraction
──────────────
Run  scripts/extract_mediapipe.py  BEFORE training to convert raw frames →
keypoint .npy files:
    <mediapipe_kpts_root>/<split>/<video_id>.npy
    shape: (T, 225)  float32  — already normalised to [0, 1] in frame coords

Architecture
────────────
Input : (B, T, keypoint_dim=225)   ← from MediapipeDataset
Output: (B, T, out_features=512)   ← same interface as CNNEncoder / Swin

    Linear projection
        ↓
    TCN stack (N dilated depthwise-separable temporal conv blocks)
        ↓
    Linear projection → out_features
        ↓
    (B, T, 512)

The TCN preserves the full temporal resolution (no pooling), so the
downstream BiLSTM/TransformerCTC sees the same T frames as the 2D CNN path.
Dilation grows as 2^i per layer to cover long-range dependencies efficiently.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Dilated Depthwise-Separable Temporal Conv Block  (TCN building block)
# ──────────────────────────────────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    """
    One residual block of the Temporal Convolutional Network.

    Architecture (pre-activation):
        LayerNorm
        Depthwise Conv1d  (kernel=k, dilation=d, same padding)
        GELU
        Pointwise Conv1d  (1×1, expand channels)
        GELU
        Dropout
        Pointwise Conv1d  (1×1, restore channels)
        + residual

    Using depthwise-separable convolutions keeps param count low while
    still capturing local temporal patterns at multiple scales.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        expand_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = int(channels * expand_ratio)
        pad    = (kernel_size - 1) * dilation // 2   # "same" causal padding

        self.norm = nn.LayerNorm(channels)

        # Depthwise temporal conv
        self.dw_conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=pad,
            groups=channels,       # depthwise
        )
        # Pointwise expansion + restoration
        self.pw1  = nn.Conv1d(channels, hidden,   kernel_size=1)
        self.pw2  = nn.Conv1d(hidden,   channels, kernel_size=1)
        self.drop = nn.Dropout(p=dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        # Pre-norm (norm operates on last dim, so transpose temporarily)
        r = x
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)  # (B, C, T)
        x = self.act(self.dw_conv(x))
        x = self.act(self.pw1(x))
        x = self.drop(x)
        x = self.pw2(x)
        return x + r    # residual


# ──────────────────────────────────────────────────────────────────────────────
# MediapipeEncoder
# ──────────────────────────────────────────────────────────────────────────────

class MediapipeEncoder(nn.Module):
    """
    Temporal Convolutional Network over pre-extracted MediaPipe keypoints.

    Expects keypoints already normalised to [0,1] in frame-pixel coordinates
    and stored as (T, keypoint_dim) float32 numpy arrays on disk
    (see  scripts/extract_mediapipe.py).

    Interface matches CNNEncoder and VideoSwinEncoder:
        input:  (B, T, keypoint_dim)   — loaded by MediapipeDataset
        output: (B, T, out_features)   — same T (no temporal downsampling)

    Args:
        keypoint_dim   : raw landmark feature dimension (default 225)
        hidden_dim     : internal TCN channel size
        out_features   : output feature dimension (match other encoders → 512)
        num_tcn_layers : number of stacked TCN blocks
        tcn_kernel     : temporal kernel size in each block
        dropout        : dropout rate throughout the network
    """

    def __init__(
        self,
        keypoint_dim:   int   = 225,
        hidden_dim:     int   = 256,
        out_features:   int   = 512,
        num_tcn_layers: int   = 4,
        tcn_kernel:     int   = 3,
        dropout:        float = 0.2,
    ):
        super().__init__()
        self.keypoint_dim = keypoint_dim
        self.out_features = out_features

        # ── Input projection: keypoint_dim → hidden_dim ──────────────
        self.input_proj = nn.Sequential(
            nn.Linear(keypoint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # ── TCN stack with exponentially growing dilation ─────────────
        # dilation = 2^i → receptive field grows as O(2^N)
        # 4 layers with kernel=3: receptive field = 1 + 2*(3-1)*(1+2+4+8) = 61 frames
        # i.e. the network "sees" ~60 neighbouring frames for each position
        self.tcn_layers = nn.ModuleList([
            _TCNBlock(
                channels    = hidden_dim,
                kernel_size = tcn_kernel,
                dilation    = 2 ** i,
                expand_ratio = 2.0,
                dropout     = dropout,
            )
            for i in range(num_tcn_layers)
        ])

        # ── Output projection: hidden_dim → out_features ─────────────
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoints: (B, T, keypoint_dim)   pre-extracted landmarks

        Returns:
            (B, T, out_features)
        """
        B, T, _ = keypoints.shape

        # Project keypoints to hidden_dim
        x = self.input_proj(keypoints)      # (B, T, hidden_dim)

        # TCN expects (B, C, T) — channels first
        x = x.transpose(1, 2)              # (B, hidden_dim, T)

        for block in self.tcn_layers:
            x = block(x)                   # (B, hidden_dim, T)

        # Back to (B, T, hidden_dim)
        x = x.transpose(1, 2)

        # Project to out_features
        out = self.output_proj(x)          # (B, T, out_features)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, KD = 4, 256, 225

    model = MediapipeEncoder(
        keypoint_dim=KD,
        hidden_dim=256,
        out_features=512,
        num_tcn_layers=4,
        tcn_kernel=3,
        dropout=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MediapipeEncoder | params: {n_params:,}")

    x = torch.randn(B, T, KD, device=device)

    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    elapsed = time.time() - t0

    print(f"  Input : {list(x.shape)}")
    print(f"  Output: {list(out.shape)}")
    print(f"  Time  : {elapsed:.3f}s")

    # Receptive field sanity check
    # With 4 layers, kernel=3, dilations=[1,2,4,8]:
    # RF = 1 + (kernel-1) × sum(dilations) = 1 + 2×(1+2+4+8) = 31 frames
    print(f"\n  Receptive field ≈ {1 + 2*(1+2+4+8)} frames  (with 4 TCN layers, k=3)")