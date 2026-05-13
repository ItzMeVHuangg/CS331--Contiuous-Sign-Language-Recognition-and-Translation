# -*- coding: utf-8 -*-
"""
models/dual_encoder_fusion.py
------------------------------
Fuses hidden states from two separate CSLR encoders (e.g., Video Swin-T and
ResNet34) before feeding into the SLT pipeline.

The key insight: ResNet34 captures 2D frame-level spatial features while
Video Swin-T captures 3D spatio-temporal patterns. Combining both yields
richer visual representations for sign language translation.

Modes:
    "gate"   -- Learned sigmoid gating: out = g * enc1 + (1 - g) * enc2
    "concat" -- Concatenate + project: out = W * [enc1; enc2]
    "add"    -- Simple addition with projection: out = W * (enc1 + enc2)
"""

import torch
import torch.nn as nn
from typing import Literal, Optional


class DualEncoderFusion(nn.Module):
    """
    Fuses visual hidden states from two CSLR encoders.

    Both encoders produce (B, T, D) hidden states. Since the two encoders
    may produce different temporal lengths (e.g., Swin clips vs. ResNet34
    frame-level + TemporalPool), we interpolate the shorter sequence to
    match the longer one before fusion.

    Parameters
    ----------
    dim1 : int
        Hidden dimension of encoder 1 (primary, e.g., Swin).
    dim2 : int
        Hidden dimension of encoder 2 (secondary, e.g., ResNet34).
    fused_dim : int
        Output dimension after fusion.
    mode : str
        Fusion strategy: "gate", "concat", or "add".
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        dim1:      int,
        dim2:      int,
        fused_dim: int   = 512,
        mode:      Literal["gate", "concat", "add"] = "gate",
        dropout:   float = 0.15,
    ):
        super().__init__()
        self.mode      = mode
        self.fused_dim = fused_dim

        # Project both encoder outputs to same dimension
        self.proj1 = nn.Sequential(
            nn.Linear(dim1, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(dim2, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if mode == "gate":
            # Gating network: takes concatenated features -> sigmoid gate
            self.gate_net = nn.Sequential(
                nn.Linear(fused_dim * 2, fused_dim),
                nn.Sigmoid(),
            )
        elif mode == "concat":
            self.out_proj = nn.Sequential(
                nn.Linear(fused_dim * 2, fused_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        # mode == "add" needs no extra parameters

        self.layer_norm = nn.LayerNorm(fused_dim)

    @staticmethod
    def _align_temporal(feat1: torch.Tensor, feat2: torch.Tensor,
                        len1: Optional[torch.Tensor] = None,
                        len2: Optional[torch.Tensor] = None):
        """
        Align temporal dimensions of two feature sequences.
        Uses the minimum length to avoid hallucinated features.
        Returns: (feat1_aligned, feat2_aligned, aligned_lens)
        """
        T1 = feat1.shape[1]
        T2 = feat2.shape[1]

        if T1 == T2:
            out_lens = len1 if len1 is not None else len2
            return feat1, feat2, out_lens

        # Interpolate to the shorter length (avoid introducing noise)
        T_target = min(T1, T2)

        if T1 != T_target:
            # feat1 is longer -> interpolate down
            feat1 = torch.nn.functional.interpolate(
                feat1.transpose(1, 2), size=T_target, mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        if T2 != T_target:
            # feat2 is longer -> interpolate down
            feat2 = torch.nn.functional.interpolate(
                feat2.transpose(1, 2), size=T_target, mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # Adjust lengths proportionally
        if len1 is not None:
            out_lens = (len1.float() * (T_target / T1)).long().clamp(min=1, max=T_target)
        elif len2 is not None:
            out_lens = (len2.float() * (T_target / T2)).long().clamp(min=1, max=T_target)
        else:
            out_lens = None

        return feat1, feat2, out_lens

    def forward(
        self,
        hidden1: torch.Tensor,              # (B, T1, dim1) -- primary encoder
        hidden2: torch.Tensor,              # (B, T2, dim2) -- secondary encoder
        lens1:   Optional[torch.Tensor] = None,
        lens2:   Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Returns:
            fused: (B, T, fused_dim) -- fused visual features
            fused_lens: (B,) -- sequence lengths after fusion
        """
        # Project to common dimension
        h1 = self.proj1(hidden1)            # (B, T1, fused_dim)
        h2 = self.proj2(hidden2)            # (B, T2, fused_dim)

        # Align temporal dimensions
        h1, h2, fused_lens = self._align_temporal(h1, h2, lens1, lens2)

        if self.mode == "gate":
            cat = torch.cat([h1, h2], dim=-1)   # (B, T, fused_dim*2)
            g   = self.gate_net(cat)              # (B, T, fused_dim) in [0,1]
            fused = g * h1 + (1.0 - g) * h2

        elif self.mode == "concat":
            cat = torch.cat([h1, h2], dim=-1)
            fused = self.out_proj(cat)

        elif self.mode == "add":
            fused = h1 + h2

        else:
            raise ValueError(f"Unknown dual fusion mode: {self.mode!r}")

        return self.layer_norm(fused), fused_lens
