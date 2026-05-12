import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeedForwardModule(nn.Module):
    """Macaron-style feed-forward module."""

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y, _ = self.mha(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout(y)


class ConvolutionModule(nn.Module):
    """Conformer convolution module (depthwise separable conv + GLU)."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "conv_kernel_size must be odd for same padding"

        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y = self.norm(x).transpose(1, 2)                # (B, D, T)
        y = self.pointwise_conv1(y)                     # (B, 2D, T)
        y = F.glu(y, dim=1)                             # (B, D, T)
        y = self.depthwise_conv(y)                      # (B, D, T)
        y = self.batch_norm(y)
        y = F.silu(y)
        y = self.pointwise_conv2(y)                     # (B, D, T)
        y = self.dropout(y)
        return y.transpose(1, 2)                        # (B, T, D)


class ConformerBlock(nn.Module):
    """Conformer block with Macaron FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        ff_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ff_expansion_factor, ff_dropout)
        self.self_attn = MultiHeadSelfAttentionModule(d_model, nhead, attn_dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, conv_dropout)
        self.ffn2 = FeedForwardModule(d_model, ff_expansion_factor, ff_dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.self_attn(x, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_norm(x)
        return x


class ConformerCTC(nn.Module):
    """
    Conformer encoder with CTC head for CSLR.

    Drop-in replacement for BiLSTM_CTC / TransformerCTC.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        num_classes: int = 1000,
        conv_kernel_size: int = 31,
        ff_expansion_factor: int = 4,
        attn_dropout: float = 0.1,
        conv_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        projection_size: int = 256,
        blank_idx: int = 0,
    ):
        super().__init__()
        self.blank_idx = blank_idx
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.GELU(),
            nn.Dropout(ff_dropout),
        )
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=512, dropout=ff_dropout)

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                nhead=nhead,
                ff_expansion_factor=ff_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                ff_dropout=ff_dropout,
                attn_dropout=attn_dropout,
                conv_dropout=conv_dropout,
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(ff_dropout)

        if projection_size > 0:
            self.projection = nn.Sequential(
                nn.Linear(d_model, projection_size),
                nn.GELU(),
                nn.Dropout(ff_dropout),
            )
            ctc_in_dim = projection_size
        else:
            self.projection = nn.Identity()
            ctc_in_dim = d_model

        self.projection_size = projection_size
        self.hidden_out_dim = ctc_in_dim
        self.ctc_head = nn.Linear(ctc_in_dim, num_classes)

    @staticmethod
    def _make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    def forward(
        self,
        features: torch.Tensor,    # (B, T, input_size)
        lengths: torch.Tensor,     # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs : (T, B, num_classes)
            hidden    : (B, T, hidden_out_dim)  — padding zeroed
        """
        _, T, _ = features.shape

        x = self.input_proj(features)
        x = self.pos_encoding(x)

        key_padding_mask = self._make_padding_mask(lengths, T)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.dropout(x)
        hidden = self.projection(x)

        valid_mask = (
            torch.arange(T, device=lengths.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        hidden = hidden * valid_mask.unsqueeze(-1).float()

        logits = self.ctc_head(hidden)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

        return log_probs, hidden
