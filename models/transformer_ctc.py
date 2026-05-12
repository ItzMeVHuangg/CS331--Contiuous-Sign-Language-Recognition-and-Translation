import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerCTC(nn.Module):
    """
    Transformer Encoder with CTC head for CSLR.

    Drop-in replacement for BiLSTM_CTC with identical interface.

    Fix log:
        - max_len reduced from 1024 → 512 (PHOENIX max_frames=256, headroom 2x).
        - enable_nested_tensor=False added explicitly to suppress UserWarning.
        - Output hidden is masked like BiLSTM_CTC for consistent LateFusion.
        - ReLU → GELU in projection.
    """

    def __init__(
        self,
        input_size:      int,
        d_model:         int   = 512,
        nhead:           int   = 8,
        num_layers:      int   = 4,
        num_classes:     int   = 1000,
        dim_feedforward: int   = 2048,
        dropout:         float = 0.3,
        projection_size: int   = 256,
        blank_idx:       int   = 0,
    ):
        super().__init__()
        self.blank_idx = blank_idx
        self.d_model   = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self.pos_encoding = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # Pre-LN: more stable training
        )
        # enable_nested_tensor=False suppresses the harmless UserWarning that
        # arises when norm_first=True (nested tensor path is disabled anyway)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.layer_norm    = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(p=dropout)

        if projection_size > 0:
            self.projection = nn.Sequential(
                nn.Linear(d_model, projection_size),
                nn.GELU(),
                nn.Dropout(p=dropout),
            )
            ctc_in_dim = projection_size
        else:
            self.projection = nn.Identity()
            ctc_in_dim = d_model

        self.projection_size = projection_size
        self.hidden_out_dim  = ctc_in_dim
        self.ctc_head        = nn.Linear(ctc_in_dim, num_classes)

    def _make_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)   # True where padded

    def forward(
        self,
        features: torch.Tensor,   # (B, T, input_size)
        lengths:  torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs : (T, B, num_classes)
            hidden    : (B, T, hidden_out_dim)  — padding zeroed
        """
        B, T, _ = features.shape

        x = self.input_proj(features)                       # (B, T, d_model)
        x = self.pos_encoding(x)

        src_key_padding_mask = self._make_padding_mask(lengths, T)

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)
        x = self.dropout_layer(x)

        hidden = self.projection(x)                         # (B, T, ctc_in_dim)

        # Zero padding positions (mirrors BiLSTM_CTC behaviour)
        mask   = (torch.arange(T, device=lengths.device).unsqueeze(0)
                  < lengths.unsqueeze(1))
        hidden = hidden * mask.unsqueeze(-1).float()

        logits    = self.ctc_head(hidden)                   # (B, T, C)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)              # (T, B, C)

        return log_probs, hidden
