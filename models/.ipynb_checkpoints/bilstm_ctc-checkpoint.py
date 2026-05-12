
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BiLSTM_CTC(nn.Module):


    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        projection_size: int = 256,
        blank_idx: int = 0,
    ):
        super().__init__()
        self.blank_idx = blank_idx

        # ── BiLSTM ──────────────────────────────────────────────────
        self.bilstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2   # bidirectional

        # ── Optional projection / bottleneck ────────────────────────
        if projection_size > 0:
            self.projection = nn.Sequential(
                nn.Linear(lstm_out_dim, projection_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            ctc_in_dim = projection_size
        else:
            self.projection = nn.Identity()
            ctc_in_dim = lstm_out_dim

        self.projection_size = projection_size
        self.hidden_out_dim  = ctc_in_dim

        # ── CTC head ────────────────────────────────────────────────
        self.ctc_head = nn.Linear(ctc_in_dim, num_classes)

        # ── Layer norm for stability ─────────────────────────────────
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout    = nn.Dropout(p=dropout)

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        lengths:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
     
        # Pack padded sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )                                                 # (B, T, 2*H)

        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # Projection / bottleneck
        hidden = self.projection(lstm_out)                # (B, T, ctc_in_dim)

        # CTC logits → log_softmax
        logits    = self.ctc_head(hidden)                 # (B, T, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)         # (B, T, num_classes)

        # CTCLoss expects (T, B, C)
        log_probs = log_probs.permute(1, 0, 2)            # (T, B, num_classes)

        return log_probs, hidden


# ──────────────────────────────────────────────────────────────────────────────
# CTC Loss wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CTCCriterion(nn.Module):

    def __init__(self, blank_idx: int = 0, reduction: str = "mean", zero_infinity: bool = True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank        = blank_idx,
            reduction    = reduction,
            zero_infinity = zero_infinity,
        )

    def forward(
        self,
        log_probs:    torch.Tensor,   # (T, B, C)
        targets:      torch.Tensor,   # (B, max_gloss_len)
        input_lengths: torch.Tensor,  # (B,) — actual T per sample
        target_lengths: torch.Tensor, # (B,) — actual gloss len per sample
    ) -> torch.Tensor:
        # CTCLoss expects targets as 1D concatenated or 2D
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)