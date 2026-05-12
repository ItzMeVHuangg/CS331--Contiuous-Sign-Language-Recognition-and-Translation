import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BiLSTM_CTC(nn.Module):
    """
    Bidirectional LSTM with CTC head for CSLR.

    Interface:
        input : features (B, T, input_size), lengths (B,)
        output: log_probs (T, B, C), hidden (B, T, hidden_out_dim)

    Fix log:
        - hidden output now masked to actual sequence lengths so LateFusion
          does not attend over padding positions.
        - Replaced ReLU in projection with GELU (smoother gradient).
    """

    def __init__(
        self,
        input_size:      int,
        hidden_size:     int,
        num_layers:      int,
        num_classes:     int,
        dropout:         float = 0.3,
        projection_size: int   = 256,
        blank_idx:       int   = 0,
    ):
        super().__init__()
        self.blank_idx = blank_idx

        self.bilstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2

        if projection_size > 0:
            self.projection = nn.Sequential(
                nn.Linear(lstm_out_dim, projection_size),
                nn.GELU(),
                nn.Dropout(p=dropout),
            )
            ctc_in_dim = projection_size
        else:
            self.projection = nn.Identity()
            ctc_in_dim = lstm_out_dim

        self.projection_size = projection_size
        self.hidden_out_dim  = ctc_in_dim

        self.ctc_head  = nn.Linear(ctc_in_dim, num_classes)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        features: torch.Tensor,   # (B, T, input_size)
        lengths:  torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs : (T, B, num_classes)
            hidden    : (B, T, hidden_out_dim)  — padding positions are zero
        """
        packed   = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, out_lengths = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)                 # (B, T_max, 2H)

        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        hidden = self.projection(lstm_out)              # (B, T_max, ctc_in_dim)

        # ── Mask padding positions to zero ───────────────────────────
        # Prevents LateFusion from using garbage values at padded steps
        T_max = hidden.size(1)
        mask  = (torch.arange(T_max, device=lengths.device)
                 .unsqueeze(0) < lengths.unsqueeze(1))  # (B, T_max)
        hidden = hidden * mask.unsqueeze(-1).float()

        logits    = self.ctc_head(hidden)               # (B, T_max, C)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)          # (T_max, B, C)

        return log_probs, hidden


# ──────────────────────────────────────────────────────────────────────────────

class CTCCriterion(nn.Module):
    def __init__(self, blank_idx: int = 0, reduction: str = "mean",
                 zero_infinity: bool = True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank_idx, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
