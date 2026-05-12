import torch
import torch.nn as nn


class TemporalPool(nn.Module):
    """
    1D temporal downsampling between CNN encoder and sequence model.

    Standard component in CSLR literature (VAC, CorrNet, SubUNet, etc.):
    reduces T by a factor of 2^num_pool_layers before feeding into LSTM/Transformer.

    Default: num_pool_layers=2  →  factor=4
        e.g. 256 frames → 64 steps fed into BiLSTM, making CTC alignment tractable.

    Interface:
        input : (B, T, D)
        output: (B, T // factor, D)
    """

    def __init__(self, num_pool_layers: int = 2):
        super().__init__()
        self.num_pool_layers = num_pool_layers
        self.factor          = 2 ** num_pool_layers
        # Stack of MaxPool1d layers; each halves the temporal dimension.
        self.pool = nn.Sequential(
            *[nn.MaxPool1d(kernel_size=2, stride=2) for _ in range(num_pool_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T // factor, D)"""
        x = x.permute(0, 2, 1)   # (B, D, T)
        x = self.pool(x)          # (B, D, T // factor)
        x = x.permute(0, 2, 1)   # (B, T // factor, D)
        return x

    def adjust_lengths(self, lengths: torch.Tensor, T_in: int) -> torch.Tensor:
        """
        Proportionally scale sequence lengths after temporal downsampling.
        Clamps to [1, T_out] to avoid degenerate CTC inputs.
        """
        T_out = T_in // self.factor
        scale = T_out / max(T_in, 1)
        return (lengths.float() * scale).long().clamp(min=1, max=T_out)
