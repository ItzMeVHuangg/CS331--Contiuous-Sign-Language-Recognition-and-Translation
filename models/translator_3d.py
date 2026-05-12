import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class Conv3DStemEncoder(nn.Module):
    """
    Lightweight temporal Conv3D stem followed by Transformer Encoder.

    The depthwise Conv3D captures local motion patterns (short-range),
    while the Transformer handles long-range temporal dependencies.

    Fix log:
        - BatchNorm3d replaced with GroupNorm(32) — BN with small batches
          (or batch_size=1 at inference) produces NaN/unstable outputs.
          GroupNorm is batch-size-independent and more robust.
        - enable_nested_tensor=False added to suppress UserWarning.
    """

    def __init__(
        self,
        d_model:         int,
        nhead:           int,
        num_layers:      int,
        dim_feedforward: int   = 2048,
        dropout:         float = 0.1,
        temporal_kernel: int   = 3,
    ):
        super().__init__()

        pad = temporal_kernel // 2
        num_groups = min(32, d_model)   # GroupNorm: ≤32 groups, divisor of d_model

        self.temporal_conv = nn.Sequential(
            # Depthwise temporal conv
            nn.Conv3d(d_model, d_model,
                      kernel_size=(temporal_kernel, 1, 1),
                      padding=(pad, 0, 0),
                      groups=d_model),
            nn.GroupNorm(num_groups, d_model),   # ← was BatchNorm3d
            nn.GELU(),
            # Pointwise mix
            nn.Conv3d(d_model, d_model, kernel_size=1),
            nn.GroupNorm(num_groups, d_model),   # ← was BatchNorm3d
        )
        self.stem_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, D = x.shape
        x_3d  = x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, D, T, 1, 1)
        x_3d  = self.temporal_conv(x_3d)
        x_out = x_3d.squeeze(-1).squeeze(-1).permute(0, 2, 1)    # (B, T, D)
        x     = self.stem_norm(x + x_out)                         # residual
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class SLTTransformer3D(nn.Module):
    """
    Sign Language Translation Transformer with Conv3D-stem encoder.

    The Conv3D stem gives the encoder explicit local temporal inductive bias,
    which helps when input features already come from clip-level encoders
    (3D CNN / Swin) and temporal resolution is reduced.

    Fix log:
        - Removed broken SwinWrapper (did not call any Swin ops — was a no-op).
          Only encoder_type="conv3d" is supported and meaningful here.
        - max_seq_len 300 → 128 (PHOENIX translations are short).
        - src scaled by √d_model to match tgt_embedding magnitude.
        - enable_nested_tensor=False added.
    """

    def __init__(
        self,
        src_dim:            int,
        tgt_vocab_size:     int,
        d_model:            int   = 512,
        nhead:              int   = 8,
        num_encoder_layers: int   = 4,
        num_decoder_layers: int   = 4,
        dim_feedforward:    int   = 2048,
        dropout:            float = 0.1,
        max_seq_len:        int   = 128,
        pad_idx:            int   = 1,
        encoder_type:       str   = "conv3d",
        temporal_kernel:    int   = 3,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx     = pad_idx
        self._scale      = math.sqrt(d_model)

        if encoder_type != "conv3d":
            raise ValueError(
                f"encoder_type must be 'conv3d', got '{encoder_type}'. "
                "The 'swin' option was removed — it was a no-op wrapper."
            )

        self.src_proj = nn.Linear(src_dim, d_model)
        self.encoder  = Conv3DStemEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            temporal_kernel=temporal_kernel)

        self.pos_encoding  = PositionalEncoding(d_model, max_len=max_seq_len + 16, dropout=dropout)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.output_proj         = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    def encode(self, src, src_key_padding_mask=None):
        src = self.src_proj(src) * self._scale
        src = self.pos_encoding(src)
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.tgt_embedding(tgt) * self._scale
        tgt_emb = self.pos_encoding(tgt_emb)
        out = self.transformer_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None):
        B, T_v, _ = src.shape
        T_tgt     = tgt.shape[1]
        src_pad   = self._make_padding_mask(src_lengths, T_v)   if src_lengths is not None else None
        tgt_pad   = self._make_padding_mask(tgt_lengths, T_tgt) if tgt_lengths is not None else None
        tgt_causal = nn.Transformer.generate_square_subsequent_mask(T_tgt, device=src.device)
        memory = self.encode(src, src_pad)
        return self.decode(tgt, memory, tgt_causal, tgt_pad, src_pad)

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx, eos_idx, src_lengths=None):
        B, device = src.size(0), src.device
        src_pad  = self._make_padding_mask(src_lengths, src.size(1)) if src_lengths is not None else None
        memory   = self.encode(src, src_pad)
        ys       = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(self.max_seq_len):
            T = ys.size(1)
            causal = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            logits = self.decode(ys, memory, causal, memory_key_padding_mask=src_pad)
            next_t = logits[:, -1, :].argmax(dim=-1)
            finished |= next_t == eos_idx
            ys = torch.cat([ys, next_t.unsqueeze(1)], dim=1)
            if finished.all():
                break
        return ys[:, 1:]
