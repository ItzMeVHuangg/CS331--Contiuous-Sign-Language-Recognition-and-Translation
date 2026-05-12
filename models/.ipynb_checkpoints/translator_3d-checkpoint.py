
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding (same as translator.py)
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────────────
# 3D Conv Stem Encoder
# Lightweight temporal modeling before Transformer attention
# ──────────────────────────────────────────────────────────────────────────────

class Conv3DStemEncoder(nn.Module):
    """
    Adds a 3D temporal convolution stem before standard Transformer layers.

    The stem captures local temporal patterns (short-range motion) that the
    global self-attention then integrates into long-range dependencies.

    Architecture:
        input (B, T, d_model)
          → reshape to (B, d_model, T, 1, 1)
          → 3D Conv (temporal kernel=3)
          → reshape back to (B, T, d_model)
          → Transformer Encoder layers
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        temporal_kernel: int = 3,
    ):
        super().__init__()

        # ── 3D temporal conv stem ────────────────────────────────────
        padding = temporal_kernel // 2
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(
                d_model, d_model,
                kernel_size=(temporal_kernel, 1, 1),
                padding=(padding, 0, 0),
                groups=d_model,         # depthwise → efficient
            ),
            nn.BatchNorm3d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv3d(d_model, d_model, kernel_size=1),   # pointwise
            nn.BatchNorm3d(d_model),
        )
        self.stem_norm = nn.LayerNorm(d_model)

        # ── Standard Transformer Encoder ─────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,                              # (B, T, d_model)
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # ── 3D conv stem: temporal feature extraction ────────────────
        # Reshape: (B, T, D) → (B, D, T, 1, 1)
        x_3d  = x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)   # (B, D, T, 1, 1)
        x_3d  = self.temporal_conv(x_3d)                          # (B, D, T, 1, 1)
        x_out = x_3d.squeeze(-1).squeeze(-1).permute(0, 2, 1)     # (B, T, D)

        # Residual + norm
        x = self.stem_norm(x + x_out)

        # ── Transformer attention: long-range dependencies ───────────
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return x   # (B, T, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# SLT Transformer 3D — full model
# ──────────────────────────────────────────────────────────────────────────────

class SLTTransformer3D(nn.Module):
    """
    Sign Language Translation Transformer with 3D-aware encoder.

    Drop-in replacement for SLTTransformer (translator.py).
    Only the encoder is changed; the decoder is identical.

    Args:
        encoder_type : "conv3d" (lightweight) | "swin" (heavy, needs timm)
        src_dim      : input feature dimension from LateFusion
        tgt_vocab_size: target vocabulary size
        d_model      : transformer hidden dimension
        ...          : same as SLTTransformer
    """

    def __init__(
        self,
        src_dim: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 300,
        pad_idx: int = 1,
        encoder_type: str = "conv3d",   # "conv3d" | "swin"
        temporal_kernel: int = 3,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx     = pad_idx
        self.encoder_type = encoder_type

        # ── Project fused features to d_model ───────────────────────
        self.src_proj = nn.Linear(src_dim, d_model)

        # ── 3D-aware Encoder ─────────────────────────────────────────
        if encoder_type == "conv3d":
            self.encoder = Conv3DStemEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                temporal_kernel=temporal_kernel,
            )
        elif encoder_type == "swin":
            self.encoder = self._build_swin_encoder(d_model)
        else:
            raise ValueError(f"encoder_type must be 'conv3d' or 'swin', got '{encoder_type}'")

        # ── Positional encoding ───────────────────────────────────────
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len + 10, dropout=dropout)

        # ── Decoder (identical to translator.py) ─────────────────────
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ── Output projection ─────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    # ------------------------------------------------------------------
    def _build_swin_encoder(self, d_model: int):
        """Video Swin Transformer encoder (requires timm)."""
        try:
            import timm
        except ImportError:
            raise ImportError("Install timm for Swin encoder: pip install timm")

        class SwinWrapper(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.swin = timm.create_model(
                    "swin_small_patch4_window7_224",
                    pretrained=True,
                    num_classes=0,
                    global_pool="",
                )
                swin_dim = self.swin.num_features
                self.proj = nn.Linear(swin_dim, d_model)
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x, src_key_padding_mask=None):
                # x: (B, T, d_model) — treat T as sequence, no 3D here
                return self.norm(self.proj(x))

        return SwinWrapper(d_model)

    # ------------------------------------------------------------------
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = self.src_proj(src)          # (B, T, d_model)
        src = self.pos_encoding(src)
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)

    # ------------------------------------------------------------------
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        out = self.transformer_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)

    # ------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        tgt_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_v, _ = src.shape
        _, T_tgt   = tgt.shape

        src_mask = self._make_padding_mask(src_lengths, T_v)   if src_lengths is not None else None
        tgt_mask_pad = self._make_padding_mask(tgt_lengths, T_tgt) if tgt_lengths is not None else None
        tgt_causal = nn.Transformer.generate_square_subsequent_mask(T_tgt, device=src.device)

        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, tgt_causal, tgt_mask_pad, src_mask)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        src_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, device = src.size(0), src.device
        src_mask = self._make_padding_mask(src_lengths, src.size(1)) if src_lengths is not None else None
        memory = self.encode(src, src_mask)

        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.max_seq_len):
            T_tgt = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt, device=device)
            logits = self.decode(ys, memory, tgt_mask, memory_key_padding_mask=src_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            finished |= next_token == eos_idx
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return ys[:, 1:]

    # ------------------------------------------------------------------
    @staticmethod
    def _make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        B = lengths.size(0)
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
        return idx >= lengths.unsqueeze(1)