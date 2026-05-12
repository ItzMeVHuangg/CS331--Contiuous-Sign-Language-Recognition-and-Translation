
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
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
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────────────
# SLT Transformer
# ──────────────────────────────────────────────────────────────────────────────

class SLTTransformer(nn.Module):
   

    def __init__(
        self,
        src_dim:            int,
        tgt_vocab_size:     int,
        d_model:            int = 512,
        nhead:              int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward:    int = 2048,
        dropout:            float = 0.1,
        max_seq_len:        int = 80,
        pad_idx:            int = 1,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx     = pad_idx

        # ── Project fused features to d_model ───────────────────────
        self.src_proj = nn.Linear(src_dim, d_model)

        # ── Target embedding + positional encoding ───────────────────
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding  = PositionalEncoding(d_model, max_len=max_seq_len + 10, dropout=dropout)

        # ── Transformer ──────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ── Output head ──────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    def encode(
        self,
        src: torch.Tensor,              # (B, T_v, src_dim)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T_v) bool
    ) -> torch.Tensor:
        """Run encoder; returns memory (B, T_v, d_model)."""
        src = self.src_proj(src)        # (B, T_v, d_model)
        src = self.pos_encoding(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return memory

    # ------------------------------------------------------------------
    def decode(
        self,
        tgt: torch.Tensor,              # (B, T_tgt) token ids
        memory: torch.Tensor,           # (B, T_v, d_model)
        tgt_mask: Optional[torch.Tensor] = None,          # (T_tgt, T_tgt) causal
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T_tgt)
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run decoder; returns logits (B, T_tgt, vocab_size)."""
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        out = self.transformer_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)    # (B, T_tgt, vocab_size)

    # ------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,              # (B, T_v, src_dim)
        tgt: torch.Tensor,              # (B, T_tgt)
        src_lengths: Optional[torch.Tensor] = None,  # (B,)
        tgt_lengths: Optional[torch.Tensor] = None,  # (B,)
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.
        Returns logits: (B, T_tgt, vocab_size)
        """
        B, T_v, _ = src.shape
        _, T_tgt   = tgt.shape

        # Build masks
        src_key_padding_mask = None
        if src_lengths is not None:
            src_key_padding_mask = self._make_padding_mask(src_lengths, T_v)

        tgt_key_padding_mask = None
        if tgt_lengths is not None:
            tgt_key_padding_mask = self._make_padding_mask(tgt_lengths, T_tgt)

        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt, device=src.device)

        memory = self.encode(src, src_key_padding_mask)
        logits = self.decode(
            tgt, memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,   # (B, T_v, src_dim)
        bos_idx: int,
        eos_idx: int,
        src_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding. Returns (B, max_seq_len) token ids."""
        B = src.size(0)
        device = src.device

        src_key_padding_mask = None
        if src_lengths is not None:
            src_key_padding_mask = self._make_padding_mask(src_lengths, src.size(1))

        memory = self.encode(src, src_key_padding_mask)

        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.max_seq_len):
            T_tgt = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt, device=device)
            logits = self.decode(ys, memory, tgt_mask=tgt_mask,
                                 memory_key_padding_mask=src_key_padding_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
            finished |= next_token == eos_idx
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return ys[:, 1:]  # strip BOS

    # ------------------------------------------------------------------
    @staticmethod
    def _make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Returns (B, max_len) bool mask: True where padded."""
        B = lengths.size(0)
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
        return idx >= lengths.unsqueeze(1)