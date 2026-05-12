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


class SLTTransformer(nn.Module):
    """
    Standard Encoder-Decoder Transformer for Sign Language Translation.

    Fix log:
        - max_seq_len reduced from 300 → 128 (PHOENIX avg ~10 tokens, max ~40).
        - src_proj output now scaled by √d_model to match tgt_embedding scale,
          ensuring the encoder and decoder see features at the same magnitude.
        - pad_idx default corrected (Vocabulary.PAD = index 1).
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
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx     = pad_idx
        self._scale      = math.sqrt(d_model)

        self.src_proj      = nn.Linear(src_dim, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding  = PositionalEncoding(d_model, max_len=max_seq_len + 16, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
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

    def encode(self, src: torch.Tensor,
               src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Scale src to same magnitude as tgt_embedding
        src = self.src_proj(src) * self._scale
        src = self.pos_encoding(src)
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask=None, tgt_key_padding_mask=None,
               memory_key_padding_mask=None) -> torch.Tensor:
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
        """Teacher-forcing. Returns logits: (B, T_tgt, vocab_size)."""
        B, T_v, _ = src.shape
        T_tgt     = tgt.shape[1]

        src_pad = self._make_padding_mask(src_lengths, T_v)  if src_lengths is not None else None
        tgt_pad = self._make_padding_mask(tgt_lengths, T_tgt) if tgt_lengths is not None else None
        tgt_causal = nn.Transformer.generate_square_subsequent_mask(T_tgt, device=src.device)

        memory = self.encode(src, src_pad)
        return self.decode(tgt, memory, tgt_causal, tgt_pad, src_pad)

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx, eos_idx, src_lengths=None):
        B, device = src.size(0), src.device
        src_pad = (self._make_padding_mask(src_lengths, src.size(1))
                   if src_lengths is not None else None)
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
