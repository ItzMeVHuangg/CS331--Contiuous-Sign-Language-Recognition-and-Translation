
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class LateFusion(nn.Module):


    def __init__(
        self,
        visual_dim: int,
        gloss_embed_dim: int,
        fused_dim: int,
        mode: Literal["concat", "add", "attention"] = "concat",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mode = mode
        self.fused_dim = fused_dim

        # ── Project visual to fused_dim ──────────────────────────────
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Project gloss to fused_dim ───────────────────────────────
        self.gloss_proj = nn.Sequential(
            nn.Linear(gloss_embed_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Mode-specific layers ─────────────────────────────────────
        if mode == "concat":
            self.output_proj = nn.Sequential(
                nn.Linear(fused_dim * 2, fused_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        elif mode == "attention":
            # Cross-attention: visual queries, gloss keys/values
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=fused_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(fused_dim)

        self.layer_norm = nn.LayerNorm(fused_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        visual_feats: torch.Tensor,   # (B, T_v, visual_dim)
        gloss_feats:  torch.Tensor,   # (B, T_g, gloss_embed_dim)
        gloss_mask:   torch.Tensor = None,  # (B, T_g) bool padding mask
    ) -> torch.Tensor:
     
        # Project both streams to fused_dim
        v = self.visual_proj(visual_feats)    # (B, T_v, fused_dim)
        g = self.gloss_proj(gloss_feats)      # (B, T_g, fused_dim)

        if self.mode == "concat":
            # Collapse gloss sequence to a single context vector, then tile
            g_ctx = g.mean(dim=1, keepdim=True).expand_as(v)  # (B, T_v, fused_dim)
            fused = torch.cat([v, g_ctx], dim=-1)              # (B, T_v, 2*fused_dim)
            fused = self.output_proj(fused)                    # (B, T_v, fused_dim)

        elif self.mode == "add":
            g_ctx = g.mean(dim=1, keepdim=True).expand_as(v)
            fused = v + g_ctx

        elif self.mode == "attention":
            # Cross-attention: queries=visual, keys/values=gloss
            key_padding_mask = gloss_mask if gloss_mask is not None else None
            attn_out, _ = self.cross_attn(
                query=v, key=g, value=g,
                key_padding_mask=key_padding_mask,
            )                                                  # (B, T_v, fused_dim)
            fused = self.attn_norm(v + attn_out)

        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")

        return self.layer_norm(fused)                          # (B, T_v, fused_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Gloss Embedding (converts predicted gloss IDs → dense vectors for fusion)
# ──────────────────────────────────────────────────────────────────────────────

class GlossEmbedding(nn.Module):
  

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, gloss_ids: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.embedding(gloss_ids))