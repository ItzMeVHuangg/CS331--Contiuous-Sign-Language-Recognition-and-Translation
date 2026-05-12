import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class LateFusion(nn.Module):
    """
    Late-fusion module combining visual hidden states with gloss embeddings.

    Modes
    ─────
    "attention"  (default, recommended)
        Visual sequence queries attend over the gloss sequence via
        cross-attention. Preserves gloss order — critical because gloss
        tokens are linguistically sequential (e.g. IX BIS HEUTE).

    "concat"
        Mean-pools the gloss sequence into a single context vector and
        concatenates it with each visual time step. Simple but loses gloss
        ordering — use only for ablation of the attention mode.

    "add"
        Adds mean-pooled gloss context to visual features. Lightest option,
        least expressive.

    Fix log (vs. previous version):
        - Default mode changed from "concat" → "attention" (concat destroys
          gloss ordering via mean pooling, which is linguistically incorrect).
        - Removed duplicate LayerNorm in attention mode (applied attn_norm
          then layer_norm again — unnecessary and destabilising).
        - Consistent residual connection in all modes before final norm.
    """

    def __init__(
        self,
        visual_dim:      int,
        gloss_embed_dim: int,
        fused_dim:       int,
        mode:            Literal["attention", "concat", "add"] = "attention",
        dropout:         float = 0.2,
        nhead:           int   = 8,
    ):
        super().__init__()
        self.mode      = mode
        self.fused_dim = fused_dim

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gloss_proj = nn.Sequential(
            nn.Linear(gloss_embed_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if mode == "concat":
            self.output_proj = nn.Sequential(
                nn.Linear(fused_dim * 2, fused_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        elif mode == "attention":
            # Cross-attention: visual (Q) attends over gloss (K, V)
            # This preserves gloss sequence order and allows each visual
            # timestep to selectively retrieve relevant gloss context.
            self.cross_attn = nn.MultiheadAttention(
                embed_dim   = fused_dim,
                num_heads   = nhead,
                dropout     = dropout,
                batch_first = True,
            )
            # Single norm after residual — no double normalisation
            self.post_attn_norm = nn.LayerNorm(fused_dim)

        # Final norm applied to all modes
        self.final_norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        visual_feats: torch.Tensor,            # (B, T_v, visual_dim)
        gloss_feats:  torch.Tensor,            # (B, T_g, gloss_embed_dim)
        gloss_mask:   Optional[torch.Tensor] = None,  # (B, T_g) True=padded
    ) -> torch.Tensor:
        """Returns: (B, T_v, fused_dim)"""

        v = self.visual_proj(visual_feats)     # (B, T_v, fused_dim)
        g = self.gloss_proj(gloss_feats)       # (B, T_g, fused_dim)

        if self.mode == "attention":
            attn_out, _ = self.cross_attn(
                query            = v,
                key              = g,
                value            = g,
                key_padding_mask = gloss_mask,
            )                                  # (B, T_v, fused_dim)
            # Residual + single norm
            fused = self.post_attn_norm(v + attn_out)

        elif self.mode == "concat":
            g_ctx = g.mean(dim=1, keepdim=True).expand_as(v)
            fused = self.output_proj(torch.cat([v, g_ctx], dim=-1))

        elif self.mode == "add":
            g_ctx = g.mean(dim=1, keepdim=True).expand_as(v)
            fused = v + g_ctx

        else:
            raise ValueError(f"Unknown fusion mode: {self.mode!r}")

        return self.final_norm(fused)          # (B, T_v, fused_dim)


# ──────────────────────────────────────────────────────────────────────────────

class GlossEmbedding(nn.Module):
    """
    Learnable embedding for gloss token IDs.

    Converts predicted (or ground-truth) gloss indices → dense vectors
    that LateFusion injects into the visual stream.
    """

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 1):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, gloss_ids: torch.Tensor) -> torch.Tensor:
        """gloss_ids: (B, T_g) → (B, T_g, embed_dim)"""
        return self.layer_norm(self.embedding(gloss_ids))
