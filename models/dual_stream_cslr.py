# -*- coding: utf-8 -*-
"""
dual_stream_cslr.py
====================
Dual-stream CSLR model: ResNet34-2D (stream 1) + Video Swin-T (stream 2)
fused via cross-attention.

Architecture
────────────
                     frames (B, T, C, H, W)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   CNNEncoder (ResNet34)        VideoSwinEncoder (Swin-T)
     (B, T, 512)                    (B, T/2, 512)
          │                               │
   BiLSTM1 + CTC                  BiLSTM2 + CTC
     (B, T, 256)                    (B, T/2, 256)
     lp1 (aux)                      lp2 (aux)
          │                               │
          └───── CrossAttention ──────────┘
                h1 query, h2 as K/V
                   (B, T, 256)
                        │
               Fused CTC head → lp_fused  (main)
                        │
                    h_fused  → SLT stage

Training losses:
    L_total = λ1 * L_ctc1 + λ2 * L_ctc2 + λ3 * L_fused

    Recommended: λ1=0.3, λ2=0.1, λ3=0.6
    (high weight on fused CTC; stream-2 lower because it's noisier)

Freezing strategy:
    Phase A (epochs 1–N_freeze): freeze stream1, only train stream2 + fusion
    Phase B (remaining):         unfreeze stream1, joint fine-tune with low LR

Fix log:
    [FIX-1] CNNEncoder.forward() called with 3 args but server's CNNEncoder only
            accepts 2 (no `lengths` param in the installed version).
            Solution: always call cnn_encoder with frames only, handle lengths
            adjustment separately via temp_pool.
    [FIX-2] Added load_stream2_from_ablation_ckpt() to load VideoSwin+BiLSTM
            weights from Variant G checkpoint into stream2.
            Key mapping: encoder.* → swin_encoder.*, seq_model.* → bilstm2.*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DualStreamCSLR(nn.Module):
    """
    Dual-stream CSLR combining ResNet34-2D and Video Swin-T via cross-attention.

    Args:
        num_classes        : size of gloss vocabulary (including blank)
        cnn_out_features   : output dim of CNNEncoder (default 512)
        swin_out_features  : output dim of VideoSwinEncoder (default 512)
        bilstm_hidden      : hidden size per direction of each BiLSTM (default 512)
        bilstm_layers      : number of BiLSTM layers (default 2)
        bilstm_dropout     : dropout for BiLSTM (default 0.3)
        projection_size    : BiLSTM projection dim — feeds cross-attn (default 256)
        fusion_nhead       : heads in cross-attention (default 8; must divide projection_size)
        fusion_dropout     : dropout in cross-attention (default 0.2)
        swin_backbone      : "swin3d_t" | "swin3d_s" | "swin3d_b" (default "swin3d_t")
        swin_pretrained    : use ImageNet-21K pre-trained Swin weights (default True)
        swin_clip_len      : temporal clip length for VideoSwin (default 16)
        swin_clip_stride   : stride between clips (default 8)
        swin_max_clips     : max clips per forward pass to limit VRAM (default 4)
    """

    def __init__(
        self,
        num_classes:       int,
        cnn_out_features:  int   = 512,
        swin_out_features: int   = 512,
        bilstm_hidden:     int   = 512,
        bilstm_layers:     int   = 2,
        bilstm_dropout:    float = 0.3,
        projection_size:   int   = 256,
        fusion_nhead:      int   = 8,
        fusion_dropout:    float = 0.2,
        swin_backbone:     str   = "swin3d_t",
        swin_pretrained:   bool  = True,
        swin_clip_len:     int   = 16,
        swin_clip_stride:  int   = 8,
        swin_max_clips:    Optional[int] = 4,
        blank_idx:         int   = 0,
    ):
        super().__init__()
        self.num_classes    = num_classes
        self.blank_idx      = blank_idx
        self.projection_size = projection_size

        # ── Stream 1: ResNet34-2D + BiLSTM ────────────────────────────────
        from models.cnn_encoder import CNNEncoder
        from models.bilstm_ctc  import BiLSTM_CTC

        self.cnn_encoder = CNNEncoder(
            backbone     = "resnet34",
            pretrained   = True,
            out_features = cnn_out_features,
            freeze_bn    = False,
        )
        self.bilstm1 = BiLSTM_CTC(
            input_size      = cnn_out_features,
            hidden_size     = bilstm_hidden,
            num_layers      = bilstm_layers,
            num_classes     = num_classes,
            dropout         = bilstm_dropout,
            projection_size = projection_size,
            blank_idx       = blank_idx,
        )

        # ── Stream 2: VideoSwin-T + BiLSTM ────────────────────────────────
        from models.video_swin_encoder import VideoSwinEncoder

        self.swin_encoder = VideoSwinEncoder(
            backbone          = swin_backbone,
            pretrained        = swin_pretrained,
            out_features      = swin_out_features,
            clip_len          = swin_clip_len,
            clip_stride       = swin_clip_stride,
            max_clips_per_fwd = swin_max_clips,
        )
        self.bilstm2 = BiLSTM_CTC(
            input_size      = swin_out_features,
            hidden_size     = bilstm_hidden,
            num_layers      = bilstm_layers,
            num_classes     = num_classes,
            dropout         = bilstm_dropout,
            projection_size = projection_size,
            blank_idx       = blank_idx,
        )

        # ── Temporal pooling for stream 1 (match ablation study) ──────────
        # 2x MaxPool1d-stride-2 applied twice → T → T//4
        # Keeps sequence length consistent with standalone variant A
        from models.temporal_pool import TemporalPool
        self.temp_pool = TemporalPool(num_pool_layers=2)

        # ── Cross-Attention Fusion ─────────────────────────────────────────
        # h1 (B, T1, D) = query  |  h2_up (B, T1, D) = key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = projection_size,
            num_heads   = fusion_nhead,
            dropout     = fusion_dropout,
            batch_first = True,
        )
        self.fusion_norm    = nn.LayerNorm(projection_size)
        self.fusion_dropout = nn.Dropout(fusion_dropout)

        # Fused CTC head
        self.fused_ctc_head = nn.Linear(projection_size, num_classes)

        # Expose for SLT adapter
        self.hidden_out_dim = projection_size

    # ── Length helpers ────────────────────────────────────────────────────────

    def _compute_swin_lengths(
        self,
        frame_lengths: torch.Tensor,   # (B,) original frame counts
        T_in:          int,
        T_swin_out:    int,            # actual swin output length from feat2.size(1)
    ) -> torch.Tensor:
        """
        Map frame_lengths → approximate lengths after VideoSwin temporal subsampling.
        Uses ratio scaling, clamped to actual output size.
        """
        scale   = T_swin_out / max(T_in, 1)
        lengths = (frame_lengths.float() * scale).ceil().long()
        return lengths.clamp(min=1, max=T_swin_out)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        frames:       torch.Tensor,           # (B, T, C, H, W)
        frame_lengths: torch.Tensor,          # (B,) valid frame counts
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            lp_fused : (T1, B, C)    — primary fused CTC log-probs
            h_fused  : (B, T1, D)    — fused hidden states (for SLT)
            lens1    : (B,)          — sequence lengths for stream 1

            lp1      : (T1, B, C)   — stream-1 aux CTC log-probs
            lp2      : (T2, B, C)   — stream-2 aux CTC log-probs
            lens2    : (B,)          — sequence lengths for stream 2
        """
        B, T_in, C, H, W = frames.shape

        # ── Stream 1: ResNet34 → TemporalPool → BiLSTM ────────────────────
        # [FIX-1] Call CNNEncoder with frames only (no lengths argument).
        #         The installed server version of CNNEncoder.forward() only
        #         accepts `self` + `frames` (2 positional args total).
        #         Passing frame_lengths caused:
        #           TypeError: CNNEncoder.forward() takes 2 positional arguments
        #           but 3 were given
        feat1 = self.cnn_encoder(frames)                    # (B, T_in, 512)
        feat1 = self.temp_pool(feat1)                       # (B, T_in//4, 512)
        lens1 = self.temp_pool.adjust_lengths(frame_lengths, T_in)
        lens1 = lens1.clamp(min=1, max=feat1.size(1))

        lp1, h1 = self.bilstm1(feat1, lens1)               # lp1:(T1,B,C) h1:(B,T1,256)

        # ── Stream 2: VideoSwin → BiLSTM ──────────────────────────────────
        feat2 = self.swin_encoder(frames)                   # (B, T2, 512)
        T2    = feat2.size(1)
        lens2 = self._compute_swin_lengths(frame_lengths, T_in, T2)

        lp2, h2 = self.bilstm2(feat2, lens2)               # lp2:(T2,B,C) h2:(B,T2,256)

        # ── Cross-Attention Fusion ─────────────────────────────────────────
        T1 = h1.size(1)
        if T2 != T1:
            # Upsample h2 → T1 for temporal alignment
            h2_aligned = F.interpolate(
                h2.permute(0, 2, 1).float(),   # (B, D, T2)
                size  = T1,
                mode  = "linear",
                align_corners = False,
            ).permute(0, 2, 1)                 # (B, T1, D)
        else:
            h2_aligned = h2

        # h1 as Query, h2_aligned as Key/Value
        attn_out, _ = self.cross_attn(
            query = h1,
            key   = h2_aligned,
            value = h2_aligned,
        )                                      # (B, T1, D)

        # Residual + LayerNorm — keeps stream-1 dominant signal
        h_fused = self.fusion_norm(h1 + self.fusion_dropout(attn_out))

        # Fused CTC
        logits_fused = self.fused_ctc_head(h_fused)                      # (B, T1, C)
        lp_fused = F.log_softmax(logits_fused, dim=-1).permute(1, 0, 2)  # (T1, B, C)

        return lp_fused, h_fused, lens1, lp1, lp2, lens2

    # ── Checkpoint utilities ──────────────────────────────────────────────────

    def load_stream1_from_ablation_ckpt(
        self,
        ckpt_path:  str,
        device:     str = "cpu",
        strict:     bool = False,
    ) -> None:
        """
        Load ResNet34-2D + BiLSTM weights from cslr_variant_A.pth
        (saved by MBtrain_ablation.py as a CSLRModel with keys
        encoder.* and seq_model.*).

        Key mapping:
            encoder.*     → cnn_encoder.*
            seq_model.*   → bilstm1.*
            temp_pool.*   → temp_pool.*   (if present)
        """
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
        # MBtrain_ablation saves a dict with key "model"
        state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

        mapped = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                mapped["cnn_encoder." + k[len("encoder."):]] = v
            elif k.startswith("seq_model."):
                mapped["bilstm1." + k[len("seq_model."):]] = v
            elif k.startswith("temp_pool."):
                mapped[k] = v

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        print(f"[DualStream] Loaded stream1 weights from: {ckpt_path}")
        print(f"  Loaded keys  : {len(mapped) - len(missing)}")
        print(f"  Missing keys : {len(missing)}")
        if missing:
            print(f"    First 5: {missing[:5]}")

    def load_stream2_from_ablation_ckpt(
        self,
        ckpt_path:  str,
        device:     str = "cpu",
    ) -> None:
        """
        Load VideoSwin + BiLSTM2 weights from Variant G checkpoint.

        Key structure found in best_path_33.pth (Variant G):
            encoder.backbone.patch_embed.*  -> swin_encoder.patch_embed.*
            encoder.backbone.features.*     -> swin_encoder.stages.*
            encoder.backbone.norm.*         -> swin_encoder.swin_norm.*
            encoder.temporal_pool.*         -> swin_encoder.temporal_pool.*
            encoder.proj.*                  -> swin_encoder.proj.*
            encoder.out_norm.*              -> swin_encoder.out_norm.*
            seq_model.*                     -> bilstm2.*

        The extra "backbone." level exists because MBtrain_ablation's
        VideoSwinEncoder stored Swin3D sub-modules under self.backbone,
        whereas the current VideoSwinEncoder stores them directly
        (self.patch_embed, self.stages, self.swin_norm).
        """
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

        # Strip optional leading "model." prefix
        sample_keys = list(state.keys())[:5]
        if all(k.startswith("model.") for k in sample_keys):
            state = {k[len("model."):]: v for k, v in state.items()}

        mapped = {}
        skipped = []
        for k, v in state.items():
            if k.startswith("seq_model."):
                # BiLSTM: seq_model.* -> bilstm2.*
                mapped["bilstm2." + k[len("seq_model."):]] = v

            elif k.startswith("encoder.backbone.patch_embed."):
                # swin patch embed
                mapped["swin_encoder.patch_embed." + k[len("encoder.backbone.patch_embed."):]] = v

            elif k.startswith("encoder.backbone.features."):
                # Swin transformer stages (stored as "features" in older VideoSwinEncoder)
                mapped["swin_encoder.stages." + k[len("encoder.backbone.features."):]] = v

            elif k.startswith("encoder.backbone.norm."):
                # Final LayerNorm of Swin
                mapped["swin_encoder.swin_norm." + k[len("encoder.backbone.norm."):]] = v

            elif k.startswith("encoder.backbone."):
                # Any other backbone sub-module (pos_drop, etc.)
                mapped["swin_encoder." + k[len("encoder.backbone."):]] = v

            elif k.startswith("encoder."):
                # proj, out_norm, temporal_pool (no backbone. level)
                mapped["swin_encoder." + k[len("encoder."):]] = v

            else:
                skipped.append(k)

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        matched = set(mapped.keys()) - set(missing)
        print(f"[DualStream] Loaded stream2 (Swin+BiLSTM2) from: {ckpt_path}")
        print(f"  Keys mapped    : {len(mapped)}")
        print(f"  Keys loaded OK : {len(matched)}")
        print(f"  Keys missing   : {len(missing)}  (keys in DualCSLR not in ckpt -- expected for new layers)")
        print(f"  Keys skipped   : {len(skipped)}")
        if len(unexpected) > 0:
            print(f"  [WARN] Still-unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}")
            print(f"         These keys were NOT loaded into the model.")
        else:
            print(f"  [OK] All mapped keys loaded successfully -- stream2 using Variant G weights.")
        if skipped:
            print(f"  Skipped: {skipped[:5]}")

    def freeze_stream1(self) -> None:
        """Freeze ResNet34-2D + BiLSTM1 — call during Phase A training."""
        for p in self.cnn_encoder.parameters():
            p.requires_grad = False
        for p in self.bilstm1.parameters():
            p.requires_grad = False
        # also freeze temp_pool (no learnable params anyway)
        print("[DualStream] Stream1 (ResNet34 + BiLSTM1) FROZEN.")

    def unfreeze_stream1(self, encoder_only: bool = False) -> None:
        """Unfreeze for Phase B joint fine-tuning."""
        for p in self.cnn_encoder.parameters():
            p.requires_grad = True
        if not encoder_only:
            for p in self.bilstm1.parameters():
                p.requires_grad = True
        print("[DualStream] Stream1 UNFROZEN for joint fine-tuning.")

    def stream1_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.cnn_encoder.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# SLT wrapper — plugs into existing run_slt_epochs from MBtrain_ablation.py
# ─────────────────────────────────────────────────────────────────────────────

class DualCSLTModel(nn.Module):
    """
    Full CSLR→SLT pipeline using DualStreamCSLR as backbone.

    Mirrors CSLTModel from MBtrain_ablation.py so run_slt_epochs() works
    without modification.

    forward()   → teacher-forcing for SLT training
    translate() → inference (predicted gloss from fused CTC → decode → embed)
    """

    def __init__(
        self,
        dual_cslr:      DualStreamCSLR,
        gloss_embed:    nn.Module,       # GlossEmbedding
        late_fusion:    nn.Module,       # LateFusion
        translator:     nn.Module,       # SLTTransformer
    ):
        super().__init__()
        self.cslr        = dual_cslr
        self.gloss_embed = gloss_embed
        self.late_fusion = late_fusion
        self.translator  = translator

    def _run_cslr(self, frames, frame_lens):
        """Shared CSLR forward — returns (lp_fused, h_fused, lens1)."""
        lp_fused, h_fused, lens1, _, _, _ = self.cslr(frames, frame_lens)
        return lp_fused, h_fused, lens1

    def forward(self, frames, frame_lens, gloss, gloss_lens, tgt, tgt_lens):
        """Teacher-forcing SLT training. Returns logits (B, T_tgt, vocab)."""
        with torch.no_grad():
            _, h_fused, lens1 = self._run_cslr(frames, frame_lens)

        gloss_emb = self.gloss_embed(gloss)
        fused     = self.late_fusion(h_fused, gloss_emb)
        return self.translator(fused, tgt, lens1, tgt_lens)

    @torch.no_grad()
    def translate(
        self,
        frames,
        frame_lens,
        gloss       = None,
        gloss_lens  = None,
        bos_idx:    int  = None,
        eos_idx:    int  = None,
        use_oracle_gloss: bool = False,
        blank_idx:  int  = 0,
    ):
        """
        Inference. If use_oracle_gloss=False, glosses are decoded from fused CTC.
        Mirrors CSLTModel.translate() interface exactly.
        """
        from utils.ctc_decoder import batch_ctc_decode

        lp_fused, h_fused, lens1 = self._run_cslr(frames, frame_lens)

        if use_oracle_gloss:
            effective_gloss = gloss
        else:
            T_out = lp_fused.size(0)
            pred_gloss_ids = batch_ctc_decode(
                lp_fused,
                lens1.clamp(max=T_out),
                blank_idx = blank_idx,
                mode      = "greedy",
            )
            max_g = max((len(g) for g in pred_gloss_ids), default=1)
            max_g = max(max_g, 1)
            device = frames.device
            effective_gloss = torch.zeros(
                frames.size(0), max_g, dtype=torch.long, device=device)
            for b, pred in enumerate(pred_gloss_ids):
                if pred:
                    t = torch.tensor(pred[:max_g], dtype=torch.long, device=device)
                    effective_gloss[b, :len(t)] = t

        gloss_emb = self.gloss_embed(effective_gloss)
        fused     = self.late_fusion(h_fused, gloss_emb)

        if hasattr(self.translator, "beam_decode"):
            return self.translator.beam_decode(fused, bos_idx, eos_idx, lens1)
        return self.translator.greedy_decode(fused, bos_idx, eos_idx, lens1)