import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Optional


class CNNEncoder(nn.Module):


    SUPPORTED = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

    def __init__(
        self,
        backbone:     str   = "resnet18",
        pretrained:   bool  = True,
        out_features: int   = 512,
        freeze_bn:    bool  = False,
    ):
        super().__init__()
        assert backbone in self.SUPPORTED, \
            f"backbone must be one of {list(self.SUPPORTED)}, got '{backbone}'"

        self.backbone_name = backbone
        self.raw_dim       = self.SUPPORTED[backbone]

        if backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base    = tv_models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base    = tv_models.resnet34(weights=weights)
        else:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base    = tv_models.resnet50(weights=weights)

        # Remove classification head; keep avgpool → (B, raw_dim, 1, 1)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        # Optional projection
        if out_features != self.raw_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.raw_dim, out_features),
                nn.GELU(),
                nn.Dropout(p=0.1),
            )
        else:
            self.proj = nn.Identity()

        # Output norm — stabilises BiLSTM / TransformerCTC input scale
        self.out_norm     = nn.LayerNorm(out_features)
        self.out_features = out_features

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def _extract_single(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        return feat.flatten(1)

    def forward(
        self,
        frames:  torch.Tensor,              # (B, T, C, H, W)
        lengths: Optional[torch.Tensor] = None,  # (B,) — valid frame counts
    ) -> torch.Tensor:
        """frames: (B, T, C, H, W) → (B, T, out_features)

        When lengths is provided, only valid frames are processed through
        the CNN to prevent zero-padded frames from corrupting BatchNorm stats.
        """
        B, T, C, H, W = frames.shape

        if lengths is not None:
            # Avoid repeated CPU-GPU sync from lengths[b].item() when lengths is CUDA.
            # Convert once to Python list for control-flow/indexing.
            lengths_list = lengths.detach().cpu().tolist()

            # Fast path: all samples use full T frames -> use fully vectorized branch.
            if all(int(l) >= T for l in lengths_list):
                flat     = frames.view(B * T, C, H, W)
                features = self._extract_single(flat)               # (B*T, raw_dim)
                features = self.proj(features)                      # (B*T, out_features)
                features = features.view(B, T, self.out_features)
                return self.out_norm(features)

            # Pack all valid frames from the batch into a single contiguous
            # tensor, run one forward pass through ResNet, then scatter back.
            valid_slices = [frames[b, :int(lengths_list[b])] for b in range(B)]
            flat_valid   = torch.cat(valid_slices, dim=0)       # (sum(L), C, H, W)

            feat_valid = self._extract_single(flat_valid)       # (sum(L), raw_dim)
            feat_valid = self.proj(feat_valid)                  # (sum(L), out_features)

            out    = torch.zeros(B, T, self.out_features,
                                 device=frames.device, dtype=feat_valid.dtype)
            offset = 0
            for b in range(B):
                L = int(lengths_list[b])
                out[b, :L] = feat_valid[offset: offset + L]
                offset += L
            return self.out_norm(out)
        else:
            flat     = frames.view(B * T, C, H, W)
            features = self._extract_single(flat)               # (B*T, raw_dim)
            features = self.proj(features)                      # (B*T, out_features)
            features = features.view(B, T, self.out_features)
            return self.out_norm(features)
