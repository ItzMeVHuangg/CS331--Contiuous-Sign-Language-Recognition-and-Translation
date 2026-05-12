
import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Optional


class CNNEncoder(nn.Module):
 

    SUPPORTED = {"resnet18": 512, "resnet50": 2048, "vgg11": 4096}

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        out_features: int = 512,
        freeze_bn: bool = False,
    ):
        super().__init__()
        assert backbone in self.SUPPORTED, f"backbone must be one of {list(self.SUPPORTED)}"

        self.backbone_name = backbone
        self.raw_dim = self.SUPPORTED[backbone]

        # ── Build backbone ──────────────────────────────────────────
        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "resnet18":
            base = tv_models.resnet18(weights=weights)
            # Remove final FC layer; keep pooling
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # → (B, 512, 1, 1)

        elif backbone == "resnet50":
            base = tv_models.resnet50(weights=weights)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # → (B, 2048, 1, 1)

        elif backbone == "vgg11":
            base = tv_models.vgg11_bn(weights=weights if pretrained else None)
            self.feature_extractor = base.features                               # → (B, 512, 7, 7)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Optional projection to out_features ────────────────────
        if out_features != self.raw_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.raw_dim, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
            )
        else:
            self.proj = nn.Identity()

        self.out_features = out_features

        # ── Optionally freeze BN ────────────────────────────────────
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    # ------------------------------------------------------------------
    def _extract_single(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, raw_dim)"""
        feat = self.feature_extractor(x)
        if self.backbone_name == "vgg11":
            feat = self.adaptive_pool(feat)
        feat = feat.flatten(1)      # (B, raw_dim)
        return feat

    # ------------------------------------------------------------------
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
    
        B, T, C, H, W = frames.shape

        # Reshape to process all frames in parallel
        frames_flat = frames.view(B * T, C, H, W)           # (B*T, C, H, W)
        features    = self._extract_single(frames_flat)      # (B*T, raw_dim)
        features    = self.proj(features)                    # (B*T, out_features)
        features    = features.view(B, T, self.out_features) # (B, T, out_features)

        return features