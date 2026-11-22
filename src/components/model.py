import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    efficientnet_b0,
    mobilenet_v2,
)


def _build_backbone(name, pretrained):
    """Both supported backbones emit (B, 1280, 7, 7) feature maps at 224px."""
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return efficientnet_b0(weights=weights).features
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    return mobilenet_v2(weights=weights).features


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class SkinCancerModel(nn.Module):
    """MobileNetV2 feature extractor + lightweight visual Transformer."""

    def __init__(self, num_classes=7, pretrained=True, backbone="mobilenet_v2"):
        super().__init__()
        self.backbone_name = backbone
        self.cnn = _build_backbone(backbone, pretrained)
        self.reduce_dim = nn.Conv2d(1280, 256, kernel_size=1)
        self.transformer = TransformerBlock(embed_dim=256, num_heads=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, x):
        """Pooled 256-d embedding, also used by the KNN out-of-distribution gate."""
        x = self.cnn(x)
        x = self.reduce_dim(x)
        batch, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x

    def forward(self, x):
        return self.classifier(self.extract_features(x))
