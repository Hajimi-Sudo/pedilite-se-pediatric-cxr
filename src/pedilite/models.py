from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidualSE(nn.Module):
    """Mobile inverted-residual block with channel recalibration."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 2, use_se: bool = True):
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        hidden = max(in_ch, int(round(in_ch * expand_ratio)))
        layers = []
        if hidden != in_ch:
            layers.extend([
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.Hardswish(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        if use_se:
            # Insert channel recalibration before the pointwise projection.
            layers.insert(-2, SEBlock(hidden))
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_residual else y


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ECABlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class CBAMBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel = SEBlock(channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.channel(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([avg, mx], dim=1))


def attention_block(name: str, channels: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SEBlock(channels)
    if name == "eca":
        return ECABlock(channels)
    if name == "cbam":
        return CBAMBlock(channels)
    raise ValueError(f"unknown attention: {name}")


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class PediLiteAttnNet(nn.Module):
    def __init__(self, num_classes: int, attention: str = "eca", dropout: float = 0.2, width_mult: float = 1.0):
        super().__init__()
        if width_mult <= 0:
            raise ValueError(f"width_mult must be positive, got {width_mult}")
        channels = [max(8, int(round(ch * width_mult))) for ch in [16, 32, 64, 96]]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(channels[0], channels[1], stride=2),
            attention_block(attention, channels[1]),
            DepthwiseSeparableConv(channels[1], channels[2], stride=2),
            attention_block(attention, channels[2]),
            DepthwiseSeparableConv(channels[2], channels[3], stride=2),
            attention_block(attention, channels[3]),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[3], num_classes),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class PediLiteSEV2(nn.Module):
    """Higher-capacity compact backbone for the upgrade protocol.

    The v2 model keeps channel attention but adds inverted-residual expansion,
    residual feature reuse, and a fifth compact stage. It is intentionally
    separate from the submitted v1 model so existing results remain reproducible.
    """

    def __init__(self, num_classes: int, dropout: float = 0.2, width_mult: float = 1.0, use_se: bool = True):
        super().__init__()
        if width_mult <= 0:
            raise ValueError(f"width_mult must be positive, got {width_mult}")
        channels = [max(8, int(round(ch * width_mult))) for ch in [16, 24, 40, 64, 96]]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.Hardswish(inplace=True),
        )
        specs = [
            (channels[0], channels[1], 2, 2),
            (channels[1], channels[2], 2, 2),
            (channels[2], channels[3], 2, 2),
            (channels[3], channels[4], 2, 2),
            (channels[4], channels[4], 1, 2),
        ]
        self.blocks = nn.Sequential(*(InvertedResidualSE(*spec, use_se=use_se) for spec in specs))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def parse_pedilite_name(name: str) -> tuple[str, float]:
    parts = name.split("_")
    if len(parts) < 2 or parts[0] != "pedilite":
        raise ValueError(f"invalid PediLite model name: {name}")
    attention = parts[1]
    width_mult = 1.0
    if len(parts) == 3:
        width_token = parts[2]
        if not width_token.startswith("w"):
            raise ValueError(f"invalid PediLite width token in model name: {name}")
        width_mult = float(width_token[1:])
    elif len(parts) > 3:
        raise ValueError(f"invalid PediLite model name: {name}")
    return attention, width_mult


def parse_pedilite_v2_name(name: str) -> float:
    prefix = "pedilite_se_v2"
    if name == prefix:
        return 1.0
    if not name.startswith(prefix + "_w"):
        raise ValueError(f"invalid PediLite v2 model name: {name}")
    return float(name[len(prefix) + 2 :])


def build_model(name: str, num_classes: int, dropout: float = 0.2, pretrained_baselines: bool = True) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    if name.startswith("pedilite_"):
        if name in {"pedilite_se_v2", "pedilite_se_v2_none"} or name.startswith("pedilite_se_v2_w"):
            use_se = name != "pedilite_se_v2_none"
            width_name = name.replace("_none", "")
            width_mult = parse_pedilite_v2_name(width_name)
            return PediLiteSEV2(num_classes=num_classes, dropout=dropout, width_mult=width_mult, use_se=use_se)
        attention, width_mult = parse_pedilite_name(name)
        return PediLiteAttnNet(num_classes=num_classes, attention=attention, dropout=dropout, width_mult=width_mult)
    if name in {
        "mobilenet_v3_small",
        "efficientnet_b0",
        "efficientnet_v2_s",
        "convnext_tiny",
        "swin_t",
        "swin_v2_t",
        "maxvit_t",
        "densenet121",
        "resnet50",
    }:
        try:
            import torchvision.models as tvm
        except Exception as exc:
            raise RuntimeError("torchvision is required for transfer-learning baselines") from exc
        if name == "mobilenet_v3_small":
            weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.mobilenet_v3_small(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if name == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.efficientnet_b0(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if name == "efficientnet_v2_s":
            weights = tvm.EfficientNet_V2_S_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.efficientnet_v2_s(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if name == "convnext_tiny":
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.convnext_tiny(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if name == "swin_t":
            weights = tvm.Swin_T_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.swin_t(weights=weights)
            model.head = nn.Linear(model.head.in_features, num_classes)
            return model
        if name == "swin_v2_t":
            weights = tvm.Swin_V2_T_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.swin_v2_t(weights=weights)
            model.head = nn.Linear(model.head.in_features, num_classes)
            return model
        if name == "maxvit_t":
            weights = tvm.MaxVit_T_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.maxvit_t(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if name == "densenet121":
            weights = tvm.DenseNet121_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.densenet121(weights=weights)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model
        if name == "resnet50":
            weights = tvm.ResNet50_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
    raise ValueError(f"unknown model: {name}")
