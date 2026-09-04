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


def build_model(name: str, num_classes: int, dropout: float = 0.2, pretrained_baselines: bool = True) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    if name.startswith("pedilite_"):
        attention, width_mult = parse_pedilite_name(name)
        return PediLiteAttnNet(num_classes=num_classes, attention=attention, dropout=dropout, width_mult=width_mult)
    if name in {"mobilenet_v3_small", "efficientnet_b0", "densenet121", "resnet50"}:
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
