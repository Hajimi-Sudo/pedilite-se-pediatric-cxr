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


class MixScaleInvertedResidual(nn.Module):
    """Inverted residual with mixed 3x3/5x5 depthwise branches.

    The expanded channels are split in half: one branch uses a 3x3 depthwise
    convolution and the other a 5x5 depthwise convolution. The branches are
    concatenated before optional SE and the projection. ``kernel_mode`` can
    collapse both branches to a single kernel for ablation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: int = 2,
        use_se: bool = False,
        kernel_mode: str = "mix",
    ):
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        if kernel_mode not in {"mix", "k3", "k5"}:
            raise ValueError(f"kernel_mode must be mix, k3, or k5, got {kernel_mode}")
        hidden = max(in_ch, int(round(in_ch * expand_ratio)))
        if kernel_mode == "mix" and hidden % 2:
            hidden += 1
        self.kernel_mode = kernel_mode
        self.hidden = hidden
        if hidden != in_ch:
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.Hardswish(inplace=True),
            )
        else:
            self.expand = nn.Identity()
        if kernel_mode == "mix":
            half = hidden // 2
            self.dw3 = nn.Conv2d(half, half, 3, stride=stride, padding=1, groups=half, bias=False)
            self.dw5 = nn.Conv2d(half, half, 5, stride=stride, padding=2, groups=half, bias=False)
            self.dw = None
        else:
            k = 3 if kernel_mode == "k3" else 5
            self.dw = nn.Conv2d(hidden, hidden, k, stride=stride, padding=k // 2, groups=hidden, bias=False)
            self.dw3 = None
            self.dw5 = None
        self.bn_dw = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)
        self.se = SEBlock(hidden) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        y = self.expand(x)
        if self.kernel_mode == "mix":
            half = self.hidden // 2
            y = torch.cat([self.dw3(y[:, :half]), self.dw5(y[:, half:])], dim=1)
        else:
            y = self.dw(y)
        y = self.act(self.bn_dw(y))
        y = self.project(self.se(y))
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


class PediLiteSE(nn.Module):
    """Final compact PediLite-SE backbone used in the manuscript.

    The internal ``pedilite_se_v2`` key is retained by the experiment configs so
    previously generated checkpoints remain reproducible; it does not denote a
    second proposed model.
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


class PediLiteMS(nn.Module):
    """Compact mix-scale inverted-residual CNN for the IVC upgrade.

    Default ``pedilite_ms`` uses mixed 3x3/5x5 depthwise branches and no SE.
    Attention and single-kernel variants are retained only for ablation.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.2,
        width_mult: float = 1.0,
        use_se: bool = False,
        kernel_mode: str = "mix",
    ):
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
        self.blocks = nn.Sequential(
            *(MixScaleInvertedResidual(*spec, use_se=use_se, kernel_mode=kernel_mode) for spec in specs)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self.out_channels = channels[-1]

    def encode(self, x):
        return self.blocks(self.stem(x))

    def forward(self, x):
        return self.head(self.encode(x))


class PediLiteMSD(nn.Module):
    """Deeper mix-scale inverted residual for the IVC capacity upgrade.

    Keeps mixed 3x3/5x5 depthwise branches, but uses MobileNet-style stage
    repeats and a higher expansion ratio in later stages. Default width stays
    well below ShuffleNet-V2 (~1.26M).
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.2,
        width_mult: float = 1.0,
        use_se: bool = False,
        kernel_mode: str = "mix",
    ):
        super().__init__()
        if width_mult <= 0:
            raise ValueError(f"width_mult must be positive, got {width_mult}")
        channels = [max(8, int(round(ch * width_mult))) for ch in [24, 32, 48, 80, 128]]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.Hardswish(inplace=True),
        )
        stage_specs = [
            (channels[0], channels[1], 2, 2, 1),
            (channels[1], channels[2], 2, 4, 2),
            (channels[2], channels[3], 2, 4, 3),
            (channels[3], channels[4], 2, 4, 2),
            (channels[4], channels[4], 1, 4, 1),
        ]
        blocks = []
        for in_ch, out_ch, stride, expand, repeats in stage_specs:
            for idx in range(repeats):
                block_stride = stride if idx == 0 else 1
                block_in = in_ch if idx == 0 else out_ch
                blocks.append(
                    MixScaleInvertedResidual(
                        block_in,
                        out_ch,
                        stride=block_stride,
                        expand_ratio=expand,
                        use_se=use_se,
                        kernel_mode=kernel_mode,
                    )
                )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))



class PediLiteMSAC(nn.Module):
    """Early-exit mix-scale CNN under a FLOPs budget.

    The question is not a new convolution. After two shared mix-scale stages a
    router decides whether the remaining depth is worth the extra FLOPs. Cheap
    and full paths share weights; evaluation hard-routes and reports an
    F1-versus-expected-FLOPs envelope. Optional ``cheap_size`` downsamples only
    the early-exit view for a scale-cascade ablation; default keeps 224 px.
    """

    adaptive_compute = True

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.2,
        width_mult: float = 1.0,
        use_se: bool = False,
        kernel_mode: str = "mix",
        cheap_size: int | None = None,
        exit_after: int = 2,
        route_threshold: float = 0.5,
        aux_weight: float = 0.5,
        cost_weight: float = 0.05,
    ):
        super().__init__()
        if exit_after < 1:
            raise ValueError(f"exit_after must be >= 1, got {exit_after}")
        self.backbone = PediLiteMS(
            num_classes=num_classes,
            dropout=dropout,
            width_mult=width_mult,
            use_se=use_se,
            kernel_mode=kernel_mode,
        )
        n_blocks = len(self.backbone.blocks)
        if exit_after >= n_blocks:
            raise ValueError(f"exit_after must be < {n_blocks}, got {exit_after}")
        self.exit_after = int(exit_after)
        self.cheap_size = None if cheap_size in {None, 224} else int(cheap_size)
        self.route_threshold = float(route_threshold)
        self.aux_weight = float(aux_weight)
        self.cost_weight = float(cost_weight)
        self.compute_mode = "route"
        exit_channels = self._channels_after(self.exit_after)
        self.exit_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(exit_channels, num_classes),
        )
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(exit_channels, 1),
        )
        self.last_p_full = None

    def _channels_after(self, n_blocks: int) -> int:
        block = self.backbone.blocks[n_blocks - 1]
        project = getattr(block, "project", None)
        if project is None:
            raise RuntimeError("mix-scale block is missing a projection")
        conv = project[0]
        return int(conv.out_channels)

    def _maybe_downscale(self, x):
        import torch.nn.functional as F

        if self.cheap_size is None:
            return x
        if x.shape[-1] == self.cheap_size and x.shape[-2] == self.cheap_size:
            return x
        return F.interpolate(x, size=self.cheap_size, mode="bilinear", align_corners=False)

    def _prefix(self, x):
        feat = self.backbone.stem(x)
        for block in self.backbone.blocks[: self.exit_after]:
            feat = block(feat)
        return feat

    def _suffix(self, feat):
        for block in self.backbone.blocks[self.exit_after :]:
            feat = block(feat)
        return self.backbone.head(feat)

    def forward(self, x, return_aux: bool = False):
        import torch

        mode = self.compute_mode
        prefix_in = self._maybe_downscale(x) if mode != "full" else x
        if mode == "full":
            logits = self.backbone(x)
            p_full = torch.ones(x.shape[0], device=x.device, dtype=logits.dtype)
            self.last_p_full = p_full
            if self.training or return_aux:
                return {
                    "logits": logits,
                    "exit_logits": logits,
                    "full_logits": logits,
                    "p_full": p_full,
                }
            return logits

        feat = self._prefix(prefix_in)
        exit_logits = self.exit_head(feat)
        p_full = self.router(feat).squeeze(1).sigmoid()
        self.last_p_full = p_full.detach()
        if mode == "cheap":
            if self.training or return_aux:
                return {
                    "logits": exit_logits,
                    "exit_logits": exit_logits,
                    "full_logits": exit_logits,
                    "p_full": p_full,
                }
            return exit_logits

        if self.training:
            full_logits = self._suffix(feat if self.cheap_size is None else self._prefix(x))
            mix = (1.0 - p_full.unsqueeze(1)) * exit_logits + p_full.unsqueeze(1) * full_logits
            return {
                "logits": mix,
                "exit_logits": exit_logits,
                "full_logits": full_logits,
                "p_full": p_full,
            }

        need_full = p_full >= self.route_threshold
        logits = exit_logits
        if bool(need_full.any()):
            logits = exit_logits.clone()
            if self.cheap_size is None:
                logits[need_full] = self._suffix(feat[need_full])
            else:
                logits[need_full] = self._suffix(self._prefix(x[need_full]))
        if return_aux:
            full_logits = torch.zeros_like(exit_logits)
            if bool(need_full.any()):
                full_logits[need_full] = logits[need_full]
            return {
                "logits": logits,
                "exit_logits": exit_logits,
                "full_logits": full_logits,
                "p_full": p_full,
            }
        return logits



class PediLiteMSSS(nn.Module):
    """Shared-weight scale-space mix-scale CNN.

    Both 160-px and 224-px views share one PediLite-MS backbone. There is no
    router: training sums the two classification losses and evaluates the
    fused logits. This targets pediatric CXR scale variation without the
    early-exit collapse already observed in ``pedilite_ms_ac``.
    """

    auxiliary_outputs = True

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.2,
        width_mult: float = 1.0,
        use_se: bool = False,
        kernel_mode: str = "mix",
        cheap_size: int = 160,
        aux_weight: float = 0.5,
    ):
        super().__init__()
        if cheap_size < 32:
            raise ValueError(f"cheap_size must be >= 32, got {cheap_size}")
        self.backbone = PediLiteMS(
            num_classes=num_classes,
            dropout=dropout,
            width_mult=width_mult,
            use_se=use_se,
            kernel_mode=kernel_mode,
        )
        self.cheap_size = int(cheap_size)
        self.aux_weight = float(aux_weight)

    def _downscale(self, x):
        import torch.nn.functional as F
        if x.shape[-1] == self.cheap_size and x.shape[-2] == self.cheap_size:
            return x
        return F.interpolate(x, size=self.cheap_size, mode="bilinear", align_corners=False)

    def forward(self, x, return_aux: bool = False):
        logits_224 = self.backbone(x)
        logits_160 = self.backbone(self._downscale(x))
        logits = 0.5 * (logits_224 + logits_160)
        if self.training or return_aux:
            return {
                "logits": logits,
                "logits_224": logits_224,
                "logits_160": logits_160,
                "exit_logits": logits_160,
            }
        return logits



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


def parse_pedilite_se_name(name: str) -> float:
    """Parse the canonical PediLite-SE name or its legacy run-key alias."""
    for prefix in ("pedilite_se", "pedilite_se_v2"):
        if name == prefix:
            return 1.0
        if name.startswith(prefix + "_w"):
            return float(name[len(prefix) + 2 :])
    raise ValueError(f"invalid PediLite-SE model name: {name}")


parse_pedilite_v2_name = parse_pedilite_se_name




def parse_pedilite_ms_ss_name(name: str) -> tuple[int, bool, str, float]:
    """Parse ``pedilite_ms_ss`` names: optional ``_s160``, ``_se``, ``_k3``/``_k5``, ``_w``."""
    if name == "pedilite_ms_ss" or name.startswith("pedilite_ms_ss_"):
        rest = name[len("pedilite_ms_ss"):]
        tokens = [tok for tok in rest.split("_") if tok]
        cheap_size = 160
        use_se = False
        kernel_mode = "mix"
        width_mult = 1.0
        for token in tokens:
            if token == "se":
                use_se = True
            elif token in {"k3", "k5"}:
                kernel_mode = token
            elif token.startswith("w"):
                width_mult = float(token[1:])
            elif token.startswith("s") and token[1:].isdigit():
                cheap_size = int(token[1:])
            else:
                raise ValueError(f"invalid PediLite-MS-SS token in model name: {name}")
        return cheap_size, use_se, kernel_mode, width_mult
    raise ValueError(f"invalid PediLite-MS-SS model name: {name}")


def parse_pedilite_ms_ac_name(name: str) -> tuple[int, bool, str, float]:
    """Parse ``pedilite_ms_ac`` names: optional ``_s160``, ``_se``, ``_k3``/``_k5``, ``_w``."""
    if name == "pedilite_ms_ac" or name.startswith("pedilite_ms_ac_"):
        rest = name[len("pedilite_ms_ac"):]
        tokens = [tok for tok in rest.split("_") if tok]
        cheap_size = None
        use_se = False
        kernel_mode = "mix"
        width_mult = 1.0
        for token in tokens:
            if token == "se":
                use_se = True
            elif token in {"k3", "k5"}:
                kernel_mode = token
            elif token.startswith("w"):
                width_mult = float(token[1:])
            elif token.startswith("s") and token[1:].isdigit():
                cheap_size = int(token[1:])
            else:
                raise ValueError(f"invalid PediLite-MS-AC token in model name: {name}")
        return cheap_size, use_se, kernel_mode, width_mult
    raise ValueError(f"invalid PediLite-MS-AC model name: {name}")


def parse_pedilite_ms_name(name: str) -> tuple[bool, str, float]:
    """Parse ``pedilite_ms`` names: optional ``_se``, ``_k3``/``_k5``, ``_w``."""
    if name.startswith("pedilite_msd"):
        raise ValueError(f"PediLite-MSD names must use parse_pedilite_msd_name: {name}")
    if name == "pedilite_ms" or name.startswith("pedilite_ms_"):
        rest = name[len("pedilite_ms"):]
        tokens = [tok for tok in rest.split("_") if tok]
        use_se = False
        kernel_mode = "mix"
        width_mult = 1.0
        for token in tokens:
            if token == "se":
                use_se = True
            elif token in {"k3", "k5"}:
                kernel_mode = token
            elif token.startswith("w"):
                width_mult = float(token[1:])
            else:
                raise ValueError(f"invalid PediLite-MS token in model name: {name}")
        return use_se, kernel_mode, width_mult
    raise ValueError(f"invalid PediLite-MS model name: {name}")


def parse_pedilite_msd_name(name: str) -> tuple[bool, str, float]:
    """Parse ``pedilite_msd`` names: optional ``_se``, ``_k3``/``_k5``, ``_w``."""
    if name == "pedilite_msd" or name.startswith("pedilite_msd_"):
        rest = name[len("pedilite_msd"):]
        tokens = [tok for tok in rest.split("_") if tok]
        use_se = False
        kernel_mode = "mix"
        width_mult = 1.0
        for token in tokens:
            if token == "se":
                use_se = True
            elif token in {"k3", "k5"}:
                kernel_mode = token
            elif token.startswith("w"):
                width_mult = float(token[1:])
            else:
                raise ValueError(f"invalid PediLite-MSD token in model name: {name}")
        return use_se, kernel_mode, width_mult
    raise ValueError(f"invalid PediLite-MSD model name: {name}")


def build_model(name: str, num_classes: int, dropout: float = 0.2, pretrained_baselines: bool = True) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    if name == "pedilite_msd" or name.startswith("pedilite_msd_"):
        use_se, kernel_mode, width_mult = parse_pedilite_msd_name(name)
        return PediLiteMSD(
            num_classes=num_classes,
            dropout=dropout,
            width_mult=width_mult,
            use_se=use_se,
            kernel_mode=kernel_mode,
        )
    if name == "pedilite_ms_ss" or name.startswith("pedilite_ms_ss_"):
        cheap_size, use_se, kernel_mode, width_mult = parse_pedilite_ms_ss_name(name)
        return PediLiteMSSS(
            num_classes=num_classes,
            dropout=dropout,
            width_mult=width_mult,
            use_se=use_se,
            kernel_mode=kernel_mode,
            cheap_size=cheap_size,
        )
    if name == "pedilite_ms_ac" or name.startswith("pedilite_ms_ac_"):
        cheap_size, use_se, kernel_mode, width_mult = parse_pedilite_ms_ac_name(name)
        return PediLiteMSAC(
            num_classes=num_classes,
            dropout=dropout,
            width_mult=width_mult,
            use_se=use_se,
            kernel_mode=kernel_mode,
            cheap_size=cheap_size,
        )
    if name == "pedilite_ms" or name.startswith("pedilite_ms_"):
        use_se, kernel_mode, width_mult = parse_pedilite_ms_name(name)
        return PediLiteMS(
            num_classes=num_classes,
            dropout=dropout,
            width_mult=width_mult,
            use_se=use_se,
            kernel_mode=kernel_mode,
        )
    if name.startswith("pedilite_"):
        if name in {"pedilite_se", "pedilite_se_none", "pedilite_se_v2", "pedilite_se_v2_none"} or name.startswith(("pedilite_se_w", "pedilite_se_v2_w")):
            use_se = not name.endswith("_none")
            width_name = name.replace("_none", "")
            width_mult = parse_pedilite_se_name(width_name)
            return PediLiteSE(num_classes=num_classes, dropout=dropout, width_mult=width_mult, use_se=use_se)
        attention, width_mult = parse_pedilite_name(name)
        return PediLiteAttnNet(num_classes=num_classes, attention=attention, dropout=dropout, width_mult=width_mult)
    if name in {
        "mobilenet_v2",
        "mobilenet_v3_small",
        "shufflenet_v2_x1_0",
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
        if name == "mobilenet_v2":
            weights = tvm.MobileNet_V2_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.mobilenet_v2(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if name == "mobilenet_v3_small":
            weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.mobilenet_v3_small(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            return model
        if name == "shufflenet_v2_x1_0":
            weights = tvm.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained_baselines else None
            model = tvm.shufflenet_v2_x1_0(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
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
