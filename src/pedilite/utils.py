from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, image_size: int = 224, device=None) -> int:
    import torch
    from torch import nn

    if device is None:
        device = next(model.parameters()).device

    flops = 0
    hooks = []

    def conv_hook(module, _inputs, output):
        nonlocal flops
        out = output
        batch, out_ch, out_h, out_w = out.shape
        kernel_h, kernel_w = module.kernel_size
        in_ch = module.in_channels
        groups = module.groups
        flops += int(batch * out_ch * out_h * out_w * (in_ch // groups) * kernel_h * kernel_w)

    def linear_hook(module, _inputs, output):
        nonlocal flops
        batch = output.shape[0] if output.ndim > 1 else 1
        flops += int(batch * module.in_features * module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        model(dummy)
    if was_training:
        model.train()
    for hook in hooks:
        hook.remove()
    return flops


def measure_latency_ms(model, image_size: int = 224, device=None, warmup: int = 10, repeats: int = 30) -> float:
    import time
    import torch

    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(dummy)
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    if was_training:
        model.train()
    return float(elapsed * 1000.0 / max(repeats, 1))
