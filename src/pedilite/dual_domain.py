"""Shared compact encoder with dataset-specific diagnostic heads.

This module is deliberately separate from the locked single-domain model so
that the original submission results remain reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from .models import DepthwiseSeparableConv, attention_block


class PediLiteDualDomainNet(nn.Module):
    """A shared PediLite encoder with Mendeley and VinDr heads.

    The encoder is shared across domains. Each head has a small learned affine
    adapter before classification, allowing a limited domain-specific shift
    without duplicating the backbone.
    """

    def __init__(self, attention: str = "se", dropout: float = 0.2, width_mult: float = 1.0):
        super().__init__()
        if width_mult <= 0:
            raise ValueError(f"width_mult must be positive, got {width_mult}")
        channels = [max(8, int(round(ch * width_mult))) for ch in [16, 32, 64, 96]]
        self.channels = channels
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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.adapters = nn.ParameterDict({
            "mendeley": nn.ParameterDict({
                        "scale": nn.Parameter(torch.ones(channels[3])),
                        "bias": nn.Parameter(torch.zeros(channels[3])),
            }),
            "vindr": nn.ParameterDict({
                        "scale": nn.Parameter(torch.ones(channels[3])),
                        "bias": nn.Parameter(torch.zeros(channels[3])),
            }),
        })
        self.heads = nn.ModuleDict({
            "mendeley": nn.Linear(channels[3], 3),
            "vindr": nn.Linear(channels[3], 2),
        })

    def encode(self, x):
        return self.pool(self.blocks(self.stem(x))).flatten(1)

    def forward(self, x, domain: str):
        if domain not in self.heads:
            raise ValueError("domain must be 'mendeley' or 'vindr'")
        features = self.encode(x)
        adapter = self.adapters[domain]
        features = features * adapter["scale"] + adapter["bias"]
        return self.heads[domain](self.dropout(features))

    def forward_both(self, x):
        features = self.encode(x)
        outputs = {}
        for domain, head in self.heads.items():
            adapter = self.adapters[domain]
            adapted = features * adapter["scale"] + adapter["bias"]
            outputs[domain] = head(self.dropout(adapted))
        return outputs


class VinDrPCXRTrainDataset:
    """VinDr-PCXR DICOM dataset for train/validation/test partitions."""

    def __init__(self, image_dir: str | Path, labels_path: str | Path, image_size: int = 224):
        try:
            import pandas as pd  # noqa: F401
            import pydicom  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("pandas and pydicom are required for VinDr training") from exc
        import pandas as pd

        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.files = {path.stem: path for path in self.image_dir.rglob("*.dicom")}
        labels = pd.read_csv(labels_path)
        required = {"image_id", "Pneumonia"}
        missing = required.difference(labels.columns)
        if missing:
            raise ValueError(f"missing VinDr label columns: {sorted(missing)}")
        labels = labels[labels["image_id"].astype(str).isin(self.files)].copy()
        labels = labels.dropna(subset=["Pneumonia"])
        labels["target"] = (labels["Pneumonia"].astype(float) > 0).astype(np.int64)
        if labels.empty:
            raise ValueError(f"no labeled DICOM files found under {self.image_dir}")
        self.rows = labels.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch
        import pydicom

        row = self.rows.iloc[index]
        image_id = str(row["image_id"])
        dicom = pydicom.dcmread(self.files[image_id])
        array = dicom.pixel_array.astype(np.float32)
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            array = array.max() - array
        low, high = np.percentile(array, [1.0, 99.0])
        if high <= low:
            low, high = float(array.min()), float(array.max())
        array = np.clip((array - low) / max(high - low, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="L").convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), torch.tensor(int(row["target"]), dtype=torch.long), image_id


def build_dual_domain_model(attention: str = "se", dropout: float = 0.2, width_mult: float = 1.0):
    return PediLiteDualDomainNet(attention=attention, dropout=dropout, width_mult=width_mult)
