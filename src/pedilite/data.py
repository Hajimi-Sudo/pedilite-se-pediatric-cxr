from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES_BINARY = ["normal", "pneumonia"]
CLASS_NAMES_THREE = ["normal", "viral_pneumonia", "bacterial_pneumonia"]


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    label_name: str


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def infer_label(path: Path, mode: str) -> str | None:
    parts = [p.lower() for p in path.parts[-5:]]
    name = path.name.lower()
    token_text = " ".join(parts + [name])
    if "normal" in token_text:
        return "normal"
    if mode == "binary":
        if "pneumonia" in token_text or "virus" in token_text or "bacteria" in token_text:
            return "pneumonia"
        return None
    if "virus" in token_text or "viral" in token_text:
        return "viral_pneumonia"
    if "bacteria" in token_text or "bacterial" in token_text:
        return "bacterial_pneumonia"
    if "pneumonia" in token_text:
        return "pneumonia" if mode == "binary" else None
    return None


def choose_label_mode(root: Path, requested: str) -> str:
    if requested in {"binary", "three_class"}:
        return requested
    labels = {infer_label(path, "three_class") for path in iter_images(root)}
    labels.discard(None)
    if {"normal", "viral_pneumonia", "bacterial_pneumonia"}.issubset(labels):
        return "three_class"
    return "binary"


def class_names_for(mode: str) -> list[str]:
    return CLASS_NAMES_THREE if mode == "three_class" else CLASS_NAMES_BINARY


def collect_samples(root: Path, mode: str) -> list[Sample]:
    class_names = class_names_for(mode)
    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    samples: list[Sample] = []
    for path in iter_images(root):
        label_name = infer_label(path, mode)
        if label_name not in label_to_id:
            continue
        samples.append(Sample(path=path, label=label_to_id[label_name], label_name=label_name))
    return samples


def audit_dataset(data_root: str | Path, requested_mode: str = "auto") -> dict:
    root = Path(data_root)
    if not root.exists():
        return {"exists": False, "data_root": str(root), "error": "data_root does not exist"}

    mode = choose_label_mode(root, requested_mode)
    splits = {}
    for split in ("train", "val", "valid", "validation", "test"):
        split_root = root / split
        if split_root.exists():
            counts = {}
            for sample in collect_samples(split_root, mode):
                counts[sample.label_name] = counts.get(sample.label_name, 0) + 1
            splits[split] = counts

    root_counts = {}
    for sample in collect_samples(root, mode):
        root_counts[sample.label_name] = root_counts.get(sample.label_name, 0) + 1

    return {
        "exists": True,
        "data_root": str(root),
        "requested_label_mode": requested_mode,
        "resolved_label_mode": mode,
        "class_names": class_names_for(mode),
        "root_counts": root_counts,
        "split_counts": splits,
        "supports_diagnostic_grading": mode == "three_class",
        "supports_clinical_severity": False
    }


class PediatricCXRDataset:
    def __init__(self, root: str | Path, mode: str, image_size: int = 224, augment: bool = False):
        self.root = Path(root)
        self.mode = mode
        self.image_size = image_size
        self.augment = augment
        self.samples = collect_samples(self.root, mode)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.augment and np.random.rand() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), torch.tensor(sample.label, dtype=torch.long), str(sample.path)


def split_roots(data_root: str | Path) -> dict[str, Path]:
    root = Path(data_root)
    val = root / "val"
    if not val.exists() and (root / "valid").exists():
        val = root / "valid"
    if not val.exists() and (root / "validation").exists():
        val = root / "validation"
    return {"train": root / "train", "val": val, "test": root / "test"}
