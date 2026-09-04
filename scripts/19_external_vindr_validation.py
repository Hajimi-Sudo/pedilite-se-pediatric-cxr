from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pedilite.metrics import (
    brier_score_multiclass,
    classification_report,
    expected_calibration_error,
    macro_auroc,
    softmax,
)
from pedilite.models import build_model
from pedilite.utils import load_json, resolve_device, save_json


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VinDrPCXRDataset:
    def __init__(self, image_dir: str | Path, labels_path: str | Path, image_size: int = 224):
        try:
            import pydicom  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("pydicom is required for VinDr-PCXR evaluation") from exc

        self.image_dir = Path(image_dir)
        self.image_size = image_size
        labels = pd.read_csv(labels_path)
        required = {"image_id", "Pneumonia"}
        missing = required.difference(labels.columns)
        if missing:
            raise ValueError(f"missing VinDr label columns: {sorted(missing)}")
        if labels["image_id"].duplicated().any():
            raise ValueError("image_labels_test.csv contains duplicate image_id values")

        files = {path.stem: path for path in self.image_dir.rglob("*.dicom")}
        labels = labels[labels["image_id"].isin(files)].copy()
        if labels.empty:
            raise ValueError(f"no labeled DICOM files found under {self.image_dir}")
        labels["target"] = (labels["Pneumonia"].astype(float) > 0).astype(np.int64)
        self.rows = labels.reset_index(drop=True)
        self.files = files

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
        arr = (arr - MEAN) / STD
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), torch.tensor(int(row["target"]), dtype=torch.long), image_id


def evaluate_run(run_dir: Path, dataset: VinDrPCXRDataset, device_name: str, batch_size: int) -> dict:
    import torch
    from torch.utils.data import DataLoader

    config = load_json(run_dir / "config_resolved.json")
    stored_metrics = load_json(run_dir / "metrics.json")
    class_names = list(config.get("class_names", ["normal", "viral_pneumonia", "bacterial_pneumonia"]))
    if len(class_names) not in {2, 3} or "normal" not in class_names:
        raise ValueError(f"expected a binary or three-class run, got class_names={class_names}")

    model = build_model(
        str(config["model"]),
        num_classes=len(class_names),
        dropout=float(config.get("dropout", 0.2)),
        pretrained_baselines=False,
    ).to(resolve_device(device_name))
    device = next(model.parameters()).device
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_all, labels_all, image_ids = [], [], []
    with torch.no_grad():
        for images, labels, ids in loader:
            logits_all.append(model(images.to(device)).cpu().numpy())
            labels_all.append(labels.numpy())
            image_ids.extend(ids)

    logits = np.concatenate(logits_all, axis=0)
    y_true = np.concatenate(labels_all, axis=0)
    temperature = float(stored_metrics.get("test", {}).get("temperature", 1.0))
    probs = softmax(logits / max(temperature, 1e-6))
    normal_idx = class_names.index("normal")
    pneumonia_indices = [idx for idx in range(len(class_names)) if idx != normal_idx]
    pneumonia_prob = probs[:, pneumonia_indices].sum(axis=1)
    y_pred = (probs.argmax(axis=1) != normal_idx).astype(np.int64)
    binary_probs = np.column_stack([1.0 - pneumonia_prob, pneumonia_prob])

    report = classification_report(y_true, y_pred, 2)
    report["class_names"] = ["non_pneumonia", "pneumonia"]
    report["auroc"] = macro_auroc(y_true, binary_probs, 2)
    report["ece"] = expected_calibration_error(binary_probs, y_true)
    report["brier"] = brier_score_multiclass(binary_probs, y_true, 2)
    return {
        "run_name": stored_metrics["run_name"],
        "model": stored_metrics["model"],
        "seed": int(stored_metrics["seed"]),
        "n_images": int(len(y_true)),
        "n_pneumonia": int(y_true.sum()),
        "n_non_pneumonia": int((1 - y_true).sum()),
        "preprocessing": {
            "dicom_scale": "per-image 1st-99th percentile to uint8",
            "resize_interpolation": "bicubic",
            "temperature": temperature,
        },
        "metrics": report,
        "image_ids": image_ids,
        "pneumonia_probability": pneumonia_prob.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def write_summary(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        save_json(result, output_dir / f"{result['run_name']}.json")
        metrics = result["metrics"]
        rows.append({
            "run_name": result["run_name"],
            "model": result["model"],
            "seed": result["seed"],
            "n_images": result["n_images"],
            "n_pneumonia": result["n_pneumonia"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "sensitivity": metrics["per_class"][1]["recall"],
            # For the pneumonia-positive binary task, specificity is the
            # one-vs-rest specificity of the pneumonia class.
            "specificity": metrics["per_class"][1]["specificity"],
            "auroc": metrics["auroc"],
            "ece": metrics["ece"],
            "brier": metrics["brier"],
        })
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json({"results": rows}, output_dir / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    dataset = VinDrPCXRDataset(args.image_dir, args.labels, args.image_size)
    results = [
        evaluate_run(Path(run_dir), dataset, args.device, args.batch_size)
        for run_dir in args.run_dirs
    ]
    write_summary(results, Path(args.out_dir))
    print(f"evaluated {len(results)} frozen runs on {len(dataset)} VinDr-PCXR test images")


if __name__ == "__main__":
    main()
