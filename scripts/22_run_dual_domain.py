"""Train and evaluate a shared PediLite encoder with two dataset heads.

The Mendeley head predicts normal/viral/bacterial. The VinDr head predicts
non-pneumonia/pneumonia. Dataset-specific train/validation/test labels are
required; the VinDr test set is never used for checkpoint or loss selection.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from pedilite.data import PediatricCXRDataset, class_names_for, split_roots
from pedilite.dual_domain import VinDrPCXRTrainDataset, build_dual_domain_model
from pedilite.metrics import softmax
from pedilite.train import evaluate_from_logits, fit_temperature, write_prediction_csv
from pedilite.utils import load_json, resolve_device, save_json, set_seed


def _loaders(config: dict):
    import torch
    from torch.utils.data import DataLoader

    image_size = int(config.get("image_size", 224))
    label_paths = [Path(config[f"vindr_{split}_labels"]).resolve() for split in ("train", "val", "test")]
    if len(set(label_paths)) != len(label_paths):
        raise ValueError("VinDr train, validation, and test labels must be separate files")
    missing = [str(path) for path in label_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing VinDr partition label files: {missing}")
    md_root = split_roots(config["mendeley_root"])
    md_ds = {
        split: PediatricCXRDataset(md_root[split], "three_class", image_size, augment=(split == "train"))
        for split in ("train", "val", "test")
    }
    vd_ds = {
        split: VinDrPCXRTrainDataset(
            config["vindr_image_dir"],
            config[f"vindr_{split}_labels"],
            image_size,
        )
        for split in ("train", "val", "test")
    }
    if any(len(ds) == 0 for ds in [*md_ds.values(), *vd_ds.values()]):
        raise ValueError("all Mendeley and VinDr splits must contain labeled samples")
    seed = int(config.get("seed", 42))
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": int(config.get("num_workers", 2)),
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    batch_size = int(config.get("batch_size", 32))
    return {
        "mendeley": {
            "train": DataLoader(md_ds["train"], batch_size=batch_size, shuffle=True, **common),
            "val": DataLoader(md_ds["val"], batch_size=batch_size, shuffle=False, **common),
            "test": DataLoader(md_ds["test"], batch_size=batch_size, shuffle=False, **common),
        },
        "vindr": {
            "train": DataLoader(vd_ds["train"], batch_size=batch_size, shuffle=True, **common),
            "val": DataLoader(vd_ds["val"], batch_size=batch_size, shuffle=False, **common),
            "test": DataLoader(vd_ds["test"], batch_size=batch_size, shuffle=False, **common),
        },
    }


def _class_weights(loader, num_classes: int, power: float, device):
    import torch

    labels = []
    dataset = loader.dataset
    if hasattr(dataset, "samples"):
        labels = [sample.label for sample in dataset.samples]
    elif hasattr(dataset, "rows"):
        labels = dataset.rows["target"].astype(int).tolist()
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"each class must occur in training data, got counts={counts.tolist()}")
    weights = (counts.sum() / (num_classes * counts)) ** float(power)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _collect(model, loader, domain: str, device):
    import torch

    model.eval()
    logits, labels, paths = [], [], []
    with torch.no_grad():
        for images, y, image_paths in loader:
            logits.append(model(images.to(device), domain).cpu().numpy())
            labels.append(y.numpy())
            paths.extend(str(p) for p in image_paths)
    return np.concatenate(logits), np.concatenate(labels), paths


def _batch_stream(loader):
    """Repeat a loader without caching its batches in memory."""
    while True:
        yield from loader


def _write_domain_metrics(out_dir: Path, model, loaders, domain: str, device, temperature: float, class_names: list[str]):
    logits, labels, paths = _collect(model, loaders[domain]["test"], domain, device)
    report = evaluate_from_logits(logits, labels, len(class_names), temperature)
    probs = softmax(logits / max(temperature, 1e-6))
    write_prediction_csv(out_dir / f"{domain}_test_predictions.csv", paths, labels, probs, class_names)
    return report


def run(config: dict) -> dict:
    import torch

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    loaders = _loaders(config)
    model = build_dual_domain_model(
        attention=str(config.get("attention", "se")),
        dropout=float(config.get("dropout", 0.2)),
        width_mult=float(config.get("width_mult", 1.0)),
    ).to(device)
    md_weight = _class_weights(
        loaders["mendeley"]["train"], 3, float(config.get("class_weight_power", 0.5)), device
    )
    vd_weight = _class_weights(
        loaders["vindr"]["train"], 2, float(config.get("class_weight_power", 0.5)), device
    )
    criterion_md = torch.nn.CrossEntropyLoss(weight=md_weight)
    criterion_vd = torch.nn.CrossEntropyLoss(weight=vd_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    out_dir = Path(config.get("results_dir", "results/dual_domain")) / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(config.get("steps_per_epoch", 0))
    if steps <= 0:
        steps = max(len(loaders["mendeley"]["train"]), len(loaders["vindr"]["train"]))
    vindr_loss_weight = float(config.get("vindr_loss_weight", 0.5))
    best_score, best_epoch = float("-inf"), 0
    best_path = out_dir / "best_model.pt"
    history = []
    for epoch in range(1, int(config.get("num_epochs", 30)) + 1):
        model.train()
        md_iter, vd_iter = _batch_stream(loaders["mendeley"]["train"]), _batch_stream(loaders["vindr"]["train"])
        losses = []
        for _ in range(steps):
            md_images, md_labels, _ = next(md_iter)
            vd_images, vd_labels, _ = next(vd_iter)
            md_logits = model(md_images.to(device), "mendeley")
            vd_logits = model(vd_images.to(device), "vindr")
            loss = (1.0 - vindr_loss_weight) * criterion_md(md_logits, md_labels.to(device))
            loss = loss + vindr_loss_weight * criterion_vd(vd_logits, vd_labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        md_val_logits, md_val_labels, _ = _collect(model, loaders["mendeley"]["val"], "mendeley", device)
        vd_val_logits, vd_val_labels, _ = _collect(model, loaders["vindr"]["val"], "vindr", device)
        md_val = evaluate_from_logits(md_val_logits, md_val_labels, 3)
        vd_val = evaluate_from_logits(vd_val_logits, vd_val_labels, 2)
        score = (1.0 - vindr_loss_weight) * md_val["macro_f1"] + vindr_loss_weight * vd_val["macro_f1"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mendeley_macro_f1": md_val["macro_f1"], "val_vindr_macro_f1": vd_val["macro_f1"], "val_composite": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), best_path)
    if best_epoch == 0:
        raise RuntimeError("no dual-domain validation checkpoint was selected")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    md_val_logits, md_val_labels, _ = _collect(model, loaders["mendeley"]["val"], "mendeley", device)
    vd_val_logits, vd_val_labels, _ = _collect(model, loaders["vindr"]["val"], "vindr", device)
    md_temp, vd_temp = fit_temperature(md_val_logits, md_val_labels), fit_temperature(vd_val_logits, vd_val_labels)
    md_report = _write_domain_metrics(out_dir, model, loaders, "mendeley", device, md_temp, class_names_for("three_class"))
    vd_report = _write_domain_metrics(out_dir, model, loaders, "vindr", device, vd_temp, class_names_for("binary"))
    metrics = {
        "run_name": config["run_name"],
        "model": "pedilite_dual_domain",
        "seed": int(config.get("seed", 42)),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_composite": best_score,
        "temperatures": {"mendeley": md_temp, "vindr": vd_temp},
        "test": {"mendeley": md_report, "vindr": vd_report},
        "history": history,
    }
    save_json(config, out_dir / "config_resolved.json")
    save_json(metrics, out_dir / "metrics.json")
    torch.save(model.state_dict(), out_dir / "model.pt")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    for model_seed in config.get("seeds", [config.get("seed", 42)]):
        run_config = copy.deepcopy(config)
        run_config["seed"] = int(model_seed)
        run_config["run_name"] = f"{config['run_name']}_seed{model_seed}"
        run_config.pop("seeds", None)
        print(run(run_config))


if __name__ == "__main__":
    main()
