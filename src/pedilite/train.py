from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .data import PediatricCXRDataset, audit_dataset, class_names_for, choose_label_mode, split_roots
from .metrics import brier_score_multiclass, classification_report, expected_calibration_error, macro_auroc, softmax
from .models import build_model
from .utils import count_parameters, estimate_flops, measure_latency_ms, resolve_device, save_json, set_seed


def make_loaders(config: dict, mode: str):
    import torch
    from torch.utils.data import DataLoader

    roots = split_roots(config["data_root"])
    missing = [name for name, root in roots.items() if not root.exists()]
    if missing:
        raise FileNotFoundError(f"missing split directories under data_root: {missing}")
    datasets = {
        "train": PediatricCXRDataset(roots["train"], mode, config["image_size"], augment=True),
        "val": PediatricCXRDataset(roots["val"], mode, config["image_size"], augment=False),
        "test": PediatricCXRDataset(roots["test"], mode, config["image_size"], augment=False),
    }
    if any(len(ds) == 0 for ds in datasets.values()):
        sizes = {k: len(v) for k, v in datasets.items()}
        raise ValueError(f"empty split after label inference: {sizes}")

    seed = int(config.get("seed", 42))
    train_fraction = float(config.get("train_fraction", 1.0))
    if 0.0 < train_fraction < 1.0:
        labels = np.asarray([sample.label for sample in datasets["train"].samples])
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        for label in sorted(set(labels.tolist())):
            label_indices = np.flatnonzero(labels == label)
            n_keep = max(1, int(round(len(label_indices) * train_fraction)))
            selected.extend(rng.choice(label_indices, size=n_keep, replace=False).tolist())
        selected = sorted(selected)
        datasets["train"] = torch.utils.data.Subset(datasets["train"], selected)

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)

    loaders = {
        key: DataLoader(
            ds,
            batch_size=config["batch_size"],
            shuffle=(key == "train"),
            num_workers=config.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
        )
        for key, ds in datasets.items()
    }
    return datasets, loaders


def run_epoch(model, loader, criterion, optimizer, device):
    import torch

    model.train()
    losses = []
    for images, labels, _paths in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def collect_logits(model, loader, device):
    import torch

    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for images, labels, _paths in loader:
            logits = model(images.to(device)).detach().cpu().numpy()
            logits_all.append(logits)
            labels_all.append(labels.numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)


def collect_logits_with_paths(model, loader, device):
    """Collect logits together with stable image paths for validation-only analyses."""
    import torch

    model.eval()
    logits_all, labels_all, paths_all = [], [], []
    with torch.no_grad():
        for images, labels, paths in loader:
            logits_all.append(model(images.to(device)).detach().cpu().numpy())
            labels_all.append(labels.numpy())
            paths_all.extend(str(path) for path in paths)
    return (
        np.concatenate(logits_all, axis=0),
        np.concatenate(labels_all, axis=0),
        paths_all,
    )


def write_prediction_csv(path: str | Path, paths: list[str], labels: np.ndarray, probs: np.ndarray, class_names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label_id", "label_name", "pred_id", "pred_name"] + [f"prob_{name}" for name in class_names]
    pred = probs.argmax(axis=1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, image_path in enumerate(paths):
            row = {
                "path": image_path,
                "label_id": int(labels[idx]),
                "label_name": class_names[int(labels[idx])],
                "pred_id": int(pred[idx]),
                "pred_name": class_names[int(pred[idx])],
            }
            row.update({f"prob_{name}": float(probs[idx, class_idx]) for class_idx, name in enumerate(class_names)})
            writer.writerow(row)


def _training_labels(dataset) -> np.ndarray:
    """Return labels for a Dataset or a torch Subset without changing sampling."""
    if hasattr(dataset, "samples"):
        return np.asarray([sample.label for sample in dataset.samples], dtype=np.int64)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent = _training_labels(dataset.dataset)
        return parent[np.asarray(dataset.indices, dtype=np.int64)]
    raise TypeError("cannot infer training labels for class-weighted loss")


def build_criterion(config: dict, train_dataset, num_classes: int):
    """Build an optional class-balanced/focal objective from training labels only."""
    import torch

    loss_name = str(config.get("loss", "cross_entropy")).lower()
    if loss_name not in {"cross_entropy", "weighted_cross_entropy", "focal"}:
        raise ValueError("loss must be 'cross_entropy', 'weighted_cross_entropy', or 'focal'")
    if loss_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()

    labels = _training_labels(train_dataset)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"every class must occur in the training split, got counts={counts.tolist()}")
    power = float(config.get("class_weight_power", 1.0))
    if power < 0:
        raise ValueError("class_weight_power must be non-negative")
    weights = (counts.sum() / (num_classes * counts)) ** power
    weights /= weights.mean()
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if loss_name == "weighted_cross_entropy":
        def weighted_cross_entropy(logits, labels):
            return torch.nn.functional.cross_entropy(
                logits,
                labels,
                weight=weight_tensor.to(logits.device),
            )

        return weighted_cross_entropy

    gamma = float(config.get("focal_gamma", 2.0))
    if gamma < 0:
        raise ValueError("focal_gamma must be non-negative")

    def focal_loss(logits, labels):
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        class_weight = weight_tensor.to(logits.device)[labels]
        return (((1.0 - pt) ** gamma) * ce * class_weight).mean()

    return focal_loss


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    temps = np.linspace(0.5, 5.0, 91)
    best_temp, best_nll = 1.0, float("inf")
    for temp in temps:
        probs = softmax(logits / temp)
        nll = -np.log(np.clip(probs[np.arange(len(labels)), labels.astype(int)], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_temp, best_nll = float(temp), float(nll)
    return best_temp


def evaluate_from_logits(logits: np.ndarray, labels: np.ndarray, num_classes: int, temperature: float = 1.0) -> dict:
    probs = softmax(logits / temperature)
    pred = probs.argmax(axis=1)
    report = classification_report(labels, pred, num_classes)
    report["auroc_macro"] = macro_auroc(labels, probs, num_classes)
    report["ece"] = expected_calibration_error(probs, labels)
    report["brier"] = brier_score_multiclass(probs, labels, num_classes)
    report["temperature"] = float(temperature)
    return report


def run_experiment(config: dict) -> dict:
    import torch

    set_seed(int(config.get("seed", 42)))
    data_root = Path(config["data_root"])
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    label_mode = choose_label_mode(data_root, config.get("label_mode", "auto"))
    class_names = class_names_for(label_mode)
    num_classes = len(class_names)
    datasets, loaders = make_loaders(config, label_mode)
    device = resolve_device(config.get("device", "auto"))

    model = build_model(
        config["model"],
        num_classes,
        dropout=float(config.get("dropout", 0.2)),
        pretrained_baselines=bool(config.get("pretrained_baselines", True)),
    ).to(device)
    criterion = build_criterion(config, datasets["train"], num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    run_dir = Path(config.get("results_dir", "results")) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(audit_dataset(data_root, config.get("label_mode", "auto")), run_dir / "label_audit.json")

    checkpoint_metric = str(config.get("checkpoint_metric", "val_macro_f1"))
    supported_checkpoints = {"val_macro_f1", "val_loss", "val_viral_recall", "val_viral_f1"}
    if checkpoint_metric not in supported_checkpoints:
        raise ValueError(f"checkpoint_metric must be one of {sorted(supported_checkpoints)}")
    maximize = checkpoint_metric != "val_loss"
    best_score = float("-inf") if maximize else float("inf")
    best_epoch = 0
    best_checkpoint_path = run_dir / "best_model.pt"
    history = []
    for epoch in range(1, int(config.get("num_epochs", 5)) + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, device)
        val_logits, val_labels = collect_logits(model, loaders["val"], device)
        val_report = evaluate_from_logits(val_logits, val_labels, num_classes)
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(
                torch.nn.functional.cross_entropy(
                    torch.from_numpy(val_logits), torch.from_numpy(val_labels)
                ).item()
            ),
            **{f"val_{k}": v for k, v in val_report.items() if isinstance(v, (int, float))},
        }
        if num_classes >= 3:
            history_row["val_viral_recall"] = float(val_report["per_class"][1]["recall"])
            history_row["val_viral_f1"] = float(val_report["per_class"][1]["f1"])
        history.append(history_row)

        score = float(history_row[checkpoint_metric])
        improved = score > best_score if maximize else score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)

    if best_epoch == 0:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
    val_logits, val_labels, val_paths = collect_logits_with_paths(model, loaders["val"], device)
    test_logits, test_labels, test_paths = collect_logits_with_paths(model, loaders["test"], device)
    temperature = fit_temperature(val_logits, val_labels) if config.get("calibrate", True) else 1.0
    if bool(config.get("save_predictions", True)):
        write_prediction_csv(
            run_dir / "validation_predictions.csv",
            val_paths,
            val_labels,
            softmax(val_logits / temperature),
            class_names,
        )
        write_prediction_csv(
            run_dir / "test_predictions.csv",
            test_paths,
            test_labels,
            softmax(test_logits / temperature),
            class_names,
        )
    test_report = evaluate_from_logits(test_logits, test_labels, num_classes, temperature=temperature)

    metrics = {
        "run_name": config["run_name"],
        "model": config["model"],
        "seed": int(config.get("seed", 42)),
        "label_mode": label_mode,
        "class_names": class_names,
        "num_classes": num_classes,
        "dataset_sizes": {k: len(v) for k, v in datasets.items()},
        "parameters": count_parameters(model),
        "flops": estimate_flops(model, int(config.get("image_size", 224)), device),
        "latency_ms": measure_latency_ms(model, int(config.get("image_size", 224)), device),
        "history": history,
        "checkpoint_metric": checkpoint_metric,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "last_epoch": int(config.get("num_epochs", 5)),
        "test": test_report,
    }
    save_json(config, run_dir / "config_resolved.json")
    save_json(metrics, run_dir / "metrics.json")
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_metrics_csv(metrics, run_dir / "metrics.csv")
    return metrics


def write_metrics_csv(metrics: dict, path: str | Path) -> None:
    row = {
        "run_name": metrics["run_name"],
        "model": metrics["model"],
        "seed": metrics["seed"],
        "label_mode": metrics["label_mode"],
        "parameters": metrics["parameters"],
        "accuracy": metrics["test"]["accuracy"],
        "macro_f1": metrics["test"]["macro_f1"],
        "macro_recall": metrics["test"]["macro_recall"],
        "macro_specificity": metrics["test"]["macro_specificity"],
        "auroc_macro": metrics["test"].get("auroc_macro"),
        "ece": metrics["test"]["ece"],
        "brier": metrics["test"]["brier"],
        "temperature": metrics["test"]["temperature"],
        "flops": metrics["flops"],
        "latency_ms": metrics["latency_ms"],
    }
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
