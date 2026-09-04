from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pedilite.data import PediatricCXRDataset, class_names_for, split_roots
from pedilite.metrics import (
    brier_score_multiclass,
    classification_report,
    confusion_matrix,
    expected_calibration_error,
    macro_auroc,
    softmax,
)
from pedilite.models import build_model
from pedilite.utils import resolve_device, save_json, set_seed


MATCHED_SEEDS = {42, 2026, 3407}
PRIMARY_MODELS = {"pedilite_se", "mobilenet_v3_small"}


@dataclass
class RunPredictions:
    run_name: str
    model: str
    seed: int
    class_names: list[str]
    paths: list[str]
    labels: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    temperature: float
    stored_metrics: dict[str, Any]

    @property
    def preds(self) -> np.ndarray:
        return self.probs.argmax(axis=1)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_seed_set(text: str | None) -> set[int]:
    if not text:
        return set(MATCHED_SEEDS)
    seeds = {int(part.strip()) for part in text.split(",") if part.strip()}
    if not seeds:
        raise ValueError("--seeds did not contain any valid integer seeds")
    return seeds


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def locate_run_dir(results_root: Path, run_name: str) -> Path:
    candidates = [
        results_root / "rescue" / run_name,
        results_root / "corrected_split" / run_name,
        results_root / "corrected_split_extended" / run_name,
        results_root / run_name,
    ]
    for path in candidates:
        if (path / "metrics.json").exists() and (path / "model.pt").exists():
            return path
    for path in results_root.glob(f"*/{run_name}"):
        if (path / "metrics.json").exists() and (path / "model.pt").exists():
            return path
    raise FileNotFoundError(f"could not locate run directory for {run_name} under {results_root}")


def rows_from_summary(summary: dict[str, Any], matched_seeds: set[int]) -> list[dict[str, Any]]:
    rows = summary.get("rows")
    if not isinstance(rows, list):
        raise ValueError("rescue_summary.json does not contain a rows list")
    selected = [
        row for row in rows
        if row.get("model") in PRIMARY_MODELS and int(row.get("seed", -1)) in matched_seeds
    ]
    expected = len(PRIMARY_MODELS) * len(matched_seeds)
    if len(selected) != expected:
        names = [row.get("run_name") for row in selected]
        raise ValueError(f"expected {expected} matched-seed rows, found {len(selected)}: {names}")
    return selected


def collect_run_predictions(run_dir: Path, device_name: str, batch_size_override: int | None) -> RunPredictions:
    import torch
    from torch.utils.data import DataLoader

    config = load_json(run_dir / "config_resolved.json")
    metrics = load_json(run_dir / "metrics.json")
    seed = int(config.get("seed", metrics.get("seed", 42)))
    set_seed(seed)

    label_mode = str(metrics["label_mode"])
    class_names = list(metrics.get("class_names") or class_names_for(label_mode))
    num_classes = len(class_names)
    roots = split_roots(config["data_root"])
    dataset = PediatricCXRDataset(roots["test"], label_mode, int(config.get("image_size", 224)), augment=False)
    if len(dataset) == 0:
        raise ValueError(f"empty test dataset for {run_dir.name}")

    batch_size = int(batch_size_override or config.get("batch_size", 32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(config.get("num_workers", 0)))
    device = resolve_device(device_name)
    model = build_model(
        str(config["model"]),
        num_classes,
        dropout=float(config.get("dropout", 0.2)),
        pretrained_baselines=False,
    ).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    paths_all: list[str] = []
    with torch.no_grad():
        for images, labels, paths in loader:
            logits = model(images.to(device)).detach().cpu().numpy()
            logits_all.append(logits)
            labels_all.append(labels.numpy())
            paths_all.extend([str(path) for path in paths])

    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    temperature = float(metrics["test"].get("temperature", 1.0))
    probs = softmax(logits / temperature)
    return RunPredictions(
        run_name=str(metrics["run_name"]),
        model=str(metrics["model"]),
        seed=int(metrics["seed"]),
        class_names=class_names,
        paths=paths_all,
        labels=labels,
        logits=logits,
        probs=probs,
        temperature=temperature,
        stored_metrics=metrics,
    )


def pneumonia_sensitivity_from_arrays(labels: np.ndarray, preds: np.ndarray, class_names: list[str]) -> float | None:
    if "pneumonia" in class_names:
        idx = class_names.index("pneumonia")
        mask = labels == idx
        return float((preds[mask] == idx).mean()) if mask.any() else None
    if "normal" in class_names and len(class_names) > 2:
        normal = class_names.index("normal")
        mask = labels != normal
        return float((preds[mask] != normal).mean()) if mask.any() else None
    return None


def report_from_probs(labels: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    preds = probs.argmax(axis=1)
    num_classes = len(class_names)
    report = classification_report(labels, preds, num_classes)
    report["auroc_macro"] = macro_auroc(labels, probs, num_classes)
    report["ece"] = expected_calibration_error(probs, labels)
    report["brier"] = brier_score_multiclass(probs, labels, num_classes)
    report["pneumonia_sensitivity"] = pneumonia_sensitivity_from_arrays(labels, preds, class_names)
    return report


def export_predictions(record: RunPredictions, data_root: Path, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pred_names = [record.class_names[int(idx)] for idx in record.preds]
    label_names = [record.class_names[int(idx)] for idx in record.labels]
    for i, path_str in enumerate(record.paths):
        path = Path(path_str)
        try:
            rel_path = str(path.relative_to(data_root))
        except ValueError:
            rel_path = str(path)
        row = {
            "run_name": record.run_name,
            "model": record.model,
            "seed": record.seed,
            "image_id": path.name,
            "relative_path": rel_path,
            "label_id": int(record.labels[i]),
            "label_name": label_names[i],
            "pred_id": int(record.preds[i]),
            "pred_name": pred_names[i],
            "correct": int(record.labels[i] == record.preds[i]),
            "confidence": float(record.probs[i].max()),
        }
        for class_idx, class_name in enumerate(record.class_names):
            row[f"prob_{class_name}"] = float(record.probs[i, class_idx])
        rows.append(row)

    fields = [
        "run_name",
        "model",
        "seed",
        "image_id",
        "relative_path",
        "label_id",
        "label_name",
        "pred_id",
        "pred_name",
        "correct",
        "confidence",
    ] + [f"prob_{name}" for name in record.class_names]
    csv_path = out_dir / "predictions" / f"{record.run_name}.csv"
    write_csv(csv_path, rows, fields)

    recomputed = report_from_probs(record.labels, record.probs, record.class_names)
    stored = record.stored_metrics["test"]
    deltas = {}
    for key in ("accuracy", "macro_f1", "auroc_macro", "ece", "brier"):
        if stored.get(key) is not None and recomputed.get(key) is not None:
            deltas[key] = abs(float(stored[key]) - float(recomputed[key]))
    return {
        "run_name": record.run_name,
        "model": record.model,
        "seed": record.seed,
        "n_test": int(len(record.labels)),
        "temperature": record.temperature,
        "prediction_csv": str(csv_path),
        "recomputed": scalar_report(recomputed),
        "stored": scalar_report(stored),
        "abs_deltas": deltas,
        "max_abs_delta": max(deltas.values()) if deltas else None,
    }


def scalar_report(report: dict[str, Any]) -> dict[str, Any]:
    keys = ("accuracy", "macro_f1", "macro_recall", "auroc_macro", "ece", "brier", "pneumonia_sensitivity")
    return {key: report.get(key) for key in keys if key in report}


def aggregate_summary(records: list[RunPredictions]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    performance_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    by_model: dict[str, list[RunPredictions]] = {}
    for record in records:
        by_model.setdefault(record.model, []).append(record)

    for model, items in sorted(by_model.items()):
        reports = [report_from_probs(item.labels, item.probs, item.class_names) for item in items]
        stored_metrics = [item.stored_metrics for item in items]
        row = {
            "model": model,
            "seeds": ",".join(str(item.seed) for item in sorted(items, key=lambda x: x.seed)),
            "n_runs": len(items),
        }
        for key in ("accuracy", "macro_f1", "auroc_macro", "ece", "brier", "pneumonia_sensitivity"):
            vals = [float(report[key]) for report in reports if report.get(key) is not None]
            if vals:
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        performance_rows.append(row)

        efficiency_rows.append({
            "model": model,
            "parameters": int(stored_metrics[0]["parameters"]),
            "flops": int(stored_metrics[0].get("flops", 0)),
            "latency_ms_mean": float(np.mean([float(m.get("latency_ms", math.nan)) for m in stored_metrics])),
            "latency_ms_sd": float(np.std([float(m.get("latency_ms", math.nan)) for m in stored_metrics], ddof=1)),
        })

    if len(efficiency_rows) == 2:
        by_name = {row["model"]: row for row in efficiency_rows}
        if "pedilite_se" in by_name and "mobilenet_v3_small" in by_name:
            ped = by_name["pedilite_se"]
            base = by_name["mobilenet_v3_small"]
            ped["parameter_reduction_vs_mobilenet"] = 1.0 - ped["parameters"] / base["parameters"]
            ped["flops_reduction_vs_mobilenet"] = 1.0 - ped["flops"] / base["flops"]
            ped["latency_reduction_vs_mobilenet"] = 1.0 - ped["latency_ms_mean"] / base["latency_ms_mean"]
    return performance_rows, efficiency_rows


def bootstrap_ci(records: list[RunPredictions], repeats: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_model: dict[str, list[RunPredictions]] = {}
    for record in records:
        by_model.setdefault(record.model, []).append(record)

    rows: list[dict[str, Any]] = []
    metrics = ("macro_f1", "auroc_macro", "pneumonia_sensitivity", "ece", "brier")
    for model, items in sorted(by_model.items()):
        n = len(items[0].labels)
        full_reports = [report_from_probs(item.labels, item.probs, item.class_names) for item in items]
        full_means = {
            metric: float(np.nanmean([report.get(metric, np.nan) for report in full_reports]))
            for metric in metrics
        }
        boot_values: dict[str, list[float]] = {metric: [] for metric in metrics}
        for _ in range(repeats):
            idx = rng.integers(0, n, size=n)
            sampled_reports = [report_from_probs(item.labels[idx], item.probs[idx], item.class_names) for item in items]
            for metric in metrics:
                vals = [report.get(metric, np.nan) for report in sampled_reports]
                vals = [float(value) for value in vals if value is not None and math.isfinite(float(value))]
                if vals:
                    boot_values[metric].append(float(np.mean(vals)))
        for metric in metrics:
            vals = np.asarray(boot_values[metric], dtype=np.float64)
            if len(vals) == 0:
                continue
            rows.append({
                "model": model,
                "metric": metric,
                "mean": full_means[metric],
                "ci_lower": float(np.percentile(vals, 2.5)),
                "ci_upper": float(np.percentile(vals, 97.5)),
                "n_bootstrap": int(len(vals)),
            })
    return rows


def make_confusion_figure(records: list[RunPredictions], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    by_model: dict[str, list[RunPredictions]] = {}
    for record in records:
        by_model.setdefault(record.model, []).append(record)
    models = [model for model in ("pedilite_se", "mobilenet_v3_small") if model in by_model]
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 4.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        items = by_model[model]
        class_names = items[0].class_names
        pooled = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for item in items:
            pooled += confusion_matrix(item.labels, item.preds, len(class_names))
        im = ax.imshow(pooled, cmap="Blues")
        ax.set_title(model)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_names)), short_names(class_names), rotation=25, ha="right")
        ax.set_yticks(range(len(class_names)), short_names(class_names))
        for i in range(pooled.shape[0]):
            for j in range(pooled.shape[1]):
                ax.text(j, i, str(int(pooled[i, j])), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def make_reliability_figure(records: list[RunPredictions], out_path: Path, n_bins: int = 10) -> None:
    import matplotlib.pyplot as plt

    by_model: dict[str, list[RunPredictions]] = {}
    for record in records:
        by_model.setdefault(record.model, []).append(record)
    models = [model for model in ("pedilite_se", "mobilenet_v3_small") if model in by_model]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), squeeze=False)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for ax, model in zip(axes[0], models):
        items = by_model[model]
        confidences = np.concatenate([item.probs.max(axis=1) for item in items])
        correct = np.concatenate([(item.preds == item.labels).astype(np.float32) for item in items])
        accuracies = []
        conf_means = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.any():
                accuracies.append(float(correct[mask].mean()))
                conf_means.append(float(confidences[mask].mean()))
            else:
                accuracies.append(np.nan)
                conf_means.append(np.nan)
        ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1)
        ax.bar(centers, np.nan_to_num(accuracies, nan=0.0), width=1 / n_bins * 0.9, alpha=0.65)
        ax.plot(conf_means, accuracies, marker="o", color="#1f77b4", linewidth=1.5)
        ece = expected_calibration_error(
            np.concatenate([item.probs for item in items]),
            np.concatenate([item.labels for item in items]),
        )
        ax.set_title(f"{model}\npooled ECE={ece:.3f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def make_performance_efficiency_figure(performance_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    perf = {row["model"]: row for row in performance_rows}
    eff = {row["model"]: row for row in efficiency_rows}
    models = [model for model in ("pedilite_se", "mobilenet_v3_small") if model in perf and model in eff]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for model in models:
        ax.scatter(eff[model]["flops"] / 1e6, perf[model]["macro_f1_mean"], s=90)
        ax.annotate(model, (eff[model]["flops"] / 1e6, perf[model]["macro_f1_mean"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("FLOPs (millions)")
    ax.set_ylabel("Macro-F1, mean of 3 seeds")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def make_gradcam_figure(run_dir: Path, out_path: Path, device_name: str, max_cases: int = 6) -> str | None:
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from PIL import Image

    config = load_json(run_dir / "config_resolved.json")
    metrics = load_json(run_dir / "metrics.json")
    label_mode = str(metrics["label_mode"])
    class_names = list(metrics.get("class_names") or class_names_for(label_mode))
    roots = split_roots(config["data_root"])
    dataset = PediatricCXRDataset(roots["test"], label_mode, int(config.get("image_size", 224)), augment=False)
    device = resolve_device(device_name)
    model = build_model(str(config["model"]), len(class_names), dropout=float(config.get("dropout", 0.2)), pretrained_baselines=False).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
    model.eval()

    conv_layers = [
        module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d) and ".fc." not in name and ".spatial." not in name
    ]
    if not conv_layers:
        return None
    target_layer = conv_layers[-1]

    pred_record = collect_run_predictions(run_dir, device_name, int(config.get("batch_size", 32)))
    selected: list[int] = []
    for class_id in range(len(class_names)):
        matches = np.flatnonzero((pred_record.labels == class_id) & (pred_record.preds == class_id))
        if len(matches):
            selected.append(int(matches[0]))
    for idx in np.flatnonzero(pred_record.labels != pred_record.preds):
        if len(selected) >= max_cases:
            break
        if int(idx) not in selected:
            selected.append(int(idx))
    selected = selected[:max_cases]
    if not selected:
        return None

    cols = min(3, len(selected))
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, idx in zip(axes_flat, selected):
        image_tensor, label_tensor, path_str = dataset[idx]
        image = image_tensor.unsqueeze(0).to(device)
        activation_holder: list[torch.Tensor] = []
        gradient_holder: list[torch.Tensor] = []

        def forward_hook(_module, _input, output):
            activation_holder.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradient_holder.append(grad_output[0])

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        model.zero_grad(set_to_none=True)
        logits = model(image)
        pred_idx = int(logits.argmax(dim=1).item())
        score = logits[0, pred_idx]
        score.backward()
        handle_fwd.remove()
        handle_bwd.remove()

        activations = activation_holder[0].detach()
        gradients = gradient_holder[0].detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / max(float(cam_np.max() - cam_np.min()), 1e-8)

        original = Image.open(path_str).convert("RGB").resize((image_tensor.shape[-1], image_tensor.shape[-2]))
        ax.imshow(original, cmap="gray")
        ax.imshow(cam_np, cmap="jet", alpha=0.35)
        true_name = class_names[int(label_tensor)]
        pred_name = class_names[pred_idx]
        ax.set_title(f"T: {short_name(true_name)} | P: {short_name(pred_name)}", fontsize=10)
        ax.axis("off")

    for ax in axes_flat[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return str(out_path)


def short_name(name: str) -> str:
    return name.replace("_pneumonia", "").replace("_", " ")


def short_names(names: list[str]) -> list[str]:
    return [short_name(name) for name in names]


def write_markdown_report(out_dir: Path, performance_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, Any]], ci_rows: list[dict[str, Any]], verification: list[dict[str, Any]]) -> None:
    perf = {row["model"]: row for row in performance_rows}
    eff = {row["model"]: row for row in efficiency_rows}
    delta = None
    if "pedilite_se" in perf and "mobilenet_v3_small" in perf:
        delta = perf["pedilite_se"]["macro_f1_mean"] - perf["mobilenet_v3_small"]["macro_f1_mean"]
    lines = [
        "# Result Hardening Report",
        "",
        "This report is eval-only. It reuses saved weights and the fixed test split; no training, split changes, or test-set threshold tuning were performed.",
        "",
        "## Main Matched-Seed Finding",
        "",
    ]
    if delta is not None:
        lines.append(f"- PediLite-SE mean macro-F1 minus MobileNetV3-small: {delta:.4f}.")
    if "pedilite_se" in eff:
        row = eff["pedilite_se"]
        if "parameter_reduction_vs_mobilenet" in row:
            lines.append(f"- PediLite-SE parameter reduction vs MobileNetV3-small: {100.0 * row['parameter_reduction_vs_mobilenet']:.1f}%.")
            lines.append(f"- PediLite-SE FLOPs reduction vs MobileNetV3-small: {100.0 * row['flops_reduction_vs_mobilenet']:.1f}%.")
    lines.extend([
        "- Claim boundary: pediatric CXR diagnostic typing / risk stratification; no clinical severity grading claim.",
        "",
        "## Generated Artifacts",
        "",
        "- `predictions/*.csv`: per-sample calibrated probabilities and predicted labels.",
        "- `bootstrap_ci.csv`: paired bootstrap 95% confidence intervals over fixed test images, averaged across matched seeds.",
        "- `tables/performance_table.csv`: seed-mean performance summary.",
        "- `tables/efficiency_table.csv`: parameter, FLOPs, and latency summary.",
        "- `figures/confusion_matrices.*`: seed-pooled confusion matrices.",
        "- `figures/reliability_diagram.*`: seed-pooled calibration plots.",
        "- `figures/gradcam_pedilite_se_seed2026.*`: qualitative Grad-CAM examples.",
        "",
        "## Verification",
        "",
        "| run | max abs delta vs stored metrics |",
        "|---|---:|",
    ])
    for item in verification:
        max_delta = item.get("max_abs_delta")
        text = "NA" if max_delta is None else f"{float(max_delta):.6g}"
        lines.append(f"| {item['run_name']} | {text} |")
    lines.extend([
        "",
        "## Bootstrap CI",
        "",
        "| model | metric | mean | 95% CI |",
        "|---|---|---:|---:|",
    ])
    for row in ci_rows:
        lines.append(
            f"| {row['model']} | {row['metric']} | {row['mean']:.4f} | [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] |"
        )
    (out_dir / "HARDENING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260613)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in sorted(MATCHED_SEEDS)))
    parser.add_argument("--gradcam-run", default="rescue_confirm_pedilite_se_seed2026")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    summary_path = Path(args.summary_json) if args.summary_json else results_root / "rescue_summary.json"
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "hardened"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_path)
    rows = rows_from_summary(summary, parse_seed_set(args.seeds))
    records: list[RunPredictions] = []
    verification: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item["model"]), int(item["seed"]))):
        run_dir = locate_run_dir(results_root, str(row["run_name"]))
        record = collect_run_predictions(run_dir, args.device, args.batch_size)
        records.append(record)
        data_root = Path(load_json(run_dir / "config_resolved.json")["data_root"])
        verification.append(export_predictions(record, data_root, out_dir))

    performance_rows, efficiency_rows = aggregate_summary(records)
    ci_rows = bootstrap_ci(records, repeats=int(args.bootstrap_repeats), seed=int(args.bootstrap_seed))

    write_csv(out_dir / "tables" / "performance_table.csv", performance_rows, sorted({key for row in performance_rows for key in row}))
    write_csv(out_dir / "tables" / "efficiency_table.csv", efficiency_rows, sorted({key for row in efficiency_rows for key in row}))
    write_csv(out_dir / "tables" / "bootstrap_ci.csv", ci_rows, ["model", "metric", "mean", "ci_lower", "ci_upper", "n_bootstrap"])
    write_csv(out_dir / "tables" / "prediction_verification.csv", verification, sorted({key for row in verification for key in row}))

    make_confusion_figure(records, out_dir / "figures" / "confusion_matrices.png")
    make_reliability_figure(records, out_dir / "figures" / "reliability_diagram.png")
    make_performance_efficiency_figure(performance_rows, efficiency_rows, out_dir / "figures" / "performance_efficiency.png")
    gradcam_dir = locate_run_dir(results_root, args.gradcam_run)
    gradcam_path = make_gradcam_figure(gradcam_dir, out_dir / "figures" / "gradcam_pedilite_se_seed2026.png", args.device)

    output = {
        "summary_source": str(summary_path),
        "out_dir": str(out_dir),
        "runs": [record.run_name for record in records],
        "performance_rows": performance_rows,
        "efficiency_rows": efficiency_rows,
        "bootstrap_ci": ci_rows,
        "prediction_verification": verification,
        "figures": {
            "confusion_matrices": str(out_dir / "figures" / "confusion_matrices.png"),
            "reliability_diagram": str(out_dir / "figures" / "reliability_diagram.png"),
            "performance_efficiency": str(out_dir / "figures" / "performance_efficiency.png"),
            "gradcam": gradcam_path,
        },
    }
    save_json(output, out_dir / "hardened_summary.json")
    write_markdown_report(out_dir, performance_rows, efficiency_rows, ci_rows, verification)
    print(json.dumps({"status": "OK", "out_dir": str(out_dir), "runs": len(records)}, indent=2))


if __name__ == "__main__":
    main()
