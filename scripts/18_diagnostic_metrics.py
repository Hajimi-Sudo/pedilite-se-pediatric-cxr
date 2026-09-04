from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_seed_filter(text: str | None) -> set[int] | None:
    if not text:
        return None
    seeds = {int(part.strip()) for part in text.split(",") if part.strip()}
    return seeds or None


def class_names(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    names = [key.removeprefix("prob_") for key in rows[0] if key.startswith("prob_")]
    if names:
        return names
    return sorted({row["label_name"] for row in rows} | {row["pred_name"] for row in rows})


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def metric_from_binary(true_pos: np.ndarray, pred_pos: np.ndarray) -> dict[str, Any]:
    tp = int(np.logical_and(true_pos, pred_pos).sum())
    fn = int(np.logical_and(true_pos, ~pred_pos).sum())
    fp = int(np.logical_and(~true_pos, pred_pos).sum())
    tn = int(np.logical_and(~true_pos, ~pred_pos).sum())
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support_positive": int(true_pos.sum()),
        "support_negative": int((~true_pos).sum()),
        "predicted_positive": int(pred_pos.sum()),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
    }


def diagnostic_rows_for_run(run_rows: list[dict[str, str]], sample_idx: np.ndarray | None = None) -> list[dict[str, Any]]:
    names = class_names(run_rows)
    if sample_idx is None:
        selected = run_rows
    else:
        selected = [run_rows[int(idx)] for idx in sample_idx]
    labels = np.asarray([row["label_name"] for row in selected])
    preds = np.asarray([row["pred_name"] for row in selected])
    out: list[dict[str, Any]] = []
    for name in names:
        metrics = metric_from_binary(labels == name, preds == name)
        metrics["target"] = name
        metrics["target_type"] = "one_vs_rest"
        out.append(metrics)
    if "normal" in names and len(names) > 2:
        metrics = metric_from_binary(labels != "normal", preds != "normal")
        metrics["target"] = "any_pneumonia"
        metrics["target_type"] = "pneumonia_vs_normal"
        out.append(metrics)
    return out


def finite_mean(values: list[float | None]) -> float | None:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not vals:
        return None
    return float(np.mean(vals))


def finite_sd(values: list[float | None]) -> float | None:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(vals) < 2:
        return 0.0 if vals else None
    return float(np.std(vals, ddof=1))


def summarize(per_run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run_rows:
        grouped[(str(row["model"]), str(row["target"]), str(row["target_type"]))].append(row)
    metric_names = ["sensitivity", "specificity", "ppv", "npv"]
    out = []
    for (model, target, target_type), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model": model,
            "target": target,
            "target_type": target_type,
            "n_runs": len(rows),
            "seeds": ",".join(str(int(row["seed"])) for row in sorted(rows, key=lambda item: int(item["seed"]))),
            "support_positive_mean": finite_mean([row["support_positive"] for row in rows]),
            "support_negative_mean": finite_mean([row["support_negative"] for row in rows]),
        }
        for metric in metric_names:
            summary[f"{metric}_mean"] = finite_mean([row.get(metric) for row in rows])
            summary[f"{metric}_sd"] = finite_sd([row.get(metric) for row in rows])
        out.append(summary)
    return out


def bootstrap_ci(run_map: dict[str, list[dict[str, Any]]], repeats: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_model: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for rows in run_map.values():
        if not rows:
            continue
        by_model[str(rows[0]["model"])].append(rows)

    metric_names = ["sensitivity", "specificity", "ppv", "npv"]
    ci_rows: list[dict[str, Any]] = []
    for model, runs in sorted(by_model.items()):
        n = len(runs[0])
        full_by_target: dict[tuple[str, str], dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
        for rows in runs:
            for result in diagnostic_rows_for_run(rows):
                for metric in metric_names:
                    full_by_target[(str(result["target"]), str(result["target_type"]))][metric].append(result.get(metric))

        boot_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for _ in range(repeats):
            idx = rng.integers(0, n, size=n)
            repeat_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
            for rows in runs:
                for result in diagnostic_rows_for_run(rows, idx):
                    for metric in metric_names:
                        value = result.get(metric)
                        if value is not None and math.isfinite(float(value)):
                            repeat_values[(str(result["target"]), str(result["target_type"]), metric)].append(float(value))
            for key, values in repeat_values.items():
                if values:
                    boot_values[key].append(float(np.mean(values)))

        for (target, target_type), metric_map in sorted(full_by_target.items()):
            for metric in metric_names:
                vals = np.asarray(boot_values.get((target, target_type, metric), []), dtype=np.float64)
                if len(vals) == 0:
                    continue
                ci_rows.append({
                    "model": model,
                    "target": target,
                    "target_type": target_type,
                    "metric": metric,
                    "mean": finite_mean(metric_map[metric]),
                    "ci_lower": float(np.percentile(vals, 2.5)),
                    "ci_upper": float(np.percentile(vals, 97.5)),
                    "n_bootstrap_values": int(len(vals)),
                })
    return ci_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260613)
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    out_dir = Path(args.out_dir)
    seed_filter = parse_seed_filter(args.seeds)

    run_map: dict[str, list[dict[str, str]]] = {}
    for path in sorted(predictions_dir.glob("*.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        seed = int(rows[0]["seed"])
        if seed_filter is not None and seed not in seed_filter:
            continue
        run_map[path.stem] = rows

    per_run: list[dict[str, Any]] = []
    for run_name, rows in sorted(run_map.items()):
        model = rows[0]["model"]
        seed = int(rows[0]["seed"])
        for result in diagnostic_rows_for_run(rows):
            result.update({"run_name": run_name, "model": model, "seed": seed})
            per_run.append(result)

    summary = summarize(per_run)
    ci_rows = bootstrap_ci(run_map, repeats=int(args.bootstrap_repeats), seed=int(args.bootstrap_seed))

    per_run_fields = [
        "run_name", "model", "seed", "target", "target_type",
        "tp", "fp", "tn", "fn", "support_positive", "support_negative",
        "predicted_positive", "sensitivity", "specificity", "ppv", "npv",
    ]
    summary_fields = [
        "model", "target", "target_type", "n_runs", "seeds",
        "support_positive_mean", "support_negative_mean",
        "sensitivity_mean", "sensitivity_sd", "specificity_mean", "specificity_sd",
        "ppv_mean", "ppv_sd", "npv_mean", "npv_sd",
    ]
    ci_fields = ["model", "target", "target_type", "metric", "mean", "ci_lower", "ci_upper", "n_bootstrap_values"]

    write_csv(out_dir / "diagnostic_metrics_per_run.csv", per_run, per_run_fields)
    write_csv(out_dir / "diagnostic_metrics_summary.csv", summary, summary_fields)
    write_csv(out_dir / "diagnostic_metrics_bootstrap_ci.csv", ci_rows, ci_fields)
    write_json(out_dir / "diagnostic_metrics_summary.json", {
        "per_run": per_run,
        "summary": summary,
        "bootstrap_ci": ci_rows,
    })
    print(json.dumps({"runs": len(run_map), "per_run_rows": len(per_run), "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
