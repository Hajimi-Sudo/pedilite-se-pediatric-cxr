from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def scalar(report: dict, key: str) -> float | None:
    value = report.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in sorted(Path(args.results_dir).glob("**/metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            metric = json.load(f)
        test = metric.get("test", {})
        rows.append({
            "run_name": metric.get("run_name", path.parent.name),
            "model": metric.get("model"),
            "seed": metric.get("seed"),
            "macro_f1": scalar(test, "macro_f1"),
            "auroc_macro": scalar(test, "auroc_macro"),
            "ece": scalar(test, "ece"),
            "brier": scalar(test, "brier"),
            "parameters": metric.get("parameters"),
            "flops": metric.get("flops"),
            "latency_ms": metric.get("latency_ms"),
        })
    if not rows:
        raise SystemExit("no metrics.json files found")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    summary = []
    for model, items in sorted(grouped.items()):
        out = {"model": model, "n_seeds": len(items), "seeds": ",".join(str(x["seed"]) for x in items)}
        for key in ("macro_f1", "auroc_macro", "ece", "brier", "parameters", "flops", "latency_ms"):
            values = [float(x[key]) for x in items if x.get(key) is not None]
            out[f"{key}_mean"] = mean(values) if values else None
            out[f"{key}_sd"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        summary.append(out)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out_dir / "per_run.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_fields = list(summary[0])
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"per_run": rows, "summary": summary}, f, indent=2)
    print(json.dumps({"runs": len(rows), "models": sorted(grouped)}, indent=2))


if __name__ == "__main__":
    main()
