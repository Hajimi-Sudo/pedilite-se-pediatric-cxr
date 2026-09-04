from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


T_975 = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
    11: 2.2010,
    12: 2.1788,
    13: 2.1604,
    14: 2.1448,
    15: 2.1314,
    16: 2.1199,
    17: 2.1098,
    18: 2.1009,
    19: 2.0930,
    20: 2.0860,
    25: 2.0595,
    30: 2.0423,
}


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def t_critical_975(df: int) -> float:
    if df in T_975:
        return T_975[df]
    if df > 30:
        return 1.9600
    return T_975[min(T_975, key=lambda key: abs(key - df))]


def t_interval(values: list[float]) -> tuple[float, float, int, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float(arr.mean()), float(arr.mean()), 0, 0.0
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    df = len(arr) - 1
    tcrit = t_critical_975(df)
    half_width = tcrit * sd / math.sqrt(len(arr))
    return mean - half_width, mean + half_width, df, tcrit


def delta_interpretation(mean_delta: float, lo: float, hi: float) -> str:
    if mean_delta >= 0:
        direction = "The mean delta favors PediLite-SE"
    else:
        direction = "The mean delta favors MobileNetV3-small"
    if lo <= 0 <= hi:
        return (
            f"{direction}, but the seed-level t interval crosses zero; "
            "state competitive performance rather than statistical superiority."
        )
    return (
        f"{direction}, and the seed-level t interval does not cross zero; "
        "still treat this as supportive rather than definitive with only three seeds."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", default="experiment/results/rescue_summary.json")
    parser.add_argument("--out-dir", default="experiment/results/hardened/tables")
    args = parser.parse_args()

    summary = load_json(args.summary_json)
    rows = [row for row in summary["rows"] if row["model"] in {"pedilite_se", "mobilenet_v3_small"}]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    seed_rows = []
    for model, items in sorted(by_model.items()):
        vals = [float(item["macro_f1"]) for item in sorted(items, key=lambda x: int(x["seed"]))]
        lo, hi, df, tcrit = t_interval(vals)
        n_seeds = len(vals)
        seed_rows.append({
            "model": model,
            "metric": "macro_f1",
            "seeds": ",".join(str(int(item["seed"])) for item in sorted(items, key=lambda x: int(x["seed"]))),
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "df": df,
            "t_critical_975": tcrit,
            "t_ci_lower_95": lo,
            "t_ci_upper_95": hi,
            "note": (
                f"Seed-level CI is based on {n_seeds} seeds; use as caution rather than a definitive significance test."
            ),
        })

    ped = {int(row["seed"]): float(row["macro_f1"]) for row in by_model["pedilite_se"]}
    mob = {int(row["seed"]): float(row["macro_f1"]) for row in by_model["mobilenet_v3_small"]}
    common = sorted(set(ped) & set(mob))
    deltas = [ped[seed] - mob[seed] for seed in common]
    lo, hi, df, tcrit = t_interval(deltas)
    mean_delta = float(np.mean(deltas))
    paired_delta = {
        "metric": "macro_f1_delta_pedilite_minus_mobilenet",
        "seeds": ",".join(str(seed) for seed in common),
        "values": deltas,
        "mean": mean_delta,
        "sd": float(np.std(deltas, ddof=1)),
        "min": float(np.min(deltas)),
        "max": float(np.max(deltas)),
        "df": df,
        "t_critical_975": tcrit,
        "t_ci_lower_95": lo,
        "t_ci_upper_95": hi,
        "interpretation": delta_interpretation(mean_delta, lo, hi),
    }

    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "seed_level_stats.csv",
        seed_rows,
        ["model", "metric", "seeds", "mean", "sd", "min", "max", "df", "t_critical_975", "t_ci_lower_95", "t_ci_upper_95", "note"],
    )
    write_json({"seed_level_stats": seed_rows, "paired_delta": paired_delta}, out_dir / "seed_level_stats.json")
    print(json.dumps({"seed_level_stats": seed_rows, "paired_delta": paired_delta}, indent=2))


if __name__ == "__main__":
    main()
