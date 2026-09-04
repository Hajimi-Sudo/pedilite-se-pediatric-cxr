from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-glob", default="results/*/metrics.csv")
    parser.add_argument("--out", default="results/performance_efficiency.png")
    args = parser.parse_args()

    files = list(Path(".").glob(args.metrics_glob))
    if not files:
        raise FileNotFoundError(f"no metrics files matched {args.metrics_glob}")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["parameters"], df["macro_f1"], s=60)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["parameters"], row["macro_f1"]), fontsize=8)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Performance-efficiency trade-off")
    ax.grid(alpha=0.25)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
