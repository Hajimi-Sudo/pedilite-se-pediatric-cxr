from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from pedilite.train import run_experiment
from pedilite.utils import load_json


def _try_lock(run_dir: Path) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "train.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    return True


def _unlock(run_dir: Path) -> None:
    lock_path = run_dir / "train.lock"
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_json(args.config)

    models = config.get("models")
    if models is None:
        models = [config["model"]]
    seeds = config.get("seeds") or [config.get("seed", 42)]
    summaries = []
    base_run_name = config["run_name"]
    results_dir = Path(config.get("results_dir", "results"))
    for model_name in models:
        for seed in seeds:
            run_config = copy.deepcopy(config)
            run_config["model"] = model_name
            run_config["seed"] = int(seed)
            run_config["run_name"] = f"{base_run_name}_{model_name}_seed{seed}"
            run_config.pop("models", None)
            run_config.pop("seeds", None)
            run_dir = results_dir / run_config["run_name"]
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                metrics = load_json(metrics_path)
                print(f"[skip] {run_config['run_name']} already has {metrics_path}")
            elif not _try_lock(run_dir):
                print(f"[skip] {run_config['run_name']} is locked by another worker")
                continue
            else:
                try:
                    metrics = run_experiment(run_config)
                finally:
                    _unlock(run_dir)
            summaries.append({
                "run_name": metrics["run_name"],
                "model": metrics["model"],
                "label_mode": metrics["label_mode"],
                "macro_f1": metrics["test"]["macro_f1"],
                "auroc_macro": metrics["test"].get("auroc_macro"),
                "ece": metrics["test"]["ece"],
                "parameters": metrics["parameters"],
                "flops": metrics.get("flops"),
                "latency_ms": metrics.get("latency_ms"),
            })
    print(summaries)


if __name__ == "__main__":
    main()
