from __future__ import annotations

import argparse
import copy

from pedilite.train import run_experiment
from pedilite.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_json(args.config)

    models = config.get("models") or [config["model"]]
    seeds = config.get("seeds") or [config.get("seed", 42)]
    summaries = []
    base_run_name = config["run_name"]
    for model_name in models:
        for seed in seeds:
            run_config = copy.deepcopy(config)
            run_config["model"] = model_name
            run_config["seed"] = int(seed)
            run_config["run_name"] = f"{base_run_name}_{model_name}_seed{seed}"
            run_config.pop("models", None)
            run_config.pop("seeds", None)
            metrics = run_experiment(run_config)
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
