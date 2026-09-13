from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pedilite.data import PediatricCXRDataset, split_roots
from pedilite.models import build_model
from pedilite.train import evaluate_from_logits
from pedilite.utils import estimate_flops, measure_latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="pedilite_ms")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", default="128,160,192,224,256")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, 3, pretrained_baselines=False).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    roots = split_roots(args.data_root)
    rows = []
    for size in [int(tok) for tok in args.sizes.split(",") if tok]:
        dataset = PediatricCXRDataset(roots["test"], "three_class", size, augment=False)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        logits_all, labels_all = [], []
        with torch.no_grad():
            for images, labels, _paths in loader:
                logits_all.append(model(images.to(device)).detach().cpu().numpy())
                labels_all.append(labels.numpy())
        report = evaluate_from_logits(np.concatenate(logits_all), np.concatenate(labels_all), 3)
        rows.append({
            "image_size": size,
            "n": len(dataset),
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "viral_recall": report["per_class"][1]["recall"],
            "flops": estimate_flops(model, size, device),
            "latency_ms": measure_latency_ms(model, size, device),
        })
        print(size, "f1", round(report["macro_f1"], 4), "viral", round(report["per_class"][1]["recall"], 4))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "seed": args.seed, "test": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
