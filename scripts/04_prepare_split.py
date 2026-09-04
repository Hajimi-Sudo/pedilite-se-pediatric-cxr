from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from pedilite.data import IMAGE_EXTENSIONS, audit_dataset, choose_label_mode, infer_label, iter_images
from pedilite.utils import save_json


def image_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def find_dataset_root(base: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in [base, *base.rglob("*")]:
        if not path.is_dir():
            continue
        train = path / "train"
        test = path / "test"
        if train.exists() and test.exists():
            count = image_count(train) + image_count(test) + image_count(path / "val")
            if count > 100:
                candidates.append((count, path))
    if not candidates:
        raise FileNotFoundError(f"could not find a train/test image dataset under {base}")
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def collect_by_label(paths: list[Path], mode: str) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for root in paths:
        if not root.exists():
            continue
        for image_path in iter_images(root):
            label = infer_label(image_path, mode)
            if label is not None:
                grouped[label].append(image_path)
    return dict(grouped)


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def materialize(grouped: dict[str, list[Path]], out_root: Path, split: str, copy_files: bool) -> dict[str, int]:
    counts = {}
    for label, paths in sorted(grouped.items()):
        counts[label] = len(paths)
        for idx, src in enumerate(sorted(paths)):
            safe_name = f"{split}_{idx:06d}_{src.name}"
            link_or_copy(src, out_root / split / label / safe_name, copy_files)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--label-mode", default="auto", choices=["auto", "binary", "three_class"])
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_base = Path(args.raw_root)
    dataset_root = find_dataset_root(raw_base)
    out_root = Path(args.out_root)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mode = choose_label_mode(dataset_root, args.label_mode)
    train_pool = collect_by_label([dataset_root / "train", dataset_root / "val", dataset_root / "valid", dataset_root / "validation"], mode)
    test_grouped = collect_by_label([dataset_root / "test"], mode)

    rng = random.Random(args.seed)
    train_grouped: dict[str, list[Path]] = {}
    val_grouped: dict[str, list[Path]] = {}
    for label, paths in sorted(train_pool.items()):
        paths = list(paths)
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * args.val_fraction)))
        val_grouped[label] = sorted(paths[:n_val])
        train_grouped[label] = sorted(paths[n_val:])

    counts = {
        "source_root": str(dataset_root),
        "out_root": str(out_root),
        "label_mode": mode,
        "train": materialize(train_grouped, out_root, "train", args.copy_files),
        "val": materialize(val_grouped, out_root, "val", args.copy_files),
        "test": materialize(test_grouped, out_root, "test", args.copy_files),
    }
    counts["audit"] = audit_dataset(out_root, mode)
    save_json(counts, out_root / "split_manifest.json")
    print(counts)


if __name__ == "__main__":
    main()
