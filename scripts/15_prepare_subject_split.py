from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


def parse_subject_like_id(path: Path) -> str | None:
    name = path.name.lower()
    stem = path.stem.lower()
    match = re.search(r"person(\d+)", name)
    if match:
        return f"person_{match.group(1)}"
    match = re.search(r"(normal\d*-im-\d+)", stem)
    if match:
        return match.group(1)
    match = re.search(r"(im-\d+)", stem)
    if match:
        return match.group(1)
    return None


def collect_grouped(dataset_root: Path, mode: str) -> dict[str, list[tuple[Path, str]]]:
    roots = [
        dataset_root / "train",
        dataset_root / "val",
        dataset_root / "valid",
        dataset_root / "validation",
        dataset_root / "test",
    ]
    grouped: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    missing_subject = 0
    for root in roots:
        if not root.exists():
            continue
        for path in iter_images(root):
            label = infer_label(path, mode)
            if label is None:
                continue
            subject_id = parse_subject_like_id(path)
            if subject_id is None:
                missing_subject += 1
                subject_id = f"unparsed_{path.stem.lower()}"
            grouped[subject_id].append((path, label))
    if missing_subject:
        print(f"warning: {missing_subject} files did not match known subject-like filename patterns")
    return dict(grouped)


def dominant_label(items: list[tuple[Path, str]]) -> str:
    counts = Counter(label for _path, label in items)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def assign_groups(
    grouped: dict[str, list[tuple[Path, str]]],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, dict[str, list[tuple[Path, str]]]]:
    if not abs(train_fraction + val_fraction + test_fraction - 1.0) < 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[str, list[tuple[Path, str]]]]] = defaultdict(list)
    for subject_id, items in grouped.items():
        by_label[dominant_label(items)].append((subject_id, items))

    split_groups: dict[str, dict[str, list[tuple[Path, str]]]] = {"train": {}, "val": {}, "test": {}}
    for label, groups in sorted(by_label.items()):
        groups = list(groups)
        rng.shuffle(groups)
        total_images = sum(len(items) for _subject_id, items in groups)
        target_test = int(round(total_images * test_fraction))
        target_val = int(round(total_images * val_fraction))

        current_test = 0
        current_val = 0
        for subject_id, items in groups:
            if current_test < target_test:
                split = "test"
                current_test += len(items)
            elif current_val < target_val:
                split = "val"
                current_val += len(items)
            else:
                split = "train"
            split_groups[split][subject_id] = items
    return split_groups


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


def materialize(split_groups: dict[str, dict[str, list[tuple[Path, str]]]], out_root: Path, copy_files: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split, groups in split_groups.items():
        label_counts: Counter[str] = Counter()
        subject_count = 0
        idx = 0
        for subject_id, items in sorted(groups.items()):
            subject_count += 1
            for src, label in sorted(items, key=lambda item: str(item[0])):
                label_counts[label] += 1
                safe_subject = re.sub(r"[^a-zA-Z0-9_.-]+", "_", subject_id)
                safe_name = f"{split}_{idx:06d}_{safe_subject}_{src.name}"
                link_or_copy(src, out_root / split / label / safe_name, copy_files)
                idx += 1
        summary[split] = {
            "subjects": subject_count,
            "images": idx,
            "labels": dict(sorted(label_counts.items())),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--label-mode", default="auto", choices=["auto", "binary", "three_class"])
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_base = Path(args.raw_root)
    dataset_root = find_dataset_root(raw_base)
    out_root = Path(args.out_root)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"out root already exists and is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    mode = choose_label_mode(dataset_root, args.label_mode)
    grouped = collect_grouped(dataset_root, mode)
    split_groups = assign_groups(
        grouped,
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
    )
    split_summary = materialize(split_groups, out_root, bool(args.copy_files))
    manifest = {
        "source_root": str(dataset_root),
        "out_root": str(out_root),
        "label_mode": mode,
        "split_strategy": "filename-derived subject-like ID grouping",
        "fractions": {
            "train": float(args.train_fraction),
            "val": float(args.val_fraction),
            "test": float(args.test_fraction),
        },
        "seed": int(args.seed),
        "total_subject_like_ids": len(grouped),
        "split_summary": split_summary,
        "audit": audit_dataset(out_root, mode),
        "limitation": "Subject-like IDs are parsed from filenames; authoritative patient metadata were not available.",
    }
    save_json(manifest, out_root / "split_manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()
