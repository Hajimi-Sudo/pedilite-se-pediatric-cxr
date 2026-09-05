from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pedilite.data import IMAGE_EXTENSIONS


def dhash(path: Path) -> int:
    image = Image.open(path).convert("L").resize((9, 8))
    values = np.asarray(image, dtype=np.int16)
    bits = (values[:, 1:] > values[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cross-split perceptual near-duplicates.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-hamming", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, str, int]] = []
    for split in ("train", "val", "valid", "validation", "test"):
        split_root = root / split
        if not split_root.exists():
            continue
        for path in sorted(split_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((split, str(path), dhash(path)))

    # Compare only cross-split pairs. The block index keeps the audit tractable
    # while still finding low-Hamming dHash collisions.
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (_split, _path, value) in enumerate(records):
        for shift in (0, 16, 32, 48):
            buckets.setdefault((shift, (value >> shift) & 0xFFFF), []).append(idx)

    pairs: set[tuple[int, int]] = set()
    for candidates in buckets.values():
        for left_pos, left in enumerate(candidates):
            for right in candidates[left_pos + 1 :]:
                if records[left][0] == records[right][0]:
                    continue
                distance = (records[left][2] ^ records[right][2]).bit_count()
                if distance <= args.max_hamming:
                    pairs.add((min(left, right), max(left, right)))

    pair_path = out_dir / "near_duplicate_pairs.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split_a", "path_a", "split_b", "path_b", "dhash_hamming"])
        for left, right in sorted(pairs):
            writer.writerow([records[left][0], records[left][1], records[right][0], records[right][1],
                             (records[left][2] ^ records[right][2]).bit_count()])

    summary = {
        "data_root": str(root),
        "images": len(records),
        "cross_split_near_duplicate_pairs": len(pairs),
        "max_hamming": args.max_hamming,
        "method": "8x8 dHash; candidate blocking on four 16-bit chunks",
    }
    (out_dir / "near_duplicate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
