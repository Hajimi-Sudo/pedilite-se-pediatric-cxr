from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


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


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/chest_xray_split")
    parser.add_argument("--out-dir", default="results/hardened/tables")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    split_files: dict[str, list[Path]] = {}
    subject_to_rows: dict[str, list[dict[str, Any]]] = {}
    hash_to_rows: dict[str, list[dict[str, Any]]] = {}
    unknown_subject_rows: list[dict[str, Any]] = []

    for split in SPLITS:
        split_root = data_root / split
        files = list(iter_images(split_root))
        split_files[split] = files
        for path in files:
            rel_path = str(path.relative_to(data_root))
            subject_id = parse_subject_like_id(path)
            row = {"split": split, "relative_path": rel_path, "subject_like_id": subject_id}
            if subject_id is None:
                unknown_subject_rows.append(row)
            else:
                subject_to_rows.setdefault(subject_id, []).append(row)
            file_hash = sha1_file(path)
            hash_to_rows.setdefault(file_hash, []).append({"split": split, "relative_path": rel_path, "sha1": file_hash})

    subject_overlaps = []
    for subject_id, rows in subject_to_rows.items():
        splits = sorted({row["split"] for row in rows})
        if len(splits) > 1:
            subject_overlaps.append({
                "subject_like_id": subject_id,
                "splits": ",".join(splits),
                "n_files": len(rows),
                "paths": ";".join(row["relative_path"] for row in rows[:20]),
            })

    duplicate_hash_overlaps = []
    for file_hash, rows in hash_to_rows.items():
        splits = sorted({row["split"] for row in rows})
        if len(splits) > 1:
            duplicate_hash_overlaps.append({
                "sha1": file_hash,
                "splits": ",".join(splits),
                "n_files": len(rows),
                "paths": ";".join(row["relative_path"] for row in rows[:20]),
            })

    summary = {
        "data_root": str(data_root),
        "split_counts": {split: len(files) for split, files in split_files.items()},
        "subject_like_id_count": len(subject_to_rows),
        "unknown_subject_like_id_files": len(unknown_subject_rows),
        "subject_like_id_overlap_count": len(subject_overlaps),
        "exact_duplicate_hash_overlap_count": len(duplicate_hash_overlaps),
        "interpretation": (
            "No filename-derived subject-like ID overlap or exact duplicate hash overlap was found"
            if not subject_overlaps and not duplicate_hash_overlaps
            else "Potential split overlap was found; inspect CSV details before using the split"
        ),
        "limitation": (
            "Filename-derived IDs and exact hashes can reduce leakage concern, but they do not prove"
            " true patient-level independence without authoritative patient metadata."
        ),
    }

    write_json(summary, out_dir / "split_integrity_audit.json")
    write_csv(out_dir / "split_subject_overlaps.csv", subject_overlaps, ["subject_like_id", "splits", "n_files", "paths"])
    write_csv(out_dir / "split_duplicate_hash_overlaps.csv", duplicate_hash_overlaps, ["sha1", "splits", "n_files", "paths"])
    write_csv(out_dir / "split_unknown_subject_ids.csv", unknown_subject_rows, ["split", "relative_path", "subject_like_id"])

    report = [
        "# Split Integrity Audit",
        "",
        f"- Data root: `{data_root}`",
        f"- Split counts: `{summary['split_counts']}`",
        f"- Filename-derived subject-like IDs: `{summary['subject_like_id_count']}`",
        f"- Files without parseable subject-like ID: `{summary['unknown_subject_like_id_files']}`",
        f"- Subject-like IDs crossing splits: `{summary['subject_like_id_overlap_count']}`",
        f"- Exact duplicate SHA1 hashes crossing splits: `{summary['exact_duplicate_hash_overlap_count']}`",
        "",
        f"Interpretation: {summary['interpretation']}.",
        "",
        f"Limitation: {summary['limitation']}",
        "",
    ]
    (out_dir / "split_integrity_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
