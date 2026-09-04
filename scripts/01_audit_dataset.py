from __future__ import annotations

import argparse
from pathlib import Path

from pedilite.data import audit_dataset
from pedilite.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--label-mode", default="auto", choices=["auto", "binary", "three_class"])
    parser.add_argument("--out", default="results/label_audit.json")
    args = parser.parse_args()

    audit = audit_dataset(args.data_root, args.label_mode)
    save_json(audit, args.out)
    print(f"wrote {Path(args.out).resolve()}")
    print(audit)


if __name__ == "__main__":
    main()
