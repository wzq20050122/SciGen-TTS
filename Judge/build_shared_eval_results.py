#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def load_ids(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build shared eval results for ids where step1 == final by copying once-evaluated results to a common folder."
    )
    parser.add_argument("--same_ids", type=Path, required=True, help="ids_step1_final_same.txt")
    parser.add_argument(
        "--source_eval_dir",
        type=Path,
        required=True,
        help="Source eval dir (usually eval_results_step1 or eval_results_final)",
    )
    parser.add_argument("--shared_eval_dir", type=Path, required=True, help="Output shared eval dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ids = load_ids(args.same_ids)
    args.shared_eval_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    missing = []

    for rid in ids:
        src = args.source_eval_dir / f"{rid}.json"
        dst = args.shared_eval_dir / f"{rid}.json"

        if not src.exists():
            missing.append(rid)
            continue

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    print(f"same_ids={len(ids)} copied={copied} skipped={skipped} missing={len(missing)}")
    if missing:
        print("Missing IDs:")
        for rid in missing:
            print(rid)


if __name__ == "__main__":
    main()
