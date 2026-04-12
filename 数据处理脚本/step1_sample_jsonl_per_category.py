#!/usr/bin/env python3
import json
import random
from pathlib import Path

SRC_DIR = Path("/root/autodl-tmp/TTS/GenExam/data/annotations")
DST_DIR = Path("/root/autodl-tmp/TTS/Dataset/step1_每个种类10条")
EXCLUDE_FILE = "All_Subjects.jsonl"
SAMPLE_SIZE = 10
SEED = 42


def main() -> None:
    random.seed(SEED)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(
        path for path in SRC_DIR.glob("*.jsonl") if path.name != EXCLUDE_FILE
    )

    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files found in {SRC_DIR}")

    for src_file in jsonl_files:
        with src_file.open("r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]

        if len(lines) < SAMPLE_SIZE:
            raise ValueError(
                f"{src_file.name} has only {len(lines)} non-empty lines, less than {SAMPLE_SIZE}"
            )

        sampled_lines = random.sample(lines, SAMPLE_SIZE)
        dst_file = DST_DIR / src_file.name

        with dst_file.open("w", encoding="utf-8") as f:
            f.writelines(sampled_lines)

        print(f"{src_file.name}: wrote {len(sampled_lines)} lines -> {dst_file}")


if __name__ == "__main__":
    main()
