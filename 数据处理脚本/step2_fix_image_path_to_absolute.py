#!/usr/bin/env python3
import json
from pathlib import Path

SRC_DIR = Path("/root/autodl-tmp/TTS/Dataset/step1_每个种类10条")
DST_DIR = Path("/root/autodl-tmp/TTS/Dataset/step2_修正图片路径")
IMAGE_PREFIX = "/root/autodl-tmp/TTS/GenExam/data/images/"


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(SRC_DIR.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files found in {SRC_DIR}")

    for src_file in jsonl_files:
        dst_file = DST_DIR / src_file.name

        with src_file.open("r", encoding="utf-8") as fin, dst_file.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if not line.strip():
                    continue

                item = json.loads(line)
                image_path = item.get("image_path", "")
                item["image_path"] = f"{IMAGE_PREFIX}{image_path}"
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"{src_file.name} -> {dst_file}")


if __name__ == "__main__":
    main()
