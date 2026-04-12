#!/usr/bin/env python3
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

SRC_DIR = Path("/root/autodl-tmp/TTS/GenExam/data/annotations")
IMAGE_ROOT = Path("/root/autodl-tmp/TTS/GenExam/data/images")
DST_DIR = Path("/root/autodl-tmp/TTS/Dataset/step3_重新找到高质量图片")
EXCLUDE_FILE = "All_Subjects.jsonl"
SAMPLE_SIZE = 2
TARGET_SIZES = [
    (1024, 1024),
    (1536, 1024),
    (1024, 1536),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def choose_best_target(width: int, height: int) -> tuple[tuple[int, int], float]:
    best_target: tuple[int, int] | None = None
    best_score: float | None = None

    for target_width, target_height in TARGET_SIZES:
        width_gap = abs(math.log(width / target_width))
        height_gap = abs(math.log(height / target_height))
        aspect_gap = abs(math.log((width / height) / (target_width / target_height)))
        score = width_gap + height_gap + 3.0 * aspect_gap

        min_side = min(width, height)
        max_side = max(width, height)
        if min_side < 700:
            score += 2.0
        elif min_side < 900:
            score += 0.6

        if max_side < 900:
            score += 1.2

        if best_score is None or score < best_score:
            best_score = score
            best_target = (target_width, target_height)

    assert best_target is not None and best_score is not None
    return best_target, best_score


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    relative_image_path = record["image_path"]
    image_path = IMAGE_ROOT / relative_image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size

    best_target, score = choose_best_target(width, height)
    return {
        "record": record,
        "absolute_image_path": str(image_path),
        "width": width,
        "height": height,
        "best_target_size": f"{best_target[0]}x{best_target[1]}",
        "score": score,
    }


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(
        path for path in SRC_DIR.glob("*.jsonl") if path.name != EXCLUDE_FILE
    )
    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files found in {SRC_DIR}")

    summary: dict[str, list[dict[str, Any]]] = {}

    for src_file in jsonl_files:
        rows = load_jsonl(src_file)
        if len(rows) < SAMPLE_SIZE:
            raise ValueError(
                f"{src_file.name} has only {len(rows)} non-empty lines, less than {SAMPLE_SIZE}"
            )

        scored_rows = [evaluate_record(row) for row in rows]
        scored_rows.sort(key=lambda item: (item["score"], -(item["width"] * item["height"])))
        selected = scored_rows[:SAMPLE_SIZE]

        dst_file = DST_DIR / src_file.name
        with dst_file.open("w", encoding="utf-8") as f:
            for item in selected:
                output_record = dict(item["record"])
                output_record["image_path"] = item["absolute_image_path"]
                output_record["original_width"] = item["width"]
                output_record["original_height"] = item["height"]
                output_record["recommended_tts_size"] = item["best_target_size"]
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

        summary[src_file.stem] = [
            {
                "id": item["record"].get("id"),
                "image_path": item["absolute_image_path"],
                "original_size": [item["width"], item["height"]],
                "recommended_tts_size": item["best_target_size"],
                "score": round(item["score"], 6),
            }
            for item in selected
        ]

        print(f"{src_file.name}: selected {len(selected)} samples -> {dst_file}")
        for item in selected:
            print(
                "  - "
                f"{item['record'].get('id')} | "
                f"{item['width']}x{item['height']} | "
                f"target {item['best_target_size']} | "
                f"score={item['score']:.4f}"
            )

    summary_path = DST_DIR / "selection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
