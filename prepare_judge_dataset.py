#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def safe_subject(record: Dict[str, Any]) -> str:
    if record.get("subject"):
        return str(record["subject"])
    rid = str(record.get("id", "unknown"))
    if "_" in rid:
        return rid.split("_", 1)[0]
    return "Unknown"


def first_existing_path(*candidates: Optional[str]) -> Optional[Path]:
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None


def list_step_images(sample_dir: Path) -> Dict[int, Path]:
    step_images: Dict[int, Path] = {}
    for step_dir in sample_dir.glob("step*"):
        if not step_dir.is_dir():
            continue
        name = step_dir.name
        if not name.startswith("step"):
            continue
        try:
            step_num = int(name[4:])
        except Exception:
            continue

        img = first_existing_path(
            step_dir.joinpath("image.png").as_posix(),
            step_dir.joinpath("image.jpg").as_posix(),
            step_dir.joinpath("image.jpeg").as_posix(),
            step_dir.joinpath("image.webp").as_posix(),
        )
        if img:
            step_images[step_num] = img

    return step_images


def _safe_name(text: str) -> str:
    import re

    text = re.sub(r"[^0-9a-zA-Z_\-\.\u4e00-\u9fff&]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "run"


def _build_run_tag(gen_model: str, edit_model: str, run_date: str) -> str:
    return _safe_name(f"{gen_model}&{edit_model}_{run_date}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TTS output into GenExam-like Judge_Dataset format.")
    parser.add_argument("--input-dir", type=Path, required=True, help="TTS output dir, e.g. /root/autodl-tmp/TTS/output_TTS")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output Judge_Dataset root dir")
    parser.add_argument("--clean", action="store_true", help="Delete output dir before generating")
    parser.add_argument("--gen-model", default="unknown_gen", help="Generation model name for run folder")
    parser.add_argument("--edit-model", default="unknown_edit", help="Edit model name for run folder")
    parser.add_argument("--run-date", default=None, help="Run date tag in folder name, format YYYYMMDD. Default=today")
    parser.add_argument("--disable-auto-run-subdir", action="store_true", help="Disable auto creating model&date subfolder under --output-dir")
    args = parser.parse_args()

    input_dir = args.input_dir
    run_date = args.run_date or datetime.now().strftime("%Y%m%d")
    run_tag = _build_run_tag(args.gen_model, args.edit_model, run_date)
    output_dir = args.output_dir if args.disable_auto_run_subdir else (args.output_dir / run_tag)

    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    ann_dir = output_dir / "annotations"
    gt_img_dir = output_dir / "images"
    candidates_dir = output_dir / "candidates"
    step1_dir = candidates_dir / "step1"
    final_dir = candidates_dir / "final"
    meta_dir = output_dir / "meta"

    for d in [ann_dir, gt_img_dir, candidates_dir, step1_dir, final_dir, meta_dir]:
        ensure_dir(d)

    all_rows: List[Dict[str, Any]] = []
    changed_ids: List[str] = []
    missing_step1: List[str] = []
    missing_final: List[str] = []
    missing_gt: List[str] = []

    sample_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])

    for sample_dir in sample_dirs:
        final_json = sample_dir / "final_result.json"
        source_json = sample_dir / "source_record.json"
        if not final_json.exists() or not source_json.exists():
            continue

        final_result = read_json(final_json)
        source_record = read_json(source_json)

        rid = str(source_record.get("id") or final_result.get("id") or sample_dir.name)
        subject = safe_subject(source_record)

        gt_src = first_existing_path(
            source_record.get("image_path"),
            final_result.get("original_image_path"),
        )
        step_images = list_step_images(sample_dir)
        step1_src = step_images.get(1)
        final_src = first_existing_path(
            final_result.get("final_image_path"),
            sample_dir.joinpath("final.png").as_posix(),
            sample_dir.joinpath("final.jpg").as_posix(),
            sample_dir.joinpath("final.jpeg").as_posix(),
            sample_dir.joinpath("final.webp").as_posix(),
        )

        gt_rel = f"{subject}/{rid}.png"
        step1_rel = f"{subject}/{rid}.png"
        final_rel = f"{subject}/{rid}.png"

        gt_dst = gt_img_dir / gt_rel
        step1_dst = step1_dir / step1_rel
        final_dst = final_dir / final_rel

        gt_ok = copy_if_exists(gt_src, gt_dst) if gt_src else False
        step1_ok = copy_if_exists(step1_src, step1_dst) if step1_src else False
        final_ok = copy_if_exists(final_src, final_dst) if final_src else False

        if not gt_ok:
            missing_gt.append(rid)
        if not step1_ok:
            missing_step1.append(rid)
        if not final_ok:
            missing_final.append(rid)

        row = dict(source_record)
        row["id"] = rid
        row["subject"] = subject

        # Write absolute paths to avoid path-join issues in downstream evaluation scripts.
        row["image_path"] = str(gt_dst.resolve()) if gt_ok else None
        row["step1_image_path"] = str(step1_dst.resolve()) if step1_ok else None
        row["final_image_path"] = str(final_dst.resolve()) if final_ok else None

        # Also expose every available intermediate step image for per-step evaluation.
        step_image_paths: Dict[str, str] = {}
        for step_num, src in sorted(step_images.items()):
            dst = candidates_dir / f"step{step_num}" / f"{subject}/{rid}.png"
            ok = copy_if_exists(src, dst)
            if ok:
                step_image_paths[f"step{step_num}"] = str(dst.resolve())
        row["step_image_paths"] = step_image_paths

        row["steps_used"] = final_result.get("steps_used")
        row["tts_success"] = final_result.get("success")
        all_rows.append(row)

        if step1_ok and final_ok:
            try:
                same = filecmp.cmp(step1_dst, final_dst, shallow=False)
            except Exception:
                same = False
            if not same:
                changed_ids.append(rid)

    all_rows.sort(key=lambda x: x.get("id", ""))

    with (ann_dir / "All_Subjects.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_subject.setdefault(row.get("subject", "Unknown"), []).append(row)
    for subject, rows in by_subject.items():
        with (ann_dir / f"{subject}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    changed_ids = sorted(changed_ids)
    all_ids = sorted([row.get("id") for row in all_rows if row.get("id")])
    changed_set = set(changed_ids)
    same_ids = [rid for rid in all_ids if rid not in changed_set]

    (meta_dir / "ids_step1_final_different.txt").write_text(
        "\n".join(changed_ids) + ("\n" if changed_ids else ""),
        encoding="utf-8",
    )
    (meta_dir / "ids_step1_final_same.txt").write_text(
        "\n".join(same_ids) + ("\n" if same_ids else ""),
        encoding="utf-8",
    )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_samples": len(all_rows),
        "step1_final_different_count": len(changed_ids),
        "missing_gt_count": len(missing_gt),
        "missing_step1_count": len(missing_step1),
        "missing_final_count": len(missing_final),
    }
    with (meta_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
