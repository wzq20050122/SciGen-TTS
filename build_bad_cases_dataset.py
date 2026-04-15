#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl_map(path: Path, key: str = "id") -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            k = obj.get(key)
            if k:
                data[str(k)] = obj
        except Exception:
            continue
    return data


def safe_copy(src: Path, dst: Path) -> Optional[str]:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.resolve())


def cal_relaxed_score_single(eval_result: Dict[str, Any]) -> Optional[float]:
    """Same formula as GenExam/cal_score.py, return None if fields are incomplete."""
    if not isinstance(eval_result, dict):
        return None
    answers = eval_result.get("answers")
    scoring_points = eval_result.get("scoring_points")
    ge = eval_result.get("global_evaluation")
    if not isinstance(answers, list) or not isinstance(scoring_points, list) or not isinstance(ge, dict):
        return None
    if len(answers) != len(scoring_points) or len(answers) == 0:
        return None

    try:
        semantic_correctness = 0.0
        for idx, ans in enumerate(answers):
            a = ans.get("answer")
            is_correct = (a is True) or (a == 1)
            if is_correct:
                semantic_correctness += float(scoring_points[idx].get("score", 0))

        readability = float(ge["Clarity and Readability"]["score"])
        logical_consistency = float(ge["Logical Consistency"]["score"])
        spelling = float(ge["Spelling"]["score"])

        relaxed_score = round(
            semantic_correctness * 0.7
            + spelling * 0.1 / 2
            + readability * 0.1 / 2
            + logical_consistency * 0.1 / 2,
            3,
        )
        return relaxed_score
    except Exception:
        return None


def compute_relaxed_score(step_dir: Path, verifier_parsed: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    # 1) If already present in verifier output, use directly
    if "relaxed_score" in verifier_parsed and verifier_parsed.get("relaxed_score") is not None:
        try:
            return float(verifier_parsed.get("relaxed_score")), "verifier_parsed"
        except Exception:
            pass

    # 2) Try exact score formula from any eval-result-like json in step dir
    candidate_files = [
        step_dir / "eval_result.json",
        step_dir / "score_result.json",
        step_dir / "meta_score.json",
    ]
    for p in candidate_files:
        obj = read_json(p)
        rs = cal_relaxed_score_single(obj or {})
        if rs is not None:
            return rs, str(p.resolve())

    for p in sorted(step_dir.glob("*.json")):
        obj = read_json(p)
        rs = cal_relaxed_score_single(obj or {})
        if rs is not None:
            return rs, str(p.resolve())

    # 3) Fallback: only verifier answer is available
    ans = verifier_parsed.get("answer")
    if ans is True:
        return 0.7, "fallback_from_verifier_answer"
    if ans is False:
        return 0.0, "fallback_from_verifier_answer"

    return None, None


def collect_step_info(case_dir: Path, step_num: int) -> Dict[str, Any]:
    step_dir = case_dir / f"step{step_num}"
    editor_payload = read_json(step_dir / "editor_payload.json") or {}
    verifier_parsed = read_json(step_dir / "verifier_parsed.json") or {}
    judge_meta = read_json(step_dir / f"meta_judge_step{step_num}.json") or {}

    relaxed_score, relaxed_score_source = compute_relaxed_score(step_dir, verifier_parsed)

    step_info = {
        "step": step_num,
        "step_dir": str(step_dir.resolve()),
        "mode": editor_payload.get("mode"),
        "instruction": editor_payload.get("instruction"),
        "current_instruction": editor_payload.get("current_instruction"),
        "verifier": {
            "answer": verifier_parsed.get("answer"),
            "relaxed_score": relaxed_score,
            "relaxed_score_source": relaxed_score_source,
            "explanation": verifier_parsed.get("explanation"),
            "edit": verifier_parsed.get("edit"),
        },
        "judge_for_verifier": {
            "answer_correct": judge_meta.get("answer_correct"),
            "explanation_quality": judge_meta.get("explanation_quality"),
            "edit_quality": judge_meta.get("edit_quality"),
            "edit_excluded": judge_meta.get("edit_excluded"),
            "judge_model": judge_meta.get("judge_model"),
            "judge_time": judge_meta.get("judge_time"),
        },
    }
    return step_info


def build_messages(prompt: str, steps_info: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": "我先给出第1版结果。"})

    for idx, s in enumerate(steps_info, start=1):
        v = s.get("verifier", {})
        answer = v.get("answer")
        explanation = v.get("explanation") or ""
        edit = v.get("edit") or ""

        if idx == 1:
            messages.append({"role": "assistant", "content": f"已生成 step{idx} 图像。"})

        if answer is False and edit:
            messages.append({"role": "user", "content": f"<image>\n请按以下反馈修改：\n{edit}"})
            messages.append({"role": "assistant", "content": f"好的，已完成 step{idx+1} 修订。"})
        elif answer is False and explanation:
            messages.append({"role": "user", "content": f"<image>\n请根据评估继续优化：\n{explanation}"})
            messages.append({"role": "assistant", "content": f"好的，已完成 step{idx+1} 修订。"})

    return messages


def load_relaxed_scores_from_dir(eval_dir: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not eval_dir.exists() or not eval_dir.is_dir():
        return out
    for p in sorted(eval_dir.glob("*.json")):
        obj = read_json(p)
        if not obj:
            continue
        cid = obj.get("id")
        if not cid:
            continue
        rs = cal_relaxed_score_single(obj)
        if rs is not None:
            out[str(cid)] = rs
    return out


def load_step_relaxed_maps(judge_output_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load relaxed scores for each evaluated step folder under judge_output_dir.

    Returns mapping like:
      {
        "step1": {id: score},
        "step2": {id: score},
        "final": {id: score},
      }
    """
    step_maps: Dict[str, Dict[str, float]] = {}
    if not judge_output_dir.exists() or not judge_output_dir.is_dir():
        return step_maps

    for p in sorted(judge_output_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("eval_results_"):
            continue
        step_key = name.removeprefix("eval_results_")
        if not step_key:
            continue
        step_maps[step_key] = load_relaxed_scores_from_dir(p)

    return step_maps


def process_case(
    case_dir: Path,
    out_root: Path,
    global_judge_map: Dict[str, Dict[str, Any]],
    step_relaxed_maps: Dict[str, Dict[str, float]],
) -> Optional[Path]:
    final_result = read_json(case_dir / "final_result.json")
    if not final_result:
        return None

    case_name = case_dir.name
    case_id = final_result.get("id", case_name)

    out_case_dir = out_root / case_name
    images_dir = out_case_dir / "images"
    out_case_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    history = final_result.get("history", [])
    step_nums = []
    for h in history:
        step = h.get("step")
        if isinstance(step, int):
            step_nums.append(step)
    step_nums = sorted(set(step_nums))

    steps_info: List[Dict[str, Any]] = []
    copied_images: List[str] = []

    for step_num in step_nums:
        step_info = collect_step_info(case_dir, step_num)

        # Prefer per-step Judge_output relaxed scores when available.
        case_id_str = str(case_id)
        step_key = f"step{step_num}"
        if step_key in step_relaxed_maps and case_id_str in step_relaxed_maps[step_key]:
            step_info.setdefault("verifier", {})["relaxed_score"] = step_relaxed_maps[step_key][case_id_str]
            step_info.setdefault("verifier", {})["relaxed_score_source"] = f"judge_output_eval_results_{step_key}"
        elif step_num == int(final_result.get("steps_used", 0) or 0):
            final_map = step_relaxed_maps.get("final", {})
            if case_id_str in final_map:
                step_info.setdefault("verifier", {})["relaxed_score"] = final_map[case_id_str]
                step_info.setdefault("verifier", {})["relaxed_score_source"] = "judge_output_eval_results_final"

        # copy step image
        src_img = case_dir / f"step{step_num}" / "image.png"
        dst_img = images_dir / f"step{step_num}.png"
        copied = safe_copy(src_img, dst_img)
        if copied:
            copied_images.append(copied)
            step_info["image"] = copied
        else:
            step_info["image"] = None

        steps_info.append(step_info)

    # copy final image if present
    final_img_src = case_dir / "final.png"
    final_img_dst = images_dir / "final.png"
    final_img_copied = safe_copy(final_img_src, final_img_dst)
    if final_img_copied:
        copied_images.append(final_img_copied)

    prompt = final_result.get("prompt") or ""
    messages = build_messages(prompt, steps_info)

    # global judge for verifier from verifier_quality.jsonl
    global_judge = global_judge_map.get(str(case_id), {})

    step1_relaxed_score = None
    final_relaxed_score = None
    final_relaxed_score_source = None
    per_step_relaxed_scores: Dict[str, Optional[float]] = {}

    if steps_info:
        for s in steps_info:
            step_num = int(s.get("step", 0) or 0)
            per_step_relaxed_scores[f"step{step_num}"] = s.get("verifier", {}).get("relaxed_score")
            if step_num == 1:
                step1_relaxed_score = s.get("verifier", {}).get("relaxed_score")
        final_step_info = max(steps_info, key=lambda x: int(x.get("step", 0) or 0))
        final_relaxed_score = final_step_info.get("verifier", {}).get("relaxed_score")
        final_relaxed_score_source = final_step_info.get("verifier", {}).get("relaxed_score_source")

    record = {
        "id": str(case_id),
        "messages": messages,
        "images": copied_images,
        "audios": [],
        "videos": [],
        "metadata": {
            "case_name": case_name,
            "source_case_dir": str(case_dir.resolve()),
            "source_jsonl": final_result.get("source_jsonl"),
            "original_prompt": prompt,
            "original_image_path": final_result.get("original_image_path"),
            "success": final_result.get("success"),
            "steps_used": final_result.get("steps_used"),
            "final_image": final_img_copied,
            "step1_relaxed_score": step1_relaxed_score,
            "final_relaxed_score": final_relaxed_score,
            "final_relaxed_score_source": final_relaxed_score_source,
            "per_step_relaxed_scores": per_step_relaxed_scores,
            "step_details": steps_info,
            "judge_for_verifier_global": {
                "answer_correct": global_judge.get("answer_correct"),
                "explanation_quality": global_judge.get("explanation_quality"),
                "edit_quality": global_judge.get("edit_quality"),
                "edit_excluded": global_judge.get("edit_excluded"),
                "judge_model": global_judge.get("judge_model"),
                "judge_time": global_judge.get("judge_time"),
            },
        },
    }

    out_json = out_case_dir / "sample.json"
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect each TTS case into unified bad_cases format.")
    parser.add_argument("--input", default="/root/autodl-tmp/TTS/output_TTS/wan2.6-t2i&wan2.6-image_20260411", help="Input run directory")
    parser.add_argument("--output", default="/root/autodl-tmp/TTS/bad_cases", help="Output directory")
    parser.add_argument("--judge_output", default="/root/autodl-tmp/TTS/Judge_output/wan2.6-t2i&wan2.6-image_20260411", help="Judge_output directory for step1/final eval scores")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    judge_output_dir = Path(args.judge_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_judge_map = read_jsonl_map(input_dir / "verifier_quality.jsonl", key="id")

    # load per-step relaxed scores from all eval_results_* folders
    step_relaxed_maps = load_step_relaxed_maps(judge_output_dir)
    # shared/same_source belong to final-equivalent cases; merge into final view.
    final_map = step_relaxed_maps.setdefault("final", {})
    final_map.update(load_relaxed_scores_from_dir(judge_output_dir / "eval_results_same_source"))
    final_map.update(load_relaxed_scores_from_dir(judge_output_dir / "eval_results_shared"))

    case_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir() and (p / "final_result.json").exists()])

    generated = []
    for case_dir in case_dirs:
        out_json = process_case(
            case_dir,
            output_dir,
            global_judge_map,
            step_relaxed_maps,
        )
        if out_json:
            generated.append(str(out_json))

    print(f"Done. Processed {len(generated)} cases.")
    print(f"Output root: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
