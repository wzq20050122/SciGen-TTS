import argparse
import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Tuple


def _write_json_atomic(path: str, data: Any) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

import requests
from PIL import Image
from tqdm import tqdm


DEFAULT_API_KEY = os.environ.get("DMX_API_KEY", "")
DEFAULT_BASE_URL = os.environ.get("DMX_BASE_URL", "https://www.dmxapi.cn/v1")
DEFAULT_MODEL_NAME = "gemini-3-flash-preview"


def _dmx_headers(api_key: str, json_request: bool = False) -> Dict[str, str]:
    headers = {"Authorization": api_key}
    if json_request:
        headers["Content-Type"] = "application/json"
    return headers


def encode_image(image_path: str, target_size: int = 1024, fmt: str = "JPEG") -> str:
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if target_size is not None and target_size > 0:
        w, h = img.size
        if max(w, h) > target_size:
            if w >= h:
                new_w = target_size
                new_h = int(h * target_size / w)
            else:
                new_h = target_size
                new_w = int(w * target_size / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)

    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    fenced = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    candidates = fenced + re.findall(r"(\{[\s\S]*\})", text)

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def _build_judge_prompt(source: Dict[str, Any], verifier_parsed: Dict[str, Any], has_reference_image: bool) -> str:
    prompt_text = source.get("prompt", "")

    reference_hint = (
        "4) A ground-truth reference image (second image), which can be used as reference for correctness.\n"
        if has_reference_image
        else ""
    )

    return f"""
You are evaluating the quality of a verifier's step1 judgment for an image generation task.

You will see:
1) The original user prompt.
2) The step1 generated image (first image).
3) The verifier's structured output (answer/explanation/edit).
{reference_hint}
Your task: judge whether verifier's answer/explanation/edit are accurate and useful with respect to the prompt and the provided step1 image.

Scoring rules:
- answer_correct: boolean
  - true if verifier's answer is correct.
  - false if verifier's answer is wrong.
- explanation_quality: float in [0,1]
  - Measures factual correctness + completeness + clarity of explanation.
  - 1.0 = fully correct, complete, and clear.
  - 0.0 = fully incorrect or unusable.
- edit_quality: float in [0,1]
  - Measures whether proposed edit is actionable, correct, minimal, and aligned with prompt deficiencies.
  - 1.0 = precise, actionable, high-value edit.
  - 0.0 = wrong or useless edit.

Important:
- Be strict but fair.
- Use one decimal or two decimals if needed.
- Return JSON only. No markdown fences.

Return exactly this schema:
{{
  "answer_correct": true,
  "explanation_quality": 0.0,
  "edit_quality": 0.0
}}

Context:
- original_prompt: {prompt_text}
- step1_image: <image>
{('- reference_image: <image>\n' if has_reference_image else '')}
Verifier parsed output:
{json.dumps(verifier_parsed, ensure_ascii=False, indent=2)}

""".strip()


def call_gemini_judge(
    image_path: str,
    source_record: Dict[str, Any],
    verifier_parsed: Dict[str, Any],
    api_key: str,
    base_url: str,
    model_name: str,
    gt_image_path: Optional[str] = None,
    max_tokens: int = 4096,
    img_size: int = 1024,
) -> Dict[str, Any]:
    img_b64 = encode_image(image_path, target_size=img_size)
    gt_img_b64 = encode_image(gt_image_path, target_size=img_size) if gt_image_path else None
    prompt = _build_judge_prompt(source_record, verifier_parsed, has_reference_image=gt_img_b64 is not None)

    image_urls = [f"data:image/jpeg;base64,{img_b64}"]
    if gt_img_b64 is not None:
        image_urls.append(f"data:image/jpeg;base64,{gt_img_b64}")

    parts = prompt.split("<image>")
    content: List[Dict[str, Any]] = []
    for i, text_part in enumerate(parts):
        if text_part:
            content.append({"type": "text", "text": text_part})
        if i < len(parts) - 1:
            if i >= len(image_urls):
                raise ValueError("Found <image> placeholder without corresponding image data")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_urls[i],
                        "detail": "high",
                    },
                }
            )

    if len(image_urls) != len(parts) - 1:
        raise ValueError(
            f"Image placeholder count mismatch: placeholders={len(parts) - 1}, images={len(image_urls)}"
        )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "stream": False,
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers=_dmx_headers(api_key, json_request=True),
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    parsed = _extract_json_block(content)
    if parsed is None:
        raise ValueError(f"Judge response is not valid JSON: {content[:500]}")

    return parsed


def _clamp_score(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def _normalize_judge(raw_judge: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer_correct": bool(raw_judge.get("answer_correct", False)),
        "explanation_quality": _clamp_score(raw_judge.get("explanation_quality", 0.0)),
        "edit_quality": _clamp_score(raw_judge.get("edit_quality", 0.0)),
    }


def _is_verifier_answer_true(answer: Any) -> bool:
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, str):
        return answer.strip().lower() in {"true", "yes", "1"}
    if isinstance(answer, (int, float)):
        return bool(answer)
    return False


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_sample_steps(run_dir: str) -> List[Tuple[str, int]]:
    pattern = os.path.join(run_dir, "*", "step*", "verifier_parsed.json")
    files = sorted(glob(pattern))
    results: List[Tuple[str, int]] = []
    for p in files:
        step_dir = os.path.dirname(p)
        sample_dir = os.path.dirname(step_dir)
        step_name = os.path.basename(step_dir)
        if not step_name.startswith("step"):
            continue
        try:
            step_num = int(step_name[4:])
        except Exception:
            continue
        results.append((sample_dir, step_num))
    return results


def _process_one_sample(
    sample_dir: str,
    step_num: int,
    api_key: str,
    base_url: str,
    model_name: str,
    force: bool,
    max_retries: int,
    sleep_seconds: float,
) -> Optional[Dict[str, Any]]:
    import time

    step_dir = os.path.join(sample_dir, f"step{step_num}")
    parsed_path = os.path.join(step_dir, "verifier_parsed.json")
    source_path = os.path.join(sample_dir, "source_record.json")
    image_path = os.path.join(step_dir, "image.png")
    out_path = os.path.join(step_dir, f"meta_judge_step{step_num}.json")

    if not os.path.exists(parsed_path) or not os.path.exists(source_path) or not os.path.exists(image_path):
        return None

    if os.path.exists(out_path) and not force:
        return _read_json(out_path)

    verifier_parsed = _read_json(parsed_path)
    source_record = _read_json(source_path)
    verifier_answer_true = _is_verifier_answer_true(verifier_parsed.get("answer", False))

    gt_image_path = source_record.get("image_path")
    if isinstance(gt_image_path, str) and gt_image_path:
        if not os.path.isabs(gt_image_path):
            gt_image_path = os.path.join(sample_dir, gt_image_path)
        if not os.path.exists(gt_image_path):
            gt_image_path = None
    else:
        gt_image_path = None

    checkpoint_path = os.path.join(step_dir, f"meta_judge_step{step_num}.checkpoint.json")
    if os.path.exists(checkpoint_path) and not force:
        try:
            checkpoint = _read_json(checkpoint_path)
            if checkpoint.get("status") == "done" and os.path.exists(out_path):
                return _read_json(out_path)
        except Exception:
            pass

    result: Dict[str, Any]
    last_err = None
    judge_raw: Optional[Dict[str, Any]] = None
    _write_json_atomic(checkpoint_path, {"status": "running", "id": source_record.get("id"), "step": step_num, "updated_at": datetime.now().isoformat()})
    for _ in range(max_retries):
        try:
            judge_raw = call_gemini_judge(
                image_path=image_path,
                source_record=source_record,
                verifier_parsed=verifier_parsed,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                gt_image_path=gt_image_path,
            )
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(sleep_seconds)

    if judge_raw is None:
        result = {
            "id": source_record.get("id", os.path.basename(sample_dir)),
            "subject": source_record.get("subject", ""),
            "sample_dir": sample_dir,
            "step": step_num,
            "judge_model": model_name,
            "judge_time": datetime.now().isoformat(),
            "error": last_err or "unknown_error",
            "answer_correct": False,
            "explanation_quality": 0.0,
            "edit_quality": 0.0,
        }
    else:
        norm = _normalize_judge(judge_raw)
        if verifier_answer_true:
            norm["edit_quality"] = 0.0
        result = {
            "id": source_record.get("id", os.path.basename(sample_dir)),
            "subject": source_record.get("subject", ""),
            "sample_dir": sample_dir,
            "step": step_num,
            "judge_model": model_name,
            "judge_time": datetime.now().isoformat(),
            **norm,
            "edit_excluded": verifier_answer_true,
        }

    _write_json_atomic(out_path, result)
    _write_json_atomic(checkpoint_path, {"status": "done", "id": result.get("id"), "step": step_num, "updated_at": datetime.now().isoformat()})

    return result


def evaluate_run(
    run_dir: str,
    api_key: str,
    base_url: str,
    model_name: str,
    force: bool = False,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
    write_jsonl: bool = True,
    max_workers: int = 8,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sample_steps = _discover_sample_steps(run_dir)
    if not sample_steps:
        raise FileNotFoundError(f"No step verifier files found under: {run_dir}")

    records: List[Dict[str, Any]] = []
    records_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    progress_path = os.path.join(run_dir, "verifier_quality_progress.json")
    existing_done: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(progress_path) and not force:
        try:
            existing_done = _read_json(progress_path)
        except Exception:
            existing_done = {}

    workers = max(1, int(max_workers))
    pending_samples = []
    for sample_dir, step_num in sample_steps:
        key = f"{sample_dir}::step{step_num}"
        if not force and key in existing_done and os.path.exists(os.path.join(sample_dir, f"step{step_num}", f"meta_judge_step{step_num}.json")):
            rec = existing_done[key]
            records.append(rec)
            records_by_key[(sample_dir, step_num)] = rec
        else:
            pending_samples.append((sample_dir, step_num))

    def _store_result(sample_dir: str, step_num: int, result: Dict[str, Any]) -> None:
        records.append(result)
        records_by_key[(sample_dir, step_num)] = result
        existing_done[f"{sample_dir}::step{step_num}"] = result
        _write_json_atomic(progress_path, existing_done)

    if workers == 1:
        for sample_dir, step_num in tqdm(pending_samples, desc="Judging verifier quality"):
            result = _process_one_sample(
                sample_dir=sample_dir,
                step_num=step_num,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                force=force,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
            if result is not None:
                _store_result(sample_dir, step_num, result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _process_one_sample,
                    sample_dir,
                    step_num,
                    api_key,
                    base_url,
                    model_name,
                    force,
                    max_retries,
                    sleep_seconds,
                ): (sample_dir, step_num)
                for sample_dir, step_num in pending_samples
            }

            for fut in tqdm(as_completed(future_map), total=len(future_map), desc=f"Judging verifier quality ({workers} workers)"):
                sample_dir, step_num = future_map[fut]
                try:
                    result = fut.result()
                    if result is not None:
                        _store_result(sample_dir, step_num, result)
                except Exception as e:
                    err = {
                        "id": "unknown",
                        "subject": "",
                        "sample_dir": "",
                        "judge_model": model_name,
                        "judge_time": datetime.now().isoformat(),
                        "error": str(e),
                        "answer_correct": False,
                        "explanation_quality": 0.0,
                        "edit_quality": 0.0,
                    }
                    _store_result(sample_dir, step_num, err)

    total = len(records)
    valid = [r for r in records if not r.get("error")]
    edit_valid = [r for r in valid if not r.get("edit_excluded", False)]
    answer_correct_rate = (sum(1 for r in valid if r.get("answer_correct")) / len(valid)) if valid else 0.0
    explanation_avg = (sum(float(r.get("explanation_quality", 0.0)) for r in valid) / len(valid)) if valid else 0.0
    edit_avg = (sum(float(r.get("edit_quality", 0.0)) for r in edit_valid) / len(edit_valid)) if edit_valid else 0.0

    by_subject: Dict[str, Dict[str, Any]] = {}
    for r in valid:
        sub = r.get("subject", "") or "Unknown"
        if sub not in by_subject:
            by_subject[sub] = {
                "count": 0,
                "answer_correct_count": 0,
                "explanation_quality_sum": 0.0,
                "edit_quality_sum": 0.0,
                "edit_count": 0,
            }
        by_subject[sub]["count"] += 1
        by_subject[sub]["answer_correct_count"] += 1 if r.get("answer_correct") else 0
        by_subject[sub]["explanation_quality_sum"] += float(r.get("explanation_quality", 0.0))
        if not r.get("edit_excluded", False):
            by_subject[sub]["edit_quality_sum"] += float(r.get("edit_quality", 0.0))
            by_subject[sub]["edit_count"] += 1

    for sub in list(by_subject.keys()):
        c = by_subject[sub]["count"]
        ec = by_subject[sub]["edit_count"]
        by_subject[sub] = {
            "count": c,
            "answer_correct_rate": by_subject[sub]["answer_correct_count"] / c if c else 0.0,
            "explanation_quality_avg": by_subject[sub]["explanation_quality_sum"] / c if c else 0.0,
            "edit_quality_avg": by_subject[sub]["edit_quality_sum"] / ec if ec else 0.0,
            "edit_valid_count": ec,
        }

    summary = {
        "run_dir": run_dir,
        "judge_model": model_name,
        "total_samples": total,
        "valid_samples": len(valid),
        "failed_samples": total - len(valid),
        "edit_excluded_samples": len(valid) - len(edit_valid),
        "edit_valid_samples": len(edit_valid),
        "answer_correct_rate": answer_correct_rate,
        "explanation_quality_avg": explanation_avg,
        "edit_quality_avg": edit_avg,
        "by_subject": by_subject,
        "generated_at": datetime.now().isoformat(),
    }

    summary_path = os.path.join(run_dir, "verifier_quality_summary.json")
    _write_json_atomic(summary_path, summary)

    if write_jsonl:
        jsonl_path = os.path.join(run_dir, "verifier_quality.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate verifier quality for all available steps using DMX Gemini3-flash")
    parser.add_argument("--run_dir", type=str, required=True, help="TTS run directory, e.g. output_TTS/<safe_run_tag>")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--force", action="store_true", help="Re-judge and overwrite existing meta_judge_step1.json")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=2.0)
    parser.add_argument("--no_jsonl", action="store_true", help="Disable writing verifier_quality.jsonl")
    parser.add_argument("--max_workers", type=int, default=8, help="Thread workers for concurrent API requests")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key is required. Please set --api_key or environment variable DMX_API_KEY")

    run_dir = os.path.abspath(args.run_dir)
    summary, records = evaluate_run(
        run_dir=run_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        force=args.force,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        write_jsonl=not args.no_jsonl,
        max_workers=args.max_workers,
    )

    print(json.dumps({"summary": summary, "records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
