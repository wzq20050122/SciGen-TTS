#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

TOOLKIT_PATH = os.environ.get("TOOLKIT_PATH", "/root/autodl-tmp/home/测评/data-process-toolkits")
SWIFT_PATH = os.environ.get("SWIFT_PATH", "/root/autodl-tmp/home/测评/vlm-train-prod")
for _path in [TOOLKIT_PATH, SWIFT_PATH]:
    if _path and os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from swift.llm import InferRequest, RequestConfig, VllmEngine

INPUT_DIR = Path("/root/autodl-tmp/TTS/Dataset/step3_重新找到高质量图片")
OUTPUT_DIR = Path("/root/autodl-tmp/TTS/output_TTS")
VERIFIER_MODEL = os.environ.get("VLLM_VERIFIER_MODEL", "/root/autodl-tmp/wzq/model/SciGen-Verifier-SFT")
VERIFIER_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "1024"))
VERIFIER_TEMPERATURE = float(os.environ.get("VLLM_TEMPERATURE", "0"))
VERIFIER_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
VERIFIER_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384"))

_ENGINE_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_ENGINE: Optional[VllmEngine] = None
_REQUEST_CONFIG: Optional[RequestConfig] = None

VERIFIER_PROMPT = """You are an expert scientific image verification model. Your task is to evaluate whether a generated image correctly follows the given instruction. You must analyze the correctness of the generated image step by step, then provide a final judgment.

**Task Types:**
- **Instruction Following**: Drawing/editing scientific diagrams based on explicit requirements (e.g., \"draw a line at x=5\", \"add axis labels\", \"mark point P\")
- **Reasoning**: Creating figures that require logical deduction (e.g., plotting calculated data points, geometric constructions based on given conditions)
- **Knowledge-based**: Illustrating scientific concepts from domain knowledge (e.g., \"draw Earth's internal layers\", \"sketch a plant cell structure\")

---

**Instruction:**
{instruction}

---

**Generated Image:**
<image>

---

**Evaluation Rules:**

1. The instruction is the ONLY evaluation standard — do not add requirements not mentioned in the instruction.
2. All equivalent implementations are correct.
3. Ignore all stylistic aspects unless explicitly required.
4. All explicitly requested elements must be present with correct values, positions, and labels.
5. The logical/scientific conclusion must be correct.

**Judgment:**
- **true**: All requirements fulfilled, scientifically/logically correct.
- **false**: Any required element missing, any value/position wrong, or any scientific/logical error.

**Output:**
<answer>true/false</answer>
<explanation>Explain why the judgment is correct or incorrect.</explanation>
<edit>If <answer> is false, provide concrete edit instructions to fix the image. If <answer> is true, do not output this tag.</edit>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--verifier-model", default=VERIFIER_MODEL)
    p.add_argument("--gen-model", default=os.environ.get("DMX_IMAGE_GEN_MODEL", os.environ.get("DMX_IMAGE_MODEL", "gpt-image-1.5")))
    p.add_argument("--edit-model", default=os.environ.get("DMX_IMAGE_EDIT_MODEL", os.environ.get("DMX_IMAGE_MODEL", "gpt-image-1.5")))
    p.add_argument("--run-date", default=None, help="Run date tag in folder name, format YYYYMMDD. Default=today")
    p.add_argument("--disable-auto-run-subdir", action="store_true", help="Disable auto creating model&date subfolder under --output-dir")
    p.add_argument("--summary-sync-root", type=Path, default=Path("/root/autodl-tmp/TTS/output_TTS"), help="Also save run_summary.json/run_meta.json under this root in model&date subfolder")
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--retry-count", type=int, default=2)
    p.add_argument("--retry-wait", type=float, default=30.0)
    p.add_argument("--retry-strategy", choices=["regenerate", "edit", "hybrid"], default="hybrid", help="How to handle verifier=false: full regenerate, image edit, or hybrid rule-based")
    p.add_argument("--hybrid-edit-max-lines", type=int, default=3, help="Hybrid: only when checklist item count < this and no global-restructure cues, prefer image edit")
    p.add_argument("--hybrid-regenerate-min-chars", type=int, default=500, help="Hybrid: long multi-action edit text tends to drift; prefer regenerate")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--image-ext", default=".png")
    p.add_argument("--verifier-max-tokens", type=int, default=VERIFIER_MAX_TOKENS)
    p.add_argument("--verifier-temperature", type=float, default=VERIFIER_TEMPERATURE)
    p.add_argument("--verifier-gpu-memory-utilization", type=float, default=VERIFIER_GPU_MEMORY_UTILIZATION)
    p.add_argument("--verifier-max-model-len", type=int, default=VERIFIER_MAX_MODEL_LEN)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_json_atomic(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def sanitize(text: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "sample"


def build_run_folder_name(gen_model: str, edit_model: str, run_date: str) -> str:
    """Build run folder name using the original RUN_TAG style.

    Keep dots and ampersands (e.g. wan2.6-t2i&wan2.6-image_20260411),
    while guarding against path separators and control characters.
    """
    base = f"{gen_model}&{edit_model}_{run_date}"
    # Keep most characters unchanged, only normalize path-breaking chars.
    base = base.replace("/", "_").replace("\\", "_")
    base = re.sub(r"[\x00-\x1f\x7f]+", "_", base)
    base = re.sub(r"\s+", "_", base).strip("._ ")
    return base or "run"


def iter_records(input_dir: Path) -> Iterable[Dict[str, Any]]:
    for jsonl in sorted(input_dir.glob("*.jsonl")):
        for row in read_jsonl(jsonl):
            row["_source_jsonl"] = str(jsonl)
            yield row


def run_template(template: str, *, payload_json: Path, output_path: Path, work_dir: Path) -> None:
    if not template.strip():
        raise RuntimeError("命令模板未配置。请设置 IMAGE_EDITOR_CMD_TEMPLATE")
    cmd = template.format(
        payload_json=shlex.quote(str(payload_json)),
        output_path=shlex.quote(str(output_path)),
        work_dir=shlex.quote(str(work_dir)),
    )
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    log_path = work_dir / "editor_command.log"
    log_text = "\n".join(
        [
            f"command: {cmd}",
            f"returncode: {result.returncode}",
            "",
            "stdout:",
            result.stdout or "",
            "",
            "stderr:",
            result.stderr or "",
        ]
    )
    log_path.write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {cmd}；日志见 {log_path}")


def extract_answer(text: str) -> Optional[bool]:
    m = re.search(r"<answer>\s*(true|false)\s*</answer>", text, re.I | re.S)
    if m:
        return m.group(1).lower() == "true"
    box_match = re.search(r"<\|begin_of_box\|>\s*(true|false)", text, re.I)
    if box_match:
        return box_match.group(1).lower() == "true"
    t = text.strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    return None


def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.I | re.S)
    if m:
        val = m.group(1).strip()
        return val or None

    if tag == "explanation":
        box_match = re.search(r"<\|begin_of_box\|>\s*(?:true|false)\s*(.*?)<\|end_of_box\|>", text, re.I | re.S)
        if box_match:
            val = box_match.group(1).strip()
            return val or None
        answer = extract_answer(text)
        if answer is not None:
            cleaned = re.sub(r"<\|begin_of_box\|>\s*(true|false)", "", text, flags=re.I)
            cleaned = re.sub(r"<\|end_of_box\|>", "", cleaned, flags=re.I)
            cleaned = cleaned.strip()
            return cleaned or None

    return None


def parse_verifier(raw_text: str) -> Dict[str, Any]:
    return {
        "answer": extract_answer(raw_text),
        "explanation": extract_tag(raw_text, "explanation"),
        "edit": extract_tag(raw_text, "edit"),
        "raw_text": raw_text,
    }


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".gif":
        mime = "image/gif"
    else:
        mime = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def build_verifier_messages(prompt: str, image_path: Path) -> list[dict]:
    image_token = "<image>"
    image_part = {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}

    if image_token not in prompt:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_part,
                ],
            }
        ]

    content: list[dict[str, Any]] = []
    pieces = prompt.split(image_token)
    for i, piece in enumerate(pieces):
        if piece:
            content.append({"type": "text", "text": piece})
        if i < len(pieces) - 1:
            content.append(image_part)

    return [{"role": "user", "content": content}]


def get_verifier_runtime(args: argparse.Namespace) -> tuple[VllmEngine, RequestConfig]:
    global _ENGINE, _REQUEST_CONFIG
    if _ENGINE is None or _REQUEST_CONFIG is None:
        with _ENGINE_LOCK:
            if _ENGINE is None or _REQUEST_CONFIG is None:
                os.environ.setdefault("MAX_PIXELS", str(1024 * 1024 * 4))
                _ENGINE = VllmEngine(
                    args.verifier_model,
                    max_model_len=args.verifier_max_model_len,
                    gpu_memory_utilization=args.verifier_gpu_memory_utilization,
                )
                _REQUEST_CONFIG = RequestConfig(
                    max_tokens=args.verifier_max_tokens,
                    temperature=args.verifier_temperature,
                )
    return _ENGINE, _REQUEST_CONFIG


def clean_edit_checklist(edit: Optional[str]) -> Optional[str]:
    """Normalize verifier edit text into a compact checklist."""
    edit_clean = (edit or "").strip()
    if not edit_clean:
        return None

    lines = [line.strip() for line in edit_clean.splitlines() if line.strip()]
    checklist: list[str] = []
    for line in lines:
        item = re.sub(r"^[-*•\d\.)\s]+", "", line).strip()
        if not item:
            continue
        checklist.append(f"- {item}")

    if checklist:
        return "\n".join(checklist)

    compact = re.sub(r"\s+", " ", edit_clean).strip()
    if compact:
        return f"- {compact}"
    return None


def build_regeneration_instruction(original_instruction: str, edit_checklist: Optional[str]) -> str:
    """Regenerate from original prompt + cleaned edit checklist only."""
    original_clean = (original_instruction or "").strip()
    checklist_clean = (edit_checklist or "").strip()

    if not checklist_clean:
        checklist_clean = "- Ensure all explicit requirements are satisfied."

    return (
        f"{original_clean}\n\n"
        "[Revision checklist (must be satisfied item by item in this regeneration)]\n"
        f"{checklist_clean}"
    )


def build_edit_instruction(original_instruction: str, edit_checklist: Optional[str]) -> str:
    """Build edit prompt that preserves correct parts and applies local fixes."""
    original_clean = (original_instruction or "").strip()
    checklist_clean = (edit_checklist or "").strip()
    if not checklist_clean:
        checklist_clean = "- Ensure all explicit requirements are satisfied."

    return (
        "Edit the existing image and preserve already-correct content.\n"
        "Apply the checklist item by item:\n"
        f"{checklist_clean}\n\n"
        "Original instruction (for reference):\n"
        f"{original_clean}"
    )


def decide_retry_action(edit_raw: Optional[str], edit_checklist: Optional[str], args: argparse.Namespace) -> str:
    """Choose next step action when verifier=false: regenerate or edit."""
    if args.retry_strategy == "regenerate":
        return "regenerate"
    if args.retry_strategy == "edit":
        return "edit"

    checklist = (edit_checklist or "").strip()
    if not checklist:
        return "regenerate"

    text = f"{(edit_raw or '').lower()}\n{checklist.lower()}"
    line_count = len([ln for ln in checklist.splitlines() if ln.strip()])

    # Global-restructure cues: if any is hit, bias to regenerate.
    global_patterns = [
        r"\barrange\b",
        r"\brearrange\b",
        r"\breorder\b",
        r"\bentire\b",
        r"\boverall\b",
        r"\ball\b",
        r"\beach\b",
        r"\bevery\b",
        r"\bfix\s+all\b",
        r"\breplace\s+all\b",
        r"\bcorrect\b[\s\S]{0,40}\border\b",
        r"\bduplicate\s+keys\b",
        r"\bconsistency\b",
        r"\bacross\b",
        r"\bwhole\s+image\b",
        r"\bouter\s+circle\b",
        r"\binner\s+circle\b",
        r"\blayout\b",
        r"\bglobal\b",
        r"\boverall\s+structure\b",
        r"\bre-?draw\b",
        r"\breconstruct\b",
        r"\brebuild\b",
    ]
    global_hits = sum(1 for p in global_patterns if re.search(p, text))

    # Strict rule:
    # edit iff (checklist item count < threshold) AND (no global-restructure cues).
    if line_count < args.hybrid_edit_max_lines and global_hits == 0:
        return "edit"

    return "regenerate"


def infer_target_size(image_path: Optional[str]) -> str:
    if not image_path or not Path(image_path).exists():
        return "auto"
    with Image.open(image_path) as img:
        width, height = img.size
    if width == height:
        return "1024x1024"
    if width > height:
        return "1536x1024"
    return "1024x1536"


def build_sample_name(record: Dict[str, Any], idx: int) -> str:
    source = Path(record.get("_source_jsonl", "unknown")).stem
    rid = record.get("id") or f"sample_{idx:06d}"
    return sanitize(f"{source}_{rid}")


def get_step_dir(sample_dir: Path, step: int) -> Path:
    return sample_dir / f"step{step}"


def render_progress(done: int, total: int, *, width: int = 30) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0"
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = done / total * 100
    return f"[{bar}] {done}/{total} ({percent:5.1f}%)"


def print_progress(done: int, total: int) -> None:
    print(f"\rProgress {render_progress(done, total)}", end="", flush=True)
    if done >= total:
        print()


def run_with_retry(action_name: str, fn, *, retry_count: int, retry_wait: float):
    last_error: Exception | None = None
    for attempt in range(retry_count + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= retry_count:
                break
            wait_seconds = retry_wait * (attempt + 1)
            print(
                f"[{action_name}] attempt {attempt + 1}/{retry_count + 1} failed: {exc}; "
                f"retrying in {wait_seconds:.1f}s"
            )
            time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


def run_editor(payload: Dict[str, Any], payload_path: Path, output_image: Path, work_dir: Path, *, retry_count: int, retry_wait: float) -> None:
    write_json_atomic(payload_path, payload)

    def _execute() -> None:
        run_template(
            os.environ.get("IMAGE_EDITOR_CMD_TEMPLATE", ""),
            payload_json=payload_path,
            output_path=output_image,
            work_dir=work_dir,
        )
        if not output_image.exists():
            raise RuntimeError(f"图片未生成: {output_image}")

    run_with_retry(
        f"editor:{output_image.name}",
        _execute,
        retry_count=retry_count,
        retry_wait=retry_wait,
    )


def run_verifier(prompt: str, image_path: Path, model_path: str, payload_path: Path, output_text: Path, *, retry_count: int, retry_wait: float, args: argparse.Namespace) -> Dict[str, Any]:
    payload = {
        "verifier_model_path": str(model_path),
        "instruction": prompt,
        "image_path": str(image_path),
        "verifier_prompt": VERIFIER_PROMPT.format(instruction=prompt.strip()),
        "output_text_path": str(output_text),
    }
    write_json_atomic(payload_path, payload)

    def _execute() -> Dict[str, Any]:
        engine, request_config = get_verifier_runtime(args)
        infer_request = InferRequest(
            messages=build_verifier_messages(payload["verifier_prompt"], image_path)
        )
        with _INFER_LOCK:
            responses = engine.infer([infer_request], request_config=request_config)
        if not responses or not responses[0].choices:
            raise RuntimeError("Verifier 未返回有效 choices")
        content = responses[0].choices[0].message.content or ""
        output_text.write_text(content, encoding="utf-8")
        return parse_verifier(content)

    return run_with_retry(
        f"verifier:{output_text.name}",
        _execute,
        retry_count=retry_count,
        retry_wait=retry_wait,
    )


def process_record(record: Dict[str, Any], idx: int, args: argparse.Namespace) -> Dict[str, Any]:
    sample_dir = args.output_dir / build_sample_name(record, idx)
    ensure_dir(sample_dir)
    final_json = sample_dir / "final_result.json"
    if args.skip_existing and final_json.exists():
        result = json.loads(final_json.read_text(encoding="utf-8"))
        result["_idx"] = idx
        result["_skipped_existing"] = True
        return result

    write_json_atomic(sample_dir / "source_record.json", record)
    write_text(sample_dir / "prompt.txt", record["prompt"])

    current_instruction = record["prompt"]
    prev_image: Optional[Path] = None
    history: List[Dict[str, Any]] = []
    last_image: Optional[Path] = None
    success = False

    def persist_final_result() -> None:
        result = {
            "id": record.get("id"),
            "source_jsonl": record.get("_source_jsonl"),
            "prompt": record.get("prompt"),
            "original_image_path": record.get("image_path"),
            "success": success,
            "steps_used": len(history),
            "final_image_path": str(sample_dir / f"final{args.image_ext}") if last_image else None,
            "history": history,
            "_idx": idx,
            "_skipped_existing": False,
        }
        write_json_atomic(final_json, result)

    def load_existing_step(step: int) -> bool:
        nonlocal success, last_image, prev_image, current_instruction
        step_dir = get_step_dir(sample_dir, step)
        image_path = step_dir / f"image{args.image_ext}"
        parsed_path = step_dir / "verifier_parsed.json"
        verifier_raw_path = step_dir / "verifier_raw.txt"
        if not (image_path.exists() and parsed_path.exists()):
            return False
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        step_info = {
            "step": step,
            "step_dir": str(step_dir),
            "image_path": str(image_path),
            "answer": parsed.get("answer"),
            "explanation": parsed.get("explanation"),
            "edit": parsed.get("edit"),
            "raw_text_path": str(verifier_raw_path),
        }
        history.append(step_info)
        last_image = image_path
        if parsed.get("answer") is True:
            success = True
            return True
        cleaned_edit = clean_edit_checklist(parsed.get("edit"))
        retry_action = decide_retry_action(parsed.get("edit"), cleaned_edit, args)
        if retry_action == "edit":
            prev_image = image_path
            current_instruction = build_edit_instruction(record["prompt"], cleaned_edit)
        else:
            prev_image = None
            current_instruction = build_regeneration_instruction(record["prompt"], cleaned_edit)
        return False

    start_step = 1
    for step in range(1, args.max_steps + 1):
        if not load_existing_step(step):
            start_step = step
            break
        start_step = step + 1
        persist_final_result()
        if success:
            return json.loads(final_json.read_text(encoding="utf-8"))

    for step in range(start_step, args.max_steps + 1):
        step_dir = get_step_dir(sample_dir, step)
        ensure_dir(step_dir)
        image_path = step_dir / f"image{args.image_ext}"
        editor_payload = step_dir / "editor_payload.json"
        verifier_payload = step_dir / "verifier_payload.json"
        verifier_raw = step_dir / "verifier_raw.txt"
        verifier_parsed = step_dir / "verifier_parsed.json"

        prev_info = history[-1] if history else {}
        mode = "initial" if step == 1 else ("edit" if prev_image else "regenerate")
        run_editor(
            {
                "task_type": "tts_image_editing",
                "step": step,
                "mode": mode,
                "record": record,
                "instruction": record["prompt"],
                "current_instruction": current_instruction,
                "previous_image": str(prev_image) if prev_image else None,
                "reference_image": record.get("image_path"),
                "target_size": infer_target_size(record.get("image_path")),
                "verifier_explanation": None,
                "verifier_edit": clean_edit_checklist(prev_info.get("edit")),
                "output_image_path": str(image_path),
                "sample_dir": str(sample_dir),
                "step_dir": str(step_dir),
            },
            editor_payload,
            image_path,
            step_dir,
            retry_count=args.retry_count,
            retry_wait=args.retry_wait,
        )

        parsed = run_verifier(
            record["prompt"],
            image_path,
            str(args.verifier_model),
            verifier_payload,
            verifier_raw,
            retry_count=args.retry_count,
            retry_wait=args.retry_wait,
            args=args,
        )
        write_json_atomic(verifier_parsed, parsed)

        step_info = {
            "step": step,
            "step_dir": str(step_dir),
            "image_path": str(image_path),
            "answer": parsed["answer"],
            "explanation": parsed["explanation"],
            "edit": parsed["edit"],
            "raw_text_path": str(verifier_raw),
        }
        history.append(step_info)
        last_image = image_path
        persist_final_result()

        if parsed["answer"] is True:
            success = True
            persist_final_result()
            break

        cleaned_edit = clean_edit_checklist(parsed["edit"])
        retry_action = decide_retry_action(parsed["edit"], cleaned_edit, args)
        if retry_action == "edit":
            prev_image = image_path
            current_instruction = build_edit_instruction(record["prompt"], cleaned_edit)
        else:
            prev_image = None
            current_instruction = build_regeneration_instruction(record["prompt"], cleaned_edit)

    if last_image is None:
        raise RuntimeError(f"样本没有生成图片: {record.get('id')}")

    final_image = sample_dir / f"final{args.image_ext}"
    shutil.copy2(last_image, final_image)
    persist_final_result()

    return json.loads(final_json.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()

    run_date = args.run_date or datetime.now().strftime("%Y%m%d")
    run_folder = build_run_folder_name(args.gen_model, args.edit_model, run_date)
    if not args.disable_auto_run_subdir:
        args.output_dir = args.output_dir / run_folder

    # Expose selected models to editor runner via environment.
    os.environ["DMX_IMAGE_GEN_MODEL"] = args.gen_model
    os.environ["DMX_IMAGE_EDIT_MODEL"] = args.edit_model

    ensure_dir(args.output_dir)
    if not args.input_dir.exists():
        raise RuntimeError(f"输入目录不存在: {args.input_dir}")
    if args.workers < 1:
        raise RuntimeError("--workers 必须 >= 1")
    if args.retry_count < 0:
        raise RuntimeError("--retry-count 必须 >= 0")
    if args.retry_wait < 0:
        raise RuntimeError("--retry-wait 必须 >= 0")

    records = list(iter_records(args.input_dir))
    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        write_json(args.output_dir / "run_summary.json", [])
        print("Done. total=0")
        return

    indexed_records = list(enumerate(records, start=1))
    total_records = len(indexed_records)
    all_results: list[dict[str, Any]] = []
    failed_results: list[dict[str, Any]] = []
    completed_count = 0
    print_progress(0, total_records)

    if args.workers == 1:
        for idx, record in indexed_records:
            try:
                result = process_record(record, idx, args)
                all_results.append(result)
                skipped = result.get("_skipped_existing", False)
                extra = " skipped_existing=True" if skipped else ""
                print(
                    f"[{idx}] id={result['id']} success={result['success']} "
                    f"steps={result['steps_used']}{extra}"
                )
            except Exception as exc:
                rid = record.get("id") or f"sample_{idx:06d}"
                failed_results.append(
                    {
                        "idx": idx,
                        "id": rid,
                        "source_jsonl": record.get("_source_jsonl"),
                        "error": str(exc),
                    }
                )
                print(f"[{idx}] id={rid} failed: {exc}")
            finally:
                completed_count += 1
                print_progress(completed_count, total_records)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_meta = {
                executor.submit(process_record, record, idx, args): (idx, record)
                for idx, record in indexed_records
            }
            for future in as_completed(future_to_meta):
                idx, record = future_to_meta[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    skipped = result.get("_skipped_existing", False)
                    extra = " skipped_existing=True" if skipped else ""
                    print(
                        f"[{idx}] id={result['id']} success={result['success']} "
                        f"steps={result['steps_used']}{extra}"
                    )
                except Exception as exc:
                    rid = record.get("id") or f"sample_{idx:06d}"
                    failed_results.append(
                        {
                            "idx": idx,
                            "id": rid,
                            "source_jsonl": record.get("_source_jsonl"),
                            "error": str(exc),
                        }
                    )
                    print(f"[{idx}] id={rid} failed: {exc}")
                finally:
                    completed_count += 1
                    print_progress(completed_count, total_records)

    all_results.sort(key=lambda item: item.get("_idx", 0))
    for item in all_results:
        item.pop("_idx", None)
        item.pop("_skipped_existing", None)

    failed_results.sort(key=lambda item: item.get("idx", 0))

    run_summary_path = args.output_dir / "run_summary.json"
    failed_summary_path = args.output_dir / "failed_samples.json"
    verifier_success_count = sum(1 for item in all_results if item.get("success") is True)
    verifier_failed_count = sum(1 for item in all_results if item.get("success") is False)
    execution_failed_count = len(failed_results)

    run_meta_payload = {
        "run_folder": str(args.output_dir),
        "gen_model": args.gen_model,
        "edit_model": args.edit_model,
        "run_date": run_date,
        "total": len(records),
        "completed_count": len(all_results),
        "success_count": verifier_success_count,
        "verifier_failed_count": verifier_failed_count,
        "execution_failed_count": execution_failed_count,
        "failed_count": verifier_failed_count + execution_failed_count,
    }

    write_json(run_summary_path, all_results)
    write_json(failed_summary_path, failed_results)
    write_json(args.output_dir / "run_meta.json", run_meta_payload)

    # Sync summary/meta to canonical output_TTS/<model&model_date>/ as requested.
    sync_root = args.summary_sync_root
    sync_dir = sync_root / run_folder
    ensure_dir(sync_dir)
    write_json(sync_dir / "run_summary.json", all_results)
    write_json(sync_dir / "failed_samples.json", failed_results)
    write_json(sync_dir / "run_meta.json", {**run_meta_payload, "run_folder": str(sync_dir)})

    print(
        "Done. "
        f"total={len(records)} "
        f"completed={len(all_results)} "
        f"success={verifier_success_count} "
        f"verifier_failed={verifier_failed_count} "
        f"execution_failed={execution_failed_count}"
    )
    print(f"Run folder: {args.output_dir}")
    print(f"Summary synced to: {sync_dir}")

    if len(all_results) == 0 and len(failed_results) > 0:
        raise RuntimeError("所有样本均失败，终止流水线")


if __name__ == "__main__":
    main()
