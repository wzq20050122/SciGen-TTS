import json
import os
import re
import hashlib
import base64
import requests
from PIL import Image
import io
import ast
import argparse
from datetime import datetime
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval_prompt import prompt_for_eval

# DMX API configuration (aligned with TTS style)
# Priority: CLI args > environment variables > defaults
DEFAULT_API_KEY = os.environ.get("DMX_API_KEY", "")
DEFAULT_BASE_URL = os.environ.get("DMX_BASE_URL", "https://www.dmxapi.cn/v1")
DEFAULT_MODEL_NAME = "gemini-3-flash-preview"


def write_json_atomic(path: Path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _dmx_headers(api_key, json_request=False):
    headers = {"Authorization": api_key}
    if json_request:
        headers["Content-Type"] = "application/json"
    return headers


def _resolve_gen_image_path(data, img_save_dir, step_key=None):
    """
    Resolve generated image path with priority:
    1) explicit step_key mapping from annotation: step_image_paths[stepN]
    2) legacy fields: step1_image_path/final_image_path
    3) fallback old style: img_save_dir/{id}.png
    4) fallback subject style: img_save_dir/{subject}/{id}.png
    """
    mode_hint = os.path.basename(os.path.normpath(img_save_dir)).lower()
    chosen_step_key = (step_key or mode_hint or "").lower()

    candidates = []

    step_image_paths = data.get("step_image_paths")
    if isinstance(step_image_paths, dict):
        # explicit requested step key
        if chosen_step_key in step_image_paths:
            candidates.append(step_image_paths.get(chosen_step_key))
        # fallback by mode hint if it looks like stepN
        if mode_hint in step_image_paths:
            candidates.append(step_image_paths.get(mode_hint))

    if chosen_step_key == "step1" and data.get("step1_image_path"):
        candidates.append(data.get("step1_image_path"))
    if chosen_step_key == "final" and data.get("final_image_path"):
        candidates.append(data.get("final_image_path"))

    # If step key is unclear, still try common fields.
    if data.get("step1_image_path"):
        candidates.append(data.get("step1_image_path"))
    if data.get("final_image_path"):
        candidates.append(data.get("final_image_path"))

    for p in candidates:
        if not p:
            continue
        if os.path.isabs(p) and os.path.exists(p):
            return p
        rel_try = os.path.join(img_save_dir, p)
        if os.path.exists(rel_try):
            return rel_try

    old_style = os.path.join(img_save_dir, f"{data['id']}.png")
    if os.path.exists(old_style):
        return old_style

    subject = data.get("subject")
    if subject:
        by_subject = os.path.join(img_save_dir, subject, f"{data['id']}.png")
        if os.path.exists(by_subject):
            return by_subject

    return old_style


def _resolve_gt_image_path(data, data_dir):
    p = data.get("image_path")
    if p and os.path.isabs(p):
        return p
    return os.path.join(data_dir, "images", p)


def _safe_name(text: str) -> str:
    import re

    text = re.sub(r"[^0-9a-zA-Z_\-\.\u4e00-\u9fff&]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "run"


def _build_run_tag(gen_model: str, edit_model: str, run_date: str) -> str:
    return _safe_name(f"{gen_model}&{edit_model}_{run_date}")


def _list_available_step_keys(all_data):
    steps = set()
    for data in all_data:
        mapping = data.get("step_image_paths")
        if isinstance(mapping, dict):
            for k in mapping.keys():
                if isinstance(k, str) and re.fullmatch(r"step\d+", k):
                    steps.add(k)
    return sorted(steps, key=lambda x: int(x[4:]))


def _derive_step_eval_dir(eval_save_dir: str, step_key: str) -> str:
    base = eval_save_dir.rstrip("/")
    leaf = os.path.basename(base)
    if re.fullmatch(r"eval_results_step\d+", leaf):
        return os.path.join(os.path.dirname(base), f"eval_results_{step_key}")
    return base


def _resolve_reuse_from_previous_step_path(data, eval_save_dir: str, step_key: str):
    """
    Reuse rule:
    - target is step2/step3/... (not step1)
    - if the target step image does not exist, reuse the nearest previous available step eval json
      by searching step{N-1}, step{N-2}, ... down to step1.
    """
    if not isinstance(step_key, str) or not re.fullmatch(r"step\d+", step_key):
        return None

    try:
        target_step_num = int(step_key[4:])
    except Exception:
        return None
    if target_step_num <= 1:
        return None

    mapping = data.get("step_image_paths")
    if isinstance(mapping, dict) and mapping.get(step_key):
        return None

    for prev_step_num in range(target_step_num - 1, 0, -1):
        prev_step_key = f"step{prev_step_num}"
        if isinstance(mapping, dict) and not mapping.get(prev_step_key):
            continue

        prev_eval_dir = _derive_step_eval_dir(eval_save_dir, prev_step_key)
        prev_json = os.path.join(prev_eval_dir, f"{data['id']}.json")
        if os.path.exists(prev_json):
            return prev_json

    return None


def _build_reused_eval_result(reuse_json_path: str, data, gen_img_save_path: str, gt_img_path: str):
    eval_result = json.loads(open(reuse_json_path, "r", encoding="utf-8").read())
    eval_result.update(data)
    eval_result["gen_img_path"] = gen_img_save_path
    eval_result["gt_img_path"] = gt_img_path
    eval_result["reused_from"] = reuse_json_path
    if "image_path" in eval_result:
        del eval_result["image_path"]
    return eval_result


def _run_eval_once(
    all_data,
    img_save_dir,
    eval_save_dir,
    data_dir,
    inference_function,
    sampled_ids,
    api_key,
    base_url,
    model_name,
    step_key,
    max_workers,
    dedup_by_hash,
):
    if dedup_by_hash:
        # Dedup requires a shared hash cache; run single-process for determinism.
        inference_and_eval_dedup(
            all_data=all_data,
            img_save_dir=img_save_dir,
            eval_save_dir=eval_save_dir,
            data_dir=data_dir,
            inference_function=inference_function,
            sampled_ids=sampled_ids,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            step_key=step_key,
            max_workers=max_workers,
        )
        return

    if max_workers > 0:
        print(f"Evaluating {step_key} with {max_workers} workers ...")
        args_list = [
            (
                data,
                img_save_dir,
                eval_save_dir,
                data_dir,
                inference_function,
                sampled_ids,
                api_key,
                base_url,
                model_name,
                step_key,
            )
            for data in all_data
        ]

        process_map(
            _inference_and_eval_single,
            args_list,
            max_workers=max_workers,
            desc=f"Evaluating {step_key}",
        )
    else:
        print(f"Evaluating {step_key} without multiprocessing ...")
        for data in tqdm(all_data):
            inference_and_eval_single(
                data,
                img_save_dir,
                eval_save_dir,
                data_dir,
                inference_function,
                sampled_ids,
                api_key,
                base_url,
                model_name,
                step_key,
            )


def encode_image(image_path, target_size=1024, fmt="JPEG"):
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
    image_data = img_buffer.getvalue()
    return base64.b64encode(image_data).decode("utf-8")


def call_vlm_eval(
    image_path,
    image_path2,
    text_prompt,
    api_key,
    base_url,
    model_name,
    max_tokens=4096,
    img_size=768,
):
    base64_image = encode_image(image_path, target_size=img_size)
    base64_image2 = encode_image(image_path2, target_size=img_size)

    content = [
        {"type": "text", "text": text_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}", "detail": "high"},
        },
    ]

    json_parameters = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "max_completion_tokens": max_tokens,
    }

    try:
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers=_dmx_headers(api_key, json_request=True),
            json=json_parameters,
            stream=False,
            timeout=180,
        )
        response.raise_for_status()

        response_json = response.json()
        ret = response_json["choices"][0]["message"]["content"]
        return ret

    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        try:
            response_json = response.json()
            if "error" in response_json:
                print(response_json["error"])
        except Exception:
            pass
        return None


def call_img_gen(text_prompt, save_path, api_key, base_url):
    response = requests.post(
        base_url.rstrip("/") + "/images/generations",
        headers=_dmx_headers(api_key, json_request=True),
        json={
            "model": "gpt-image-1",
            "prompt": text_prompt,
            "n": 1,
            "moderation": "low",
            "quality": "high",
        },
        timeout=180,
    )
    response.raise_for_status()
    image_base64 = response.json()["data"][0]["b64_json"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    print(f"Image saved to {save_path}")


def eval_single(gen_img_path, gt_img_path, t2i_prompt, scoring_points, api_key, base_url, model_name):
    assert os.path.exists(gen_img_path), f"Image {gen_img_path} not found"
    assert os.path.exists(gt_img_path), f"Image {gt_img_path} not found"

    count = 0
    while True:
        count += 1
        if count > 3:
            print(f"Failed to get response from VLM after {count} times")
            raise ValueError("Failed to get response from VLM")

        response = call_vlm_eval(
            image_path=gen_img_path,
            image_path2=gt_img_path,
            text_prompt=prompt_for_eval.format(prompt=t2i_prompt, scoring_points=scoring_points),
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            max_tokens=16384,
            img_size=768,
        )
        if response is not None:
            try:
                response = json.loads(response.split("```json")[-1].split("```")[0])
            except Exception:
                try:
                    response = ast.literal_eval(response.split("```json")[-1].split("```")[0])
                except Exception:
                    continue
            try:
                assert "global_evaluation" in response and "answers" in response, f"Invalid response: {response}"
                assert len(response["answers"]) == len(scoring_points), (
                    f"Invalid number of answers: {len(response['answers'])}"
                )
                assert all(item["answer"] in [0, 1] for item in response["answers"]), (
                    f"Invalid answer: {response['answers']}"
                )
                assert (
                    "Clarity and Readability" in response["global_evaluation"]
                    and "Logical Consistency" in response["global_evaluation"]
                    and "Spelling" in response["global_evaluation"]
                ), f"Invalid global evaluation: {response['global_evaluation']}"
            except Exception:
                continue
            break

    return response


def inference_and_eval_single(
    data,
    img_save_dir,
    eval_save_dir,
    data_dir,
    inference_function,
    sampled_ids,
    api_key,
    base_url,
    model_name,
    step_key=None,
):
    json_save_path = os.path.join(eval_save_dir, f"{data['id']}.json")
    gen_img_save_path = _resolve_gen_image_path(data, img_save_dir, step_key=step_key)

    if sampled_ids is not None and data["id"] not in sampled_ids:
        return

    if os.path.exists(json_save_path):
        print("Skipping already evaluated data ...", data["id"])
        return

    if os.path.exists(gen_img_save_path):
        print(f"Image already generated: {gen_img_save_path}")
    else:
        os.makedirs(os.path.dirname(gen_img_save_path), exist_ok=True)
        print("Generating image ...")
        assert inference_function is not None, (
            f"Image {data['id']} has not been generated and inference function is not set"
        )
        inference_function(text_prompt=data["prompt"], save_path=gen_img_save_path)

    gt_img_path = _resolve_gt_image_path(data, data_dir)

    reuse_json_path = _resolve_reuse_from_previous_step_path(data, eval_save_dir, step_key or "")
    if reuse_json_path:
        eval_result = _build_reused_eval_result(
            reuse_json_path=reuse_json_path,
            data=data,
            gen_img_save_path=gen_img_save_path,
            gt_img_path=gt_img_path,
        )
    else:
        eval_result = eval_single(
            gen_img_path=gen_img_save_path,
            gt_img_path=gt_img_path,
            t2i_prompt=data["prompt"],
            scoring_points=[item["question"] for item in data["scoring_points"]],
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )

        eval_result.update(data)
        eval_result["gen_img_path"] = gen_img_save_path
        eval_result["gt_img_path"] = gt_img_path
        del eval_result["image_path"]

    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    write_json_atomic(Path(json_save_path), eval_result)
    print(f"Saved eval results to {json_save_path}")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def inference_and_eval_dedup(
    all_data,
    img_save_dir,
    eval_save_dir,
    data_dir,
    inference_function,
    sampled_ids,
    api_key,
    base_url,
    model_name,
    step_key=None,
    max_workers=8,
):
    """
    Two-stage dedup pipeline:
    1) parallel hash scan and grouping
    2) evaluate one canonical sample per hash, then fan out results to all members

    Also supports step-level reuse from step1 for samples that did not generate later-step images.
    """
    os.makedirs(eval_save_dir, exist_ok=True)

    hash_cache_path = os.path.join(eval_save_dir, "_hash_cache.json")
    if os.path.exists(hash_cache_path):
        try:
            with open(hash_cache_path, "r", encoding="utf-8") as f:
                hash_cache = json.load(f)
        except Exception:
            hash_cache = {}
    else:
        hash_cache = {}

    # hash -> canonical result json path
    hash_to_result = {
        v.get("image_hash"): k
        for k, v in hash_cache.items()
        if isinstance(v, dict) and v.get("image_hash") and isinstance(k, str)
    }

    candidates = []
    for data in all_data:
        rid = data["id"]
        if sampled_ids is not None and rid not in sampled_ids:
            continue

        json_save_path = os.path.join(eval_save_dir, f"{rid}.json")
        if os.path.exists(json_save_path):
            continue

        gen_img_save_path = _resolve_gen_image_path(data, img_save_dir, step_key=step_key)
        if not os.path.exists(gen_img_save_path):
            os.makedirs(os.path.dirname(gen_img_save_path), exist_ok=True)
            assert inference_function is not None, (
                f"Image {rid} has not been generated and inference function is not set"
            )
            inference_function(text_prompt=data["prompt"], save_path=gen_img_save_path)

        gt_img_path = _resolve_gt_image_path(data, data_dir)
        candidates.append(
            {
                "data": data,
                "id": rid,
                "json_save_path": json_save_path,
                "gen_img_save_path": gen_img_save_path,
                "gt_img_path": gt_img_path,
            }
        )

    if not candidates:
        write_json_atomic(Path(hash_cache_path), hash_cache)
        return

    # Stage 0: fast path reuse from step1 (no hashing/VLM needed)
    pending = []
    for item in candidates:
        data = item["data"]
        reuse_json_path = _resolve_reuse_from_previous_step_path(data, eval_save_dir, step_key or "")
        if reuse_json_path:
            eval_result = _build_reused_eval_result(
                reuse_json_path=reuse_json_path,
                data=data,
                gen_img_save_path=item["gen_img_save_path"],
                gt_img_path=item["gt_img_path"],
            )
            if not eval_result.get("image_hash"):
                eval_result["image_hash"] = _file_sha256(item["gen_img_save_path"])

            write_json_atomic(Path(item["json_save_path"]), eval_result)

            hash_cache[item["json_save_path"]] = {
                "id": item["id"],
                "image_hash": eval_result.get("image_hash"),
            }
            if eval_result.get("image_hash"):
                hash_to_result[eval_result["image_hash"]] = item["json_save_path"]
        else:
            pending.append(item)

    if not pending:
        write_json_atomic(Path(hash_cache_path), hash_cache)
        return

    # Stage 1: parallel hash computation
    workers = max(1, int(max_workers or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_file_sha256, item["gen_img_save_path"]): idx
            for idx, item in enumerate(pending)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Hashing {step_key or 'unknown'}"):
            idx = futures[fut]
            pending[idx]["image_hash"] = fut.result()

    # Stage 2: group by hash
    groups = {}
    for item in pending:
        h = item["image_hash"]
        groups.setdefault(h, []).append(item)

    # Stage 3: evaluate one canonical sample per hash (or reuse cached canonical)
    # VLM API calls are executed concurrently for cache-miss hash groups.
    canonical_by_hash = {}
    to_eval = []

    for h, items in groups.items():
        canonical_path = hash_to_result.get(h)
        if canonical_path and os.path.exists(canonical_path):
            canonical_eval = json.loads(open(canonical_path, "r", encoding="utf-8").read())
            canonical_by_hash[h] = {
                "eval": canonical_eval,
                "canonical_path": canonical_path,
                "rep_json_save_path": items[0]["json_save_path"],
            }
        else:
            to_eval.append((h, items))

    if to_eval:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(to_eval)))) as executor:
            futures = {}
            for h, items in to_eval:
                rep = items[0]
                rep_data = rep["data"]
                fut = executor.submit(
                    eval_single,
                    gen_img_path=rep["gen_img_save_path"],
                    gt_img_path=rep["gt_img_path"],
                    t2i_prompt=rep_data["prompt"],
                    scoring_points=[item["question"] for item in rep_data["scoring_points"]],
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                )
                futures[fut] = (h, rep)

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {step_key or 'unknown'} (dedup)"):
                h, rep = futures[fut]
                rep_data = rep["data"]
                canonical_eval = fut.result()
                canonical_eval.update(rep_data)
                canonical_eval["gen_img_path"] = rep["gen_img_save_path"]
                canonical_eval["gt_img_path"] = rep["gt_img_path"]
                canonical_eval["image_hash"] = h
                if "image_path" in canonical_eval:
                    del canonical_eval["image_path"]

                canonical_by_hash[h] = {
                    "eval": canonical_eval,
                    "canonical_path": None,
                    "rep_json_save_path": rep["json_save_path"],
                }

    # Fan out canonical result to all samples in each hash group
    for h, items in groups.items():
        canonical_entry = canonical_by_hash[h]
        canonical_eval = canonical_entry["eval"]
        canonical_path = canonical_entry["canonical_path"]
        rep_json_save_path = canonical_entry["rep_json_save_path"]

        for item in items:
            eval_result = dict(canonical_eval)
            eval_result.update(item["data"])
            eval_result["gen_img_path"] = item["gen_img_save_path"]
            eval_result["gt_img_path"] = item["gt_img_path"]
            eval_result["image_hash"] = h

            if canonical_path and os.path.exists(canonical_path):
                eval_result["reused_from"] = canonical_path
            elif item["json_save_path"] != rep_json_save_path:
                eval_result["reused_from"] = rep_json_save_path

            if "image_path" in eval_result:
                del eval_result["image_path"]

            write_json_atomic(Path(item["json_save_path"]), eval_result)

            hash_cache[item["json_save_path"]] = {"id": item["id"], "image_hash": h}

        # update canonical index for following groups/runs
        hash_to_result[h] = rep_json_save_path
    with open(hash_cache_path, "w", encoding="utf-8") as f:
        json.dump(hash_cache, f, ensure_ascii=False, indent=2)


def _inference_and_eval_single(args):
    (
        data,
        img_save_dir,
        eval_save_dir,
        data_dir,
        inference_function,
        sampled_ids,
        api_key,
        base_url,
        model_name,
        step_key,
    ) = args
    return inference_and_eval_single(
        data,
        img_save_dir,
        eval_save_dir,
        data_dir,
        inference_function,
        sampled_ids,
        api_key,
        base_url,
        model_name,
        step_key,
    )


def inference_and_eval(
    img_save_dir,
    eval_save_dir,
    data_dir="./data/full",
    inference_function=None,
    sampled_id_path=None,
    max_workers=-1,
    api_key="",
    base_url=DEFAULT_BASE_URL,
    model_name=DEFAULT_MODEL_NAME,
    step_key=None,
    dedup_by_hash=False,
):
    with open(os.path.join(data_dir, "annotations", "All_Subjects.jsonl"), "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f.readlines()]

    if sampled_id_path is not None:
        with open(sampled_id_path, "r", encoding="utf-8") as f:
            sampled_ids = [x.strip() for x in f.readlines() if x.strip()]
    else:
        sampled_ids = None

    if step_key == "step_all":
        available_steps = _list_available_step_keys(all_data)
        if not available_steps:
            print("No stepN keys found in annotations.step_image_paths, nothing to evaluate for step_all.")
            return

        base_img_dir = img_save_dir
        base_eval_dir = eval_save_dir
        for sk in available_steps:
            sk_img_dir = os.path.join(base_img_dir, sk)
            sk_eval_dir = base_eval_dir
            if os.path.basename(base_eval_dir.rstrip("/")) != f"eval_results_{sk}":
                sk_eval_dir = os.path.join(os.path.dirname(base_eval_dir.rstrip("/")), f"eval_results_{sk}")

            _run_eval_once(
                all_data,
                sk_img_dir,
                sk_eval_dir,
                data_dir,
                inference_function,
                sampled_ids,
                api_key,
                base_url,
                model_name,
                sk,
                max_workers,
                dedup_by_hash,
            )
        return

    _run_eval_once(
        all_data,
        img_save_dir,
        eval_save_dir,
        data_dir,
        inference_function,
        sampled_ids,
        api_key,
        base_url,
        model_name,
        step_key,
        max_workers,
        dedup_by_hash,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--img_save_dir", type=str, default="./gen_imgs")
    parser.add_argument("--step_key", type=str, default=None, help="Target candidate step key, e.g. step1/step2/step3/final")
    parser.add_argument("--eval_save_dir", type=str, default="./eval_results")
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--sampled_id_path", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=-1)
    parser.add_argument("--dedup_by_hash", action="store_true", help="Deduplicate repeated images by file hash and reuse eval results")

    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)

    parser.add_argument("--gen_model", default=os.environ.get("DMX_IMAGE_GEN_MODEL", os.environ.get("DMX_IMAGE_MODEL", "gpt-image-1.5")))
    parser.add_argument("--edit_model", default=os.environ.get("DMX_IMAGE_EDIT_MODEL", os.environ.get("DMX_IMAGE_MODEL", "gpt-image-1.5")))
    parser.add_argument("--run_date", default=None, help="Run date tag in folder name, format YYYYMMDD. Default=today")
    parser.add_argument("--disable_auto_run_subdir", action="store_true", help="Disable auto creating model&date subfolder under img/eval save dirs")

    args = parser.parse_args()

    run_date = args.run_date or datetime.now().strftime("%Y%m%d")
    run_tag = _build_run_tag(args.gen_model, args.edit_model, run_date)
    if not args.disable_auto_run_subdir:
        # Put run tag under parent folder:
        # /root/.../Judge_output/<run_tag>/eval_results_xxx
        eval_parent = os.path.dirname(args.eval_save_dir.rstrip("/"))
        eval_leaf = os.path.basename(args.eval_save_dir.rstrip("/"))
        args.eval_save_dir = os.path.join(eval_parent, run_tag, eval_leaf)

        # generated images are grouped only when this script performs generation
        if args.run_inference:
            img_parent = os.path.dirname(args.img_save_dir.rstrip("/"))
            img_leaf = os.path.basename(args.img_save_dir.rstrip("/"))
            args.img_save_dir = os.path.join(img_parent, run_tag, img_leaf)

    if not args.api_key:
        raise ValueError("API key is required. Please set --api_key or environment variable DMX_API_KEY")

    if args.run_inference:
        inference_function = lambda text_prompt, save_path: call_img_gen(
            text_prompt=text_prompt,
            save_path=save_path,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    else:
        inference_function = None

    effective_sampled_id_path = (
        args.sampled_id_path
        if args.sampled_id_path
        else (os.path.join(args.data_dir, "mini_sample_ids.txt") if args.mini else None)
    )

    inference_and_eval(
        img_save_dir=args.img_save_dir,
        eval_save_dir=args.eval_save_dir,
        inference_function=inference_function,
        data_dir=args.data_dir,
        sampled_id_path=effective_sampled_id_path,
        max_workers=args.max_workers,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        step_key=args.step_key,
        dedup_by_hash=args.dedup_by_hash,
    )
