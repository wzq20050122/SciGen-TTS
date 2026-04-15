#!/usr/bin/env python3
"""DMX API image generation/edit runner.

Supports two endpoint families:
1) OpenAI-style images endpoints (/images/generations, /images/edits)
2) DMX Responses endpoint (/responses) used by wan/qwen/... model families
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = os.environ.get("DMX_BASE_URL", "https://www.dmxapi.cn/v1")
# Backward-compatible fallback order:
# 1) Dedicated env vars: DMX_IMAGE_GEN_MODEL / DMX_IMAGE_EDIT_MODEL
# 2) Legacy shared env var: DMX_IMAGE_MODEL
# 3) hard-coded default
DEFAULT_GEN_MODEL = os.environ.get("DMX_IMAGE_GEN_MODEL", os.environ.get("DMX_IMAGE_MODEL", "gpt-image-1.5"))
DEFAULT_EDIT_MODEL = os.environ.get("DMX_IMAGE_EDIT_MODEL", os.environ.get("DMX_IMAGE_MODEL", "gpt-image-1.5"))

# OpenAI image-style endpoints (works for gpt-image family)
DEFAULT_EDIT_URL = f"{DEFAULT_BASE_URL.rstrip('/')}/images/edits"
DEFAULT_GEN_URL = f"{DEFAULT_BASE_URL.rstrip('/')}/images/generations"
DEFAULT_RESPONSES_URL = f"{DEFAULT_BASE_URL.rstrip('/')}/responses"

# Allow endpoint override for non-OpenAI image families (e.g., wan/qwen/doubao).
DEFAULT_GEN_URL = os.environ.get("DMX_IMAGE_GEN_URL", DEFAULT_GEN_URL)
DEFAULT_EDIT_URL = os.environ.get("DMX_IMAGE_EDIT_URL", DEFAULT_EDIT_URL)
DEFAULT_GEN_PROMPT_EXTEND = os.environ.get("DMX_GEN_PROMPT_EXTEND", "true")
DEFAULT_EDIT_PROMPT_EXTEND = os.environ.get("DMX_EDIT_PROMPT_EXTEND", "false")

QWEN_IMAGE_MAX_ALLOWED_SIZES = (
    (1664, 928),
    (1472, 1104),
    (1328, 1328),
    (1104, 1472),
    (928, 1664),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help="Model used for image generations")
    p.add_argument("--edit-model", default=DEFAULT_EDIT_MODEL, help="Model used for image edits")
    return p.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_headers(*, json_request: bool = False) -> dict:
    api_key = os.environ.get("DMX_API_KEY")
    if not api_key:
        raise ValueError("DMX_API_KEY is not set")
    headers = {"Authorization": api_key}
    if json_request:
        headers["Content-Type"] = "application/json"
    return headers


def build_prompt(payload: dict) -> str:
    return (payload.get("current_instruction") or payload.get("instruction") or "").strip()


def env_flag(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def mime_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def normalize_size(size: str) -> str:
    # Some endpoints expect 1024*1024 instead of 1024x1024.
    if isinstance(size, str) and "x" in size:
        return size.replace("x", "*")
    return size


def endpoint_uses_responses(url: str) -> bool:
    return url.rstrip("/").endswith("/responses")


def model_uses_responses(model_name: str) -> bool:
    model = (model_name or "").strip().lower()
    if not model:
        return False
    responses_prefixes = (
        "wan",
        "qwen-image-max",
        "qwen-image-plus",
        "qwen-image-edit-max",
        "qwen-image-edit-plus",
    )
    return model.startswith(responses_prefixes)


def resolve_endpoint_url(explicit_url: str, *, model_name: str, fallback_url: str) -> str:
    if explicit_url:
        return explicit_url
    if model_uses_responses(model_name):
        return DEFAULT_RESPONSES_URL
    return fallback_url


def parse_size(size: str) -> tuple[int, int] | None:
    if not isinstance(size, str):
        return None
    match = re.fullmatch(r"\s*(\d+)\s*[x*]\s*(\d+)\s*", size)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def normalize_model_size(size: str, *, model_name: str) -> str:
    model = (model_name or "").strip().lower()
    if model == "qwen-image-max":
        parsed = parse_size(size)
        if parsed is None:
            return "1664*928"
        target_w, target_h = parsed
        target_ratio = target_w / max(target_h, 1)
        best_w, best_h = min(
            QWEN_IMAGE_MAX_ALLOWED_SIZES,
            key=lambda wh: abs((wh[0] / wh[1]) - target_ratio),
        )
        return f"{best_w}*{best_h}"
    return normalize_size(size)


def _to_data_url(image_path: Path) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type_for(image_path)};base64,{image_b64}"


def _responses_raise_with_details(response: requests.Response, model_name: str, url: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        details = ""
        try:
            details = response.text
        except Exception:
            pass
        raise requests.HTTPError(
            f"{e}. model={model_name}, url={url}, response={details}",
            response=response,
        )


def _json_post(url: str, payload: dict, model_name: str) -> dict:
    response = requests.post(url, headers=get_headers(json_request=True), json=payload, timeout=300)
    _responses_raise_with_details(response, model_name, url)
    return response.json()


def call_generate(prompt: str, size: str, model_name: str) -> dict:
    url = resolve_endpoint_url(
        os.environ.get("DMX_IMAGE_GEN_URL", ""),
        model_name=model_name,
        fallback_url=f"{DEFAULT_BASE_URL.rstrip('/')}/images/generations",
    )
    if endpoint_uses_responses(url):
        payload = {
            "model": model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]
            },
            "parameters": {
                "prompt_extend": env_flag(DEFAULT_GEN_PROMPT_EXTEND),
                "watermark": False,
                "n": 1,
                "negative_prompt": "",
                "size": normalize_model_size(size, model_name=model_name),
                "stream": False,
            },
        }
        return _json_post(url, payload, model_name)

    payload = {
        "prompt": prompt,
        "n": 1,
        "model": model_name,
        "size": size,
        "background": "auto",
        "moderation": "auto",
        "output_format": "png",
        "quality": "auto",
    }
    return _json_post(url, payload, model_name)


def call_edit(previous_image: Path, prompt: str, size: str, model_name: str) -> dict:
    url = resolve_endpoint_url(
        os.environ.get("DMX_IMAGE_EDIT_URL", ""),
        model_name=model_name,
        fallback_url=f"{DEFAULT_BASE_URL.rstrip('/')}/images/edits",
    )
    if endpoint_uses_responses(url):
        payload = {
            "model": model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt},
                            {"image": _to_data_url(previous_image)},
                        ],
                    }
                ]
            },
            "parameters": {
                "prompt_extend": env_flag(DEFAULT_EDIT_PROMPT_EXTEND),
                "watermark": False,
                "n": 1,
                "size": normalize_model_size(size, model_name=model_name),
                "stream": False,
                "negative_prompt": "",
            },
        }
        return _json_post(url, payload, model_name)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "size": size,
        "background": "auto",
        "quality": "auto",
        "output_format": "png",
        "output_compression": 100,
        "input_fidelity": "high",
    }
    with previous_image.open("rb") as image_file:
        files = [
            (
                "image",
                (previous_image.name, image_file, mime_type_for(previous_image)),
            )
        ]
        response = requests.post(
            url,
            headers=get_headers(),
            data=payload,
            files=files,
            timeout=300,
        )
    _responses_raise_with_details(response, model_name, url)
    return response.json()


def _looks_like_base64(s: str) -> bool:
    # Heuristic gate: avoid treating normal URLs/text as base64.
    if not s or len(s) < 64:
        return False
    if s.startswith(("http://", "https://")):
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")
    return all(ch in allowed for ch in s)


def _try_decode_data_or_b64(value: str) -> bytes | None:
    if not value:
        return None

    if value.startswith("data:") and ";base64," in value:
        b64 = value.split(";base64,", 1)[1]
        try:
            return base64.b64decode(b64, validate=True)
        except Exception:
            return None

    if not _looks_like_base64(value):
        return None

    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        return None


def extract_image_bytes(response_json: dict[str, Any]) -> bytes:
    # 1) OpenAI images response
    data = response_json.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            b64 = first.get("b64_json")
            decoded = _try_decode_data_or_b64(b64) if isinstance(b64, str) else None
            if decoded:
                return decoded
            url = first.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return requests.get(url, timeout=300).content

    # 2) Responses-style output (best-effort for multiple possible shapes)
    output = response_json.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            for key in ("result", "image", "b64_json", "url"):
                val = item.get(key)
                if isinstance(val, str):
                    decoded = _try_decode_data_or_b64(val)
                    if decoded:
                        return decoded
                    if val.startswith(("http://", "https://")):
                        return requests.get(val, timeout=300).content

            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    for key in ("image", "image_url", "b64_json", "url", "text"):
                        val = c.get(key)
                        if isinstance(val, str):
                            decoded = _try_decode_data_or_b64(val)
                            if decoded:
                                return decoded
                            if val.startswith(("http://", "https://")):
                                return requests.get(val, timeout=300).content

    raise RuntimeError(f"No image found in response: {response_json}")


def save_image_from_response(response_json: dict, output_path: Path) -> None:
    raw = extract_image_bytes(response_json)
    # Basic signature validation to avoid writing non-image bytes.
    if not (
        raw.startswith(b"\x89PNG\r\n\x1a\n")
        or raw.startswith(b"\xff\xd8\xff")
        or raw.startswith(b"RIFF")  # WebP in RIFF container
    ):
        raise RuntimeError(f"Downloaded content is not a valid image header: {raw[:32]!r}")
    output_path.write_bytes(raw)


def main() -> None:
    args = parse_args()
    payload = load_payload(args.payload)
    prompt = build_prompt(payload)
    size = payload.get("target_size", "auto")

    previous_image = payload.get("previous_image")
    if previous_image and Path(previous_image).exists():
        response_json = call_edit(Path(previous_image), prompt, size, args.edit_model)
    else:
        response_json = call_generate(prompt, size, args.gen_model)

    save_image_from_response(response_json, args.output)


if __name__ == "__main__":
    main()
