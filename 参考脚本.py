"""
重构后的模型推理脚本
====================

功能：
1. 读取 JSONL 数据集
2. 使用 vLLM 离线批量推理
3. 注入业务特定的 Prompt 指令
4. 验证并解析模型输出（XML 格式）
5. 保存结果到输出文件

注意：此脚本使用已合并 LoRA 权重的模型，需要先运行 merge_lora.py 合并 LoRA 权重

使用说明：
---------
# 步骤1：合并 LoRA 权重（为每个 checkpoint 指定不同的名称）
python merge_lora.py \
    --adapters /home/gary/SciGen_training/指令遵循部分LORA结果/EXP1/output/低分辨率/checkpoint-3123\
    --output_name checkpoint-3123

python merge_lora.py \
    --adapters /path/to/checkpoint-2000 \
    --output_name exp1_ckpt2000

# 步骤2：运行推理脚本（指定对应的合并模型）
python 测评脚本.py \
    --model_path /home/devdata/Dataset/SciGen-Verify/Lora-Merge-Tmp/exp1_ckpt1000 \
    --datasets_path /path/to/data.jsonl

# 处理整个目录的 jsonl 文件
python 测评脚本.py \
    --model_path /home/devdata/Dataset/SciGen-Verify/Lora-Merge-Tmp/exp1_ckpt1000 \
    --datasets_dir /path/to/datasets/ \
    --save_meta_name output.jsonl

# 调试模式（仅处理前10条）
python 测评脚本.py \
    --model_path /home/devdata/Dataset/SciGen-Verify/Lora-Merge-Tmp/exp1_ckpt1000 \
    --datasets_path input.jsonl \
    --debug --debug_sample_size 10

参数说明：
--datasets_path     : 输入 JSONL 文件路径
--datasets_dir      : 输入 JSONL 文件目录（与 datasets_path 二选一）
--save_meta_name    : 输出结果文件路径
--batch_size        : 批处理大小（默认 50）
--debug             : 启用调试模式
--debug_sample_size : 调试模式下处理的样本数量
--source_dir        : 媒体路径替换的源目录
--target_dir        : 媒体路径替换的目标目录
--max_tokens        : 最大生成 token 数
--model_path        : 已合并 LoRA 的模型路径（通过 merge_lora.py 的 --output_name 指定名称）
--max_image_pixels  : 图像最大像素数（影响图像缩放与 MAX_PIXELS）
--max_model_len     : 模型最大长度
"""

import os
import json
import sys
import re
import argparse
import time
import math
import tempfile
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

# ============ 路径配置 ============
TOOLKIT_PATH = os.environ.get("TOOLKIT_PATH", "/root/autodl-tmp/home/测评/data-process-toolkits")
SWIFT_PATH = os.environ.get("SWIFT_PATH", "/root/autodl-tmp/home/测评/vlm-train-prod")
for _path in [TOOLKIT_PATH, SWIFT_PATH]:
    if _path and os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# ============ 环境配置 ============
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from datatool.utils.data import load_message_from_data

# ============ 默认配置常量 ============
DEFAULT_DATASETS_DIR = "/root/autodl-tmp/home/测评/Benchmark_v2.3/bench_v2.3/MetaFiles"
MAX_TOKENS = 8192
MAX_IMAGE_PIXELS = 200704
MAX_MODEL_LEN = 10240
SOURCE_DIR = None
TARGET_DIR = None

DEBUG_MODE = False

NOTHINK_MODE = True

# 预处理并行 worker 数（可根据 CPU 核数调整）
PREPROCESS_WORKERS = 12
FIXED_INFER_BATCH_SIZE = 2000


# ============ 日志工具函数 ============

def log(msg, level="INFO"):
    """
    带时间戳的日志输出函数
    
    Args:
        msg: 日志消息内容
        level: 日志级别（INFO/WARN/ERROR）
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] [{level}] {msg}")


def debug_log(title, content):
    """
    调试日志输出（仅在 DEBUG_MODE 为 True 时生效）
    
    Args:
        title: 日志标题
        content: 日志内容
    """
    if not DEBUG_MODE:
        return
    print(f"\n{'='*20} DEBUG: {title} {'='*20}")
    print(content)
    print(f"{'='*50}\n")


# ============ 图像处理工具函数 ============

def _is_base64_payload(value: str) -> bool:
    """
    判断字符串是否为 Base64 编码的图像数据
    
    Args:
        value: 待检测的字符串
        
    Returns:
        bool: 如果是 Base64 图像数据返回 True
    """
    if not isinstance(value, str):
        return False
    if len(value) < 200:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", value) is not None


def _normalize_image_url(url: str) -> str:
    """
    标准化图像 URL，确保格式正确
    
    Args:
        url: 原始图像 URL 或 Base64 数据
        
    Returns:
        str: 标准化后的图像 URL
    """
    if not isinstance(url, str):
        return url
    lower = url.lower()
    if lower.startswith("data:image/") or lower.startswith("http://") or lower.startswith("https://"):
        return url
    if _is_base64_payload(url):
        return f"data:image/png;base64,{url}"
    return url


def normalize_messages_for_dmx(messages: list) -> list:
    """
    标准化消息格式，处理图像 URL
    
    Args:
        messages: 原始消息列表
        
    Returns:
        list: 标准化后的消息列表
    """
    normalized = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_url = part.get("image_url", {})
                    if isinstance(image_url, dict) and "url" in image_url:
                        updated = dict(image_url)
                        updated["url"] = _normalize_image_url(image_url.get("url"))
                        part = dict(part)
                        part["image_url"] = updated
                new_content.append(part)
            msg = dict(msg)
            msg["content"] = new_content
        normalized.append(msg)
    return normalized


def is_http_url(path):
    """
    判断路径是否为 HTTP URL
    
    Args:
        path: 文件路径或 URL
        
    Returns:
        bool: 如果是 HTTP URL 返回 True
    """
    return isinstance(path, str) and re.match(r"^https?://", path) is not None


def resize_image_if_needed(image_path):
    """
    根据需要缩放图像，确保不超过最大像素限制
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        str: 处理后的图像路径（可能是缓存路径）
    """
    if not image_path or is_http_url(image_path):
        return image_path
    if not os.path.exists(image_path):
        return image_path
    try:
        from PIL import Image
    except Exception:
        log("需要 Pillow 才能缩放图片", "ERROR")
        return image_path
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            pixels = w * h
            if pixels <= MAX_IMAGE_PIXELS:
                return image_path
            scale = math.sqrt(MAX_IMAGE_PIXELS / pixels)
            new_w = max(int(w * scale), 1)
            new_h = max(int(h * scale), 1)
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            img = img.convert("RGB").resize((new_w, new_h), resample)
            cache_dir = os.path.join(tempfile.gettempdir(), "qwen_resize_cache")
            os.makedirs(cache_dir, exist_ok=True)
            try:
                mtime = os.path.getmtime(image_path)
            except Exception:
                mtime = 0
            key = f"{image_path}|{mtime}|{MAX_IMAGE_PIXELS}"
            filename = hashlib.md5(key.encode("utf-8")).hexdigest() + ".jpg"
            out_path = os.path.join(cache_dir, filename)
            if not os.path.exists(out_path):
                # 原子写入：先写临时文件再 rename，避免多线程竞争写坏文件
                tmp_path = out_path + f".tmp.{threading.get_ident()}"
                img.save(tmp_path, format="JPEG", quality=92, optimize=True)
                os.replace(tmp_path, out_path)
            return out_path
    except Exception as e:
        log(f"图片缩放失败: {image_path} - {e}", "WARN")
        return image_path


# ============ 业务逻辑函数 ============

def extract_tag_content(text, tag):
    """
    从文本中提取指定 XML 标签的内容
    
    Args:
        text: 包含标签的文本
        tag: 标签名称（如 'answer', 'explanation'）
        
    Returns:
        str or None: 提取的内容，如果未找到则返回 None
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def validate_and_parse_response(result, ground_truth):
    """
    验证并解析模型响应，确保包含必要的 XML 标签且逻辑一致
    
    Args:
        result: 模型返回的原始响应字符串
        ground_truth: 数据的真实标签（true/false）
        
    Returns:
        dict or None: 如果验证通过返回包含解析结果的字典，否则返回 None
    """
    response = re.sub(r'<\|user\|>\s*$', '', result).strip()
    box_content = response
    
    answer = extract_tag_content(box_content, 'answer')
    if not answer:
        log("验证失败: <answer> 标签缺失或为空", "WARN")
        return None
    
    answer_lower = answer.lower()
    if answer_lower not in ['true', 'false']:
        log(f"验证失败: <answer> 内容必须是 true/false，当前为: {answer}", "WARN")
        return None
    
    explanation = extract_tag_content(box_content, 'explanation')
    if not explanation:
        log("验证失败: <explanation> 标签缺失或为空", "WARN")
        return None
    
    edit = extract_tag_content(box_content, 'edit')
    if answer_lower == 'false' and not edit:
        log("验证失败: answer 为 false 时，<edit> 标签必须存在且不能为空", "WARN")
        return None
    if answer_lower == 'true':
        edit = None
    
    is_true = (answer_lower == ground_truth.lower())
    
    return {
        'response': response,
        'is_true': is_true,
        'answer': answer_lower,
        'explanation': explanation,
        'edit': edit if edit else None
    }


def prepare_item(item):
    """
    处理单条数据：
    1. 准备输入 (User Prompt + Image)
    2. 组装推理消息
    3. 返回待推理数据
    
    Args:
        item: 数据条目
        
    Returns:
        dict or None: 处理后的数据字典，失败返回 None
    """
    try:
        item_id = item.get('id', 'unknown')
        item_copy = deepcopy(item)
        item_messages = item_copy.get("messages", [])
        
        user_found = False
        for msg in item_messages:
            if msg.get("role") == "user":
                user_found = True
        if not user_found:
            log(f"未在 item {item_id} 中找到 user message", "WARN")
            return None
        
        # 处理图像路径
        if item_copy.get("images"):
            item_copy["images"] = [resize_image_if_needed(p) for p in item_copy["images"]]
        
        # 加载消息并转换为 API 格式
        api_messages, _ = load_message_from_data(
            item_copy,
            load_media=True,
            source_dir=SOURCE_DIR,
            target_dir=TARGET_DIR,
            media_to_base64=True
        )
        api_messages = normalize_messages_for_dmx(api_messages)

        if NOTHINK_MODE:
            for msg in api_messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                part["text"] = part.get("text", "") + "/nothink"
                                break
                        else:
                            content.append({"type": "text", "text": "/nothink"})
                    elif isinstance(content, str):
                        msg["content"] = content + "/nothink"
                    break

        # 获取真实答案 (Ground Truth)
        ground_truth = None
        if len(item_messages) >= 2 and item_messages[1].get('role') == 'assistant':
            ground_truth = item_messages[1].get('content', '').strip()
        
        if not ground_truth or ground_truth.lower() not in ['true', 'false']:
            log(f"警告: item {item_id} 的 ground_truth 无效: {ground_truth}", "WARN")
            return None
        
        return {
            "id": item_id,
            "api_messages": api_messages,
            "ground_truth": ground_truth
        }
    
    except Exception as e:
        log(f"处理 item {item.get('id', 'unknown')} 时出错: {e}", "ERROR")
        return None


def run_batch_infer(engine, request_config, batch_items):
    """
    执行批量推理
    
    Args:
        engine: vLLM 引擎实例
        request_config: 请求配置
        batch_items: 批量数据项列表
        
    Returns:
        list: 推理结果列表（包含格式验证通过和未通过的结果）
    """
    from swift.llm import InferRequest

    def build_format_failed_output(item, full_response, finish_reason):
        return {
            "id": item["id"],
            "longcot_by_qwen3_vl_32b_thinking": {
                "response": full_response,
                "is_true": False,
                "answer": None,
                "explanation": None,
                "edit": None,
                "finish_reason": finish_reason,
                "format_valid": False
            }
        }

    def parse_infer_response(item, response):
        if response is None or not response.choices:
            return {
                "status": "infer_failed",
                "output_item": None
            }

        choice = response.choices[0]
        full_response = choice.message.content or ""
        finish_reason = choice.finish_reason
        if DEBUG_MODE:
            debug_log(f"Model Output (Item {item['id']})", full_response)

        parsed_result = validate_and_parse_response(full_response, item["ground_truth"])
        if parsed_result is None:
            return {
                "status": "format_failed",
                "raw_response": full_response,
                "finish_reason": finish_reason
            }

        parsed_result["finish_reason"] = finish_reason
        parsed_result["format_valid"] = True
        output_item = {
            "id": item["id"],
            "longcot_by_qwen3_vl_32b_thinking": parsed_result
        }
        return {
            "status": "ok",
            "output_item": output_item
        }

    infer_requests = [InferRequest(messages=item["api_messages"]) for item in batch_items]
    responses = engine.infer(infer_requests, request_config=request_config)
    results = [None] * len(batch_items)
    retry_indices = []
    retry_items = []

    for idx, (item, response) in enumerate(zip(batch_items, responses)):
        parsed = parse_infer_response(item, response)
        if parsed["status"] == "ok":
            results[idx] = parsed["output_item"]
        elif parsed["status"] == "format_failed":
            retry_indices.append(idx)
            retry_items.append(item)

    if retry_items:
        log(f"格式校验失败，开始重试 {len(retry_items)} 条数据...")
        retry_requests = [InferRequest(messages=item["api_messages"]) for item in retry_items]
        retry_responses = engine.infer(retry_requests, request_config=request_config)
        for idx, item, response in zip(retry_indices, retry_items, retry_responses):
            parsed = parse_infer_response(item, response)
            if parsed["status"] == "ok":
                results[idx] = parsed["output_item"]
            elif parsed["status"] == "format_failed":
                log(f"格式验证重试失败，保存原始输出 item {item['id']}", "WARN")
                print(f"\n{'='*20} MODEL OUTPUT (FAILED ITEM {item['id']}) {'='*20}")
                print(parsed["raw_response"])
                print(f"{'='*60}\n")
                results[idx] = build_format_failed_output(item, parsed["raw_response"], parsed["finish_reason"])

    return results


def prepare_batch_parallel(batch):
    """
    使用线程池并行预处理一个 batch
    
    Args:
        batch: 原始数据条目列表
        
    Returns:
        tuple: (prepared_items, fail_count)
    """
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as pool:
        results = list(pool.map(prepare_item, batch))
    prepared = [r for r in results if r is not None]
    failed = len(batch) - len(prepared)
    return prepared, failed


def load_pending_items(dataset_file, output_file, debug_sample_size):
    existing_ids = set()
    if os.path.exists(output_file):
        log(f"检测到输出文件，正在加载已处理 id: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        item_id = item.get('id')
                        if item_id is not None:
                            existing_ids.add(item_id)
                    except Exception:
                        continue
        except Exception as e:
            log(f"读取输出文件失败: {e}", "ERROR")
            raise

    items = []
    skipped_count = 0
    log(f"正在读取数据: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = item.get("id")
            if item_id in existing_ids:
                skipped_count += 1
                continue
            items.append(item)

    if existing_ids:
        log(f"过滤已处理数据: {skipped_count}")
    if debug_sample_size > 0:
        items = items[:debug_sample_size]
        log(f"调试模式，仅处理前 {len(items)} 条数据")
    return items


def collect_all_infer_entries(dataset_files, output_file_map, debug_sample_size):
    entries = []
    for dataset_file in dataset_files:
        output_file = output_file_map[dataset_file]
        dataset_name = Path(dataset_file).name
        items = load_pending_items(dataset_file, output_file, debug_sample_size)
        if len(items) == 0:
            log(f"{dataset_name} 无可推理数据，跳过。")
            continue
        log(f"{dataset_name} 待推理条数: {len(items)}")
        for item in items:
            entries.append({
                "dataset_name": dataset_name,
                "output_file": output_file,
                "item": item
            })
    return entries


def prepare_infer_entries_parallel(entries):
    if not entries:
        return [], 0

    def _prepare_from_entry(entry):
        return prepare_item(entry["item"])

    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as pool:
        prepared_results = list(pool.map(_prepare_from_entry, entries))

    prepared_entries = []
    fail_count = 0
    for entry, prepared in zip(entries, prepared_results):
        if prepared is None:
            fail_count += 1
            continue
        prepared_entries.append({
            "dataset_name": entry["dataset_name"],
            "output_file": entry["output_file"],
            "prepared_item": prepared
        })
    return prepared_entries, fail_count


def collect_dataset_files(datasets_path_args, datasets_dir):
    dataset_files = []
    raw_paths = []
    for value in datasets_path_args:
        for path_piece in value.split(","):
            path_piece = path_piece.strip()
            if path_piece:
                raw_paths.append(path_piece)

    if raw_paths:
        for input_path in raw_paths:
            if not os.path.exists(input_path):
                log(f"输入路径不存在: {input_path}", "ERROR")
                exit(1)
            if os.path.isdir(input_path):
                directory_files = [str(p) for p in Path(input_path).rglob("*.jsonl")]
                if not directory_files:
                    log(f"目录中未找到 jsonl 文件: {input_path}", "ERROR")
                    exit(1)
                dataset_files.extend(directory_files)
            else:
                dataset_files.append(input_path)
    else:
        datasets_dir = datasets_dir.strip() or DEFAULT_DATASETS_DIR
        if not os.path.exists(datasets_dir):
            log(f"输入目录不存在: {datasets_dir}", "ERROR")
            exit(1)
        dataset_files = [str(p) for p in Path(datasets_dir).rglob("*.jsonl")]
        if not dataset_files:
            log(f"目录中未找到 jsonl 文件: {datasets_dir}", "ERROR")
            exit(1)

    dataset_files = list(dict.fromkeys(dataset_files))
    dataset_files.sort()
    return dataset_files


def main():
    """
    主函数：解析参数、加载数据、执行推理、保存结果
    """
    global DEBUG_MODE, SOURCE_DIR, TARGET_DIR, MAX_TOKENS, MAX_IMAGE_PIXELS, MAX_MODEL_LEN
    
    parser = argparse.ArgumentParser(description="Batch inference with vLLM offline engine")
    parser.add_argument("--datasets_path", type=str, nargs="*", default=[],
                        help="Path(s) to input JSONL file(s). Supports multiple values or comma-separated values.")
    parser.add_argument("--datasets_dir", type=str, default=DEFAULT_DATASETS_DIR,
                        help="Path to the input JSONL directory.")
    parser.add_argument("--save_meta_name", type=str, default="output_results.jsonl",
                        help="Path to save the output JSONL file.")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for vLLM inference.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode to print detailed logs.")
    parser.add_argument("--source_dir", type=str, default="",
                        help="Optional source_dir for media path replacement.")
    parser.add_argument("--target_dir", type=str, default="",
                        help="Optional target_dir for media path replacement.")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                        help="Max tokens for completion.")
    parser.add_argument("--max_image_pixels", type=int, default=MAX_IMAGE_PIXELS,
                        help="Max image pixels for resizing and MAX_PIXELS.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Merged model path for vLLM (LoRA already merged).")
    parser.add_argument("--max_model_len", type=int, default=MAX_MODEL_LEN,
                        help="Max model length for vLLM.")
    parser.add_argument("--debug_sample_size", type=int, default=0,
                        help="If >0, only process first N items for debugging.")
    
    args = parser.parse_args()
    
    # 设置全局变量
    DEBUG_MODE = args.debug
    SOURCE_DIR = args.source_dir.strip() or None
    TARGET_DIR = args.target_dir.strip() or None
    MAX_TOKENS = args.max_tokens
    MAX_IMAGE_PIXELS = args.max_image_pixels
    MAX_MODEL_LEN = args.max_model_len
    debug_sample_size = max(args.debug_sample_size, 0)
    
    # 验证模型路径
    if not os.path.exists(args.model_path):
        log(f"模型路径不存在: {args.model_path}", "ERROR")
        exit(1)
    
    dataset_files = collect_dataset_files(args.datasets_path, args.datasets_dir)

    ckpt_name = Path(args.model_path.rstrip("/")).name or "model"
    save_meta_path = Path(args.save_meta_name)
    output_file_map = {}
    if len(dataset_files) > 1:
        if save_meta_path.suffix.lower() == ".jsonl":
            base_output_dir = save_meta_path.parent if str(save_meta_path.parent) else Path(".")
        else:
            base_output_dir = save_meta_path
        ckpt_output_dir = base_output_dir / ckpt_name
        os.makedirs(ckpt_output_dir, exist_ok=True)
        for dataset_file in dataset_files:
            output_file_map[dataset_file] = str(ckpt_output_dir / Path(dataset_file).name)
    else:
        only_file = dataset_files[0]
        if save_meta_path.suffix.lower() == ".jsonl":
            output_file_map[only_file] = args.save_meta_name
        else:
            ckpt_output_dir = save_meta_path / ckpt_name
            os.makedirs(ckpt_output_dir, exist_ok=True)
            output_file_map[only_file] = str(ckpt_output_dir / Path(only_file).name)

    success_count = 0
    fail_count = 0
    format_fail_count = 0
    engine = None
    request_config = None
    output_handles = {}

    try:
        from swift.llm import VllmEngine, RequestConfig
        os.environ.setdefault("MAX_PIXELS", str(MAX_IMAGE_PIXELS))
        os.environ.setdefault("VLLM_MAX_NUM_SEQS", "64")
        os.environ.setdefault("VLLM_MAX_NUM_BATCHED_TOKENS", "16384")

        for dataset_file in dataset_files:
            output_file = output_file_map[dataset_file]
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

        if args.batch_size != 1000:
            log(f"已忽略 --batch_size 参数：当前策略固定为每批 {FIXED_INFER_BATCH_SIZE} 条。")

        all_entries = collect_all_infer_entries(dataset_files, output_file_map, debug_sample_size)
        if not all_entries:
            print(f"\n处理完成! 成功: {success_count}, 格式验证失败: {format_fail_count}, 推理失败: {fail_count}")
            log("保存成功!")
            return

        prepared_all_result = {}

        def preprocess_all_entries():
            prepared, failed = prepare_infer_entries_parallel(all_entries)
            prepared_all_result["prepared"] = prepared
            prepared_all_result["failed"] = failed

        preprocess_thread = threading.Thread(target=preprocess_all_entries)
        preprocess_thread.start()

        engine = VllmEngine(
            args.model_path,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.96
        )
        request_config = RequestConfig(
            max_tokens=MAX_TOKENS,
            temperature=0.95,
            top_p=0.95,
            top_k=40
        )

        def get_output_handle(output_file):
            if output_file not in output_handles:
                log(f"正在保存结果到: {output_file}")
                write_mode = 'a' if os.path.exists(output_file) else 'w'
                output_handles[output_file] = open(output_file, write_mode, encoding='utf-8')
            return output_handles[output_file]

        preprocess_thread.join()
        fail_count += prepared_all_result.get("failed", 0)
        prepared_all_entries = prepared_all_result.get("prepared", [])

        if prepared_all_entries:
            total = len(prepared_all_entries)
            total_batches = (total + FIXED_INFER_BATCH_SIZE - 1) // FIXED_INFER_BATCH_SIZE
            for batch_idx in range(total_batches):
                start = batch_idx * FIXED_INFER_BATCH_SIZE
                end = min(start + FIXED_INFER_BATCH_SIZE, total)
                current_batch = prepared_all_entries[start:end]
                log(f"[Global Batch {batch_idx + 1}/{total_batches}] 推理 {len(current_batch)} 条...")
                batch_results = run_batch_infer(
                    engine,
                    request_config,
                    [entry["prepared_item"] for entry in current_batch]
                )
                for entry, result_item in zip(current_batch, batch_results):
                    if result_item:
                        handle = get_output_handle(entry["output_file"])
                        handle.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                        handle.flush()
                        if result_item.get("longcot_by_qwen3_vl_32b_thinking", {}).get("format_valid", False):
                            success_count += 1
                        else:
                            format_fail_count += 1
                    else:
                        fail_count += 1

        print(f"\n处理完成! 成功: {success_count}, 格式验证失败: {format_fail_count}, 推理失败: {fail_count}")
        log("保存成功!")
    except Exception as e:
        log(f"保存结果失败: {e}", "ERROR")
    finally:
        for handle in output_handles.values():
            handle.close()


if __name__ == "__main__":
    main()
