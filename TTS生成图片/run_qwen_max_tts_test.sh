#!/usr/bin/env bash
set -euo pipefail

: "${DMX_API_KEY:?DMX_API_KEY is not set}"

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
LIMIT="${LIMIT:-1}"
WORKERS="${WORKERS:-1}"
MAX_STEPS="${MAX_STEPS:-3}"
RETRY_COUNT="${RETRY_COUNT:-2}"
RETRY_WAIT="${RETRY_WAIT:-30}"

export DMX_BASE_URL="${DMX_BASE_URL:-https://www.dmxapi.cn/v1}"

if [ -z "${IMAGE_EDITOR_CMD_TEMPLATE:-}" ]; then
  export IMAGE_EDITOR_CMD_TEMPLATE='python3 "/root/autodl-tmp/TTS/TTS生成图片/example_image_editor_runner.py" --payload {payload_json} --output {output_path}'
fi

python3 "/root/autodl-tmp/TTS/TTS生成图片/run_tts_image_framework.py" \
  --gen-model "qwen-image-max" \
  --edit-model "qwen-image-edit-max-2026-01-16" \
  --run-date "${RUN_DATE}" \
  --limit "${LIMIT}" \
  --workers "${WORKERS}" \
  --max-steps "${MAX_STEPS}" \
  --retry-count "${RETRY_COUNT}" \
  --retry-wait "${RETRY_WAIT}" \
  "$@"
