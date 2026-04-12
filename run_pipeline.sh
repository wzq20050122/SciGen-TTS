#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline for:
# 1) TTS generation/editing
# 2) Prepare Judge_Dataset
# 3) Judge eval (diff-step1 / diff-final / same-once)
# 4) Build shared eval dir
# 5) Score subset + merged final weighted score

# -----------------------------
# User-configurable variables
# -----------------------------
DMX_IMAGE_GEN_MODEL="${DMX_IMAGE_GEN_MODEL:-wan2.6-t2i}"
DMX_IMAGE_EDIT_MODEL="${DMX_IMAGE_EDIT_MODEL:-wan2.6-image}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
MAX_WORKERS="${MAX_WORKERS:-4}"

# IMPORTANT: set your key before running, e.g.
# export DMX_API_KEY='your_key'
: "${DMX_API_KEY:?DMX_API_KEY is not set}"

# Optional env defaults
export DMX_BASE_URL="${DMX_BASE_URL:-https://www.dmxapi.cn/v1}"
export TOOLKIT_PATH="${TOOLKIT_PATH:-/root/autodl-tmp/wzq/data-process-toolkits}"
export SWIFT_PATH="${SWIFT_PATH:-/root/autodl-tmp/wzq/vlm-train-prod}"
export VLLM_VERIFIER_MODEL="${VLLM_VERIFIER_MODEL:-/root/autodl-tmp/wzq/model/SciGen-Verifier-SFT}"
export VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-13000}"
export VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-1.0}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"

if [ -z "${IMAGE_EDITOR_CMD_TEMPLATE:-}" ]; then
  export IMAGE_EDITOR_CMD_TEMPLATE='python3 "/root/autodl-tmp/TTS/TTS生成图片/example_image_editor_runner.py" --payload {payload_json} --output {output_path}'
fi

# Avoid libgomp warning when env var is invalid/empty.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  export OMP_NUM_THREADS=1
fi

export DMX_IMAGE_GEN_MODEL
export DMX_IMAGE_EDIT_MODEL

RUN_TAG="${DMX_IMAGE_GEN_MODEL}&${DMX_IMAGE_EDIT_MODEL}_${RUN_DATE}"
# run_tts_image_framework.py 会对目录名做 sanitize（例如把 . 和 & 变成 _）
SAFE_RUN_TAG="$(python3 - <<'PY'
import re, os
raw = f"{os.environ.get('DMX_IMAGE_GEN_MODEL','')}&{os.environ.get('DMX_IMAGE_EDIT_MODEL','')}_{os.environ.get('RUN_DATE','')}"
safe = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "_", raw)
safe = re.sub(r"_+", "_", safe).strip("_") or "run"
print(safe)
PY
)"

# Paths
TTS_RUNNER="/root/autodl-tmp/TTS/TTS生成图片/run_tts_image_framework.py"
PREPARE_DATASET="/root/autodl-tmp/TTS/prepare_judge_dataset.py"
JUDGE_RUN="/root/autodl-tmp/TTS/Judge/run_eval.py"
BUILD_SHARED="/root/autodl-tmp/TTS/Judge/build_shared_eval_results.py"
CAL_SCORE="/root/autodl-tmp/TTS/Judge/cal_score.py"
MERGE_SCORES="/root/autodl-tmp/TTS/Judge/merge_scores.py"

OUTPUT_TTS_ROOT="/root/autodl-tmp/TTS/output_TTS"
JUDGE_DATASET_ROOT="/root/autodl-tmp/TTS/Judge_Dataset"
JUDGE_OUTPUT_ROOT="/root/autodl-tmp/TTS/Judge_output"

TTS_RUN_DIR="${OUTPUT_TTS_ROOT}/${SAFE_RUN_TAG}"
JUDGE_DATASET_DIR="${JUDGE_DATASET_ROOT}/${SAFE_RUN_TAG}"
JUDGE_RUN_DIR="${JUDGE_OUTPUT_ROOT}/${SAFE_RUN_TAG}"

mkdir -p "${JUDGE_RUN_DIR}"

echo "[INFO] RUN_TAG(raw)=${RUN_TAG}"
echo "[INFO] RUN_TAG(safe)=${SAFE_RUN_TAG}"
echo "[INFO] TTS_RUN_DIR=${TTS_RUN_DIR}"
echo "[INFO] JUDGE_DATASET_DIR=${JUDGE_DATASET_DIR}"
echo "[INFO] JUDGE_RUN_DIR=${JUDGE_RUN_DIR}"

# -----------------------------
# 1) TTS
# -----------------------------
echo "[STEP 1/6] Running TTS framework..."
python3 "${TTS_RUNNER}" \
  --output-dir "${OUTPUT_TTS_ROOT}" \
  --gen-model "${DMX_IMAGE_GEN_MODEL}" \
  --edit-model "${DMX_IMAGE_EDIT_MODEL}" \
  --run-date "${RUN_DATE}" \
  --workers 4 \
  --retry-count 2 \
  --retry-wait 5 \
  --skip-existing

# -----------------------------
# 2) Prepare Judge_Dataset
# -----------------------------
echo "[STEP 2/6] Preparing Judge_Dataset..."
python "${PREPARE_DATASET}" \
  --input-dir "${TTS_RUN_DIR}" \
  --output-dir "${JUDGE_DATASET_ROOT}" \
  --gen-model "${DMX_IMAGE_GEN_MODEL}" \
  --edit-model "${DMX_IMAGE_EDIT_MODEL}" \
  --run-date "${RUN_DATE}" \
  --clean

# -----------------------------
# 3) Judge evals
# -----------------------------
echo "[STEP 3/6] Running judge eval for different subset (step1)..."
python "${JUDGE_RUN}" \
  --data_dir "${JUDGE_DATASET_DIR}" \
  --max_workers "${MAX_WORKERS}" \
  --img_save_dir "${JUDGE_DATASET_DIR}/candidates/step1" \
  --eval_save_dir "${JUDGE_OUTPUT_ROOT}/eval_results_step1" \
  --sampled_id_path "${JUDGE_DATASET_DIR}/meta/ids_step1_final_different.txt" \
  --gen_model "${DMX_IMAGE_GEN_MODEL}" \
  --edit_model "${DMX_IMAGE_EDIT_MODEL}" \
  --run_date "${RUN_DATE}"

echo "[STEP 3/6] Running judge eval for different subset (final)..."
python "${JUDGE_RUN}" \
  --data_dir "${JUDGE_DATASET_DIR}" \
  --max_workers "${MAX_WORKERS}" \
  --img_save_dir "${JUDGE_DATASET_DIR}/candidates/final" \
  --eval_save_dir "${JUDGE_OUTPUT_ROOT}/eval_results_final" \
  --sampled_id_path "${JUDGE_DATASET_DIR}/meta/ids_step1_final_different.txt" \
  --gen_model "${DMX_IMAGE_GEN_MODEL}" \
  --edit_model "${DMX_IMAGE_EDIT_MODEL}" \
  --run_date "${RUN_DATE}"

echo "[STEP 3/6] Running judge eval for same subset (judge once)..."
python "${JUDGE_RUN}" \
  --data_dir "${JUDGE_DATASET_DIR}" \
  --max_workers "${MAX_WORKERS}" \
  --img_save_dir "${JUDGE_DATASET_DIR}/candidates/step1" \
  --eval_save_dir "${JUDGE_OUTPUT_ROOT}/eval_results_same_source" \
  --sampled_id_path "${JUDGE_DATASET_DIR}/meta/ids_step1_final_same.txt" \
  --gen_model "${DMX_IMAGE_GEN_MODEL}" \
  --edit_model "${DMX_IMAGE_EDIT_MODEL}" \
  --run_date "${RUN_DATE}"

# -----------------------------
# 4) Build shared dir
# -----------------------------
echo "[STEP 4/6] Building shared eval directory..."
python "${BUILD_SHARED}" \
  --same_ids "${JUDGE_DATASET_DIR}/meta/ids_step1_final_same.txt" \
  --source_eval_dir "${JUDGE_RUN_DIR}/eval_results_same_source" \
  --shared_eval_dir "${JUDGE_RUN_DIR}/eval_results_shared"

# -----------------------------
# 5) Subset scoring
# -----------------------------
echo "[STEP 5/6] Calculating subset scores..."
python "${CAL_SCORE}" \
  --eval_results_dir "${JUDGE_RUN_DIR}/eval_results_step1" \
  --sampled_id_path "${JUDGE_DATASET_DIR}/meta/ids_step1_final_different.txt" \
  > "${JUDGE_RUN_DIR}/score_step1_subset.txt"

python "${CAL_SCORE}" \
  --eval_results_dir "${JUDGE_RUN_DIR}/eval_results_final" \
  --sampled_id_path "${JUDGE_DATASET_DIR}/meta/ids_step1_final_different.txt" \
  > "${JUDGE_RUN_DIR}/score_final_subset.txt"

# -----------------------------
# 6) Merge final weighted score
# -----------------------------
echo "[STEP 6/6] Merging final weighted scores..."
python "${MERGE_SCORES}" \
  --same_ids "${JUDGE_DATASET_DIR}/meta/ids_step1_final_same.txt" \
  --different_ids "${JUDGE_DATASET_DIR}/meta/ids_step1_final_different.txt" \
  --eval_shared_dir "${JUDGE_RUN_DIR}/eval_results_shared" \
  --eval_step1_diff_dir "${JUDGE_RUN_DIR}/eval_results_step1" \
  --eval_final_diff_dir "${JUDGE_RUN_DIR}/eval_results_final" \
  --output_json "${JUDGE_RUN_DIR}/final_weighted_scores.json"

echo "[DONE] Pipeline finished."
echo "[DONE] Final score file: ${JUDGE_RUN_DIR}/final_weighted_scores.json"
