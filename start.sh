#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-unitale-tts-local}"

export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
export INDEXTTS_MODEL_DIR="${INDEXTTS_MODEL_DIR:-$HF_MIRROR_DIR/IndexTeam/IndexTTS-2}"
export INDEXTTS_CFG_PATH="${INDEXTTS_CFG_PATH:-$INDEXTTS_MODEL_DIR/config.yaml}"
export INDEXTTS_CODE_DIR="${INDEXTTS_CODE_DIR:-$PROJECT_DIR/vendor/index-tts}"
export QWEN_LIBS="${QWEN_LIBS:-$PROJECT_DIR/vendor/qwen_libs}"
export PROMPTS_DIR="${PROMPTS_DIR:-$PROJECT_DIR/prompts}"
export RUNTIME_CACHE_DIR="${RUNTIME_CACHE_DIR:-$PROJECT_DIR/.cache/runtime}"
export GPU_LOCK_FILE="${GPU_LOCK_FILE:-$RUNTIME_CACHE_DIR/gpu-runtime.lock}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
export CLEAN_UNKNOWN_PYTHON_PROCESSES="${CLEAN_UNKNOWN_PYTHON_PROCESSES:-0}"
export INDEXTTS_DEVICE="${INDEXTTS_DEVICE:-}"
export INDEXTTS_USE_FP16="${INDEXTTS_USE_FP16:-1}"
export INDEXTTS_USE_CUDA_KERNEL="${INDEXTTS_USE_CUDA_KERNEL:-0}"
export INDEXTTS_NUM_BEAMS="${INDEXTTS_NUM_BEAMS:-1}"
export CUDA_RELEASE_DELAY="${CUDA_RELEASE_DELAY:-2.0}"
export QWEN_REQUEST_TIMEOUT="${QWEN_REQUEST_TIMEOUT:-120}"
export MIMO_BASE_URL="${MIMO_BASE_URL:-https://api.xiaomimimo.com/v1}"
export MIMO_MODEL="${MIMO_MODEL:-mimo-v2.5-tts-voicedesign}"
export MIMO_AUTH_HEADER="${MIMO_AUTH_HEADER:-api-key}"
export MIMO_TIMEOUT="${MIMO_TIMEOUT:-300}"
export MIMO_MAX_CHARS_PER_CHUNK="${MIMO_MAX_CHARS_PER_CHUNK:-300}"
export MIMO_PAUSE_MS="${MIMO_PAUSE_MS:-250}"
export MIMO_OPTIMIZE_TEXT_PREVIEW="${MIMO_OPTIMIZE_TEXT_PREVIEW:-0}"
export MIMO_MIN_REQUEST_INTERVAL_SECONDS="${MIMO_MIN_REQUEST_INTERVAL_SECONDS:-0}"
export MIMO_MAX_RETRIES="${MIMO_MAX_RETRIES:-3}"
export MIMO_RETRY_BASE_SECONDS="${MIMO_RETRY_BASE_SECONDS:-5}"
export MIMO_RETRY_MAX_SECONDS="${MIMO_RETRY_MAX_SECONDS:-60}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8300}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUNTIME_CACHE_DIR/hf_modules}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$RUNTIME_CACHE_DIR/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_CACHE_DIR/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_CACHE_DIR/xdg}"
mkdir -p "$PROMPTS_DIR" "$HF_MODULES_CACHE" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$(dirname "$GPU_LOCK_FILE")"

echo "=================================================="
echo "   Unitale AI local backend"
echo "=================================================="
echo "Main conda env:      $CONDA_ENV"
echo "Qwen model:          $QWEN_MODEL_DIR"
echo "IndexTTS2 model:     $INDEXTTS_MODEL_DIR"
echo "IndexTTS2 code:      $INDEXTTS_CODE_DIR"
echo "Qwen sidecar libs:   $QWEN_LIBS"
echo "MiMo base URL:       $MIMO_BASE_URL"
echo "MiMo model:          $MIMO_MODEL"
echo "MiMo API key:        $([[ -n "${MIMO_API_KEY:-}" ]] && echo configured || echo missing)"
echo "Prompts dir:         $PROMPTS_DIR"
echo "HF modules cache:    $HF_MODULES_CACHE"
echo "GPU lock file:       $GPU_LOCK_FILE"
echo "IndexTTS2 device:    ${INDEXTTS_DEVICE:-auto}"
echo "IndexTTS2 fp16:      $INDEXTTS_USE_FP16"
echo "IndexTTS2 beams:     $INDEXTTS_NUM_BEAMS"
echo "CUDA kernel:         $INDEXTTS_USE_CUDA_KERNEL"
echo "Main API:            http://$HOST:$PORT"
echo "Main health:         http://127.0.0.1:$PORT/v1/health"
echo "Qwen design route:   http://127.0.0.1:$PORT/v1/qwen/design"
echo "MiMo design route:   http://127.0.0.1:$PORT/v1/mimo/design"
echo "=================================================="

cd "$PROJECT_DIR"

main_pid=""

cleanup() {
  local status=$?
  trap - INT TERM EXIT

  if [[ -n "$main_pid" ]] && kill -0 "$main_pid" 2>/dev/null; then
    kill "$main_pid" 2>/dev/null || true
  fi

  wait "$main_pid" 2>/dev/null || true
  exit "$status"
}

trap cleanup INT TERM EXIT

conda run --no-capture-output -n "$CONDA_ENV" python api.py &
main_pid=$!

wait "$main_pid"
