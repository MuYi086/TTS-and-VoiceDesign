#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-unitale-tts-local}"

export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
export INDEXTTS_MODEL_DIR="${INDEXTTS_MODEL_DIR:-$HF_MIRROR_DIR/IndexTeam/IndexTTS-2}"
export DOTS_MODEL_DIR="${DOTS_MODEL_DIR:-$HF_MIRROR_DIR/rednote-hilab/dots.tts-base}"
export LONGCAT_MODEL_DIR="${LONGCAT_MODEL_DIR:-$HF_MIRROR_DIR/meituan-longcat/LongCat-AudioDiT-1B}"
export MOSS_MODEL_DIR="${MOSS_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5}"
export LONGCAT_TOKENIZER_PATH="${LONGCAT_TOKENIZER_PATH:-$HF_MIRROR_DIR/google/umt5-base}"
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
export DOTS_CONDA_ENV="${DOTS_CONDA_ENV:-dots_tts}"
export DOTS_LANGUAGE="${DOTS_LANGUAGE:-chinese}"
export DOTS_TEMPLATE_NAME="${DOTS_TEMPLATE_NAME:-}"
export DOTS_PRECISION="${DOTS_PRECISION:-bfloat16}"
export DOTS_SEED="${DOTS_SEED:-42}"
export DOTS_ODE_METHOD="${DOTS_ODE_METHOD:-euler}"
export DOTS_NUM_STEPS="${DOTS_NUM_STEPS:-10}"
export DOTS_GUIDANCE_SCALE="${DOTS_GUIDANCE_SCALE:-1.2}"
export DOTS_SPEAKER_SCALE="${DOTS_SPEAKER_SCALE:-1.5}"
export DOTS_MAX_GENERATE_LENGTH="${DOTS_MAX_GENERATE_LENGTH:-500}"
export DOTS_MAX_CHARS_PER_CHUNK="${DOTS_MAX_CHARS_PER_CHUNK:-120}"
export DOTS_PAUSE_MS="${DOTS_PAUSE_MS:-250}"
export DOTS_NORMALIZE_TEXT="${DOTS_NORMALIZE_TEXT:-0}"
export DOTS_PROFILE_INFERENCE="${DOTS_PROFILE_INFERENCE:-0}"
export DOTS_REQUEST_TIMEOUT="${DOTS_REQUEST_TIMEOUT:-300}"
export LONGCAT_CONDA_ENV="${LONGCAT_CONDA_ENV:-longcat_audiodit}"
export LONGCAT_REPO_PATH="${LONGCAT_REPO_PATH:-$PROJECT_DIR/vendor/LongCat-AudioDiT}"
export MOSS_CONDA_ENV="${MOSS_CONDA_ENV:-moss-tts-py310}"
export MOSS_HELPER_SCRIPT="${MOSS_HELPER_SCRIPT:-$HOME/github/timbre-design/scripts/tts_local_moss_tts_local_transformer.py}"
export MOSS_LANGUAGE="${MOSS_LANGUAGE:-Chinese}"
export MOSS_INSTRUCTION="${MOSS_INSTRUCTION:-}"
export MOSS_QUALITY="${MOSS_QUALITY:-}"
export MOSS_TOKENS="${MOSS_TOKENS:-}"
export MOSS_MAX_NEW_TOKENS="${MOSS_MAX_NEW_TOKENS:-4096}"
export MOSS_N_VQ_FOR_INFERENCE="${MOSS_N_VQ_FOR_INFERENCE:-}"
export MOSS_AUDIO_TEMPERATURE="${MOSS_AUDIO_TEMPERATURE:-1.7}"
export MOSS_AUDIO_TOP_P="${MOSS_AUDIO_TOP_P:-0.8}"
export MOSS_AUDIO_TOP_K="${MOSS_AUDIO_TOP_K:-25}"
export MOSS_AUDIO_REPETITION_PENALTY="${MOSS_AUDIO_REPETITION_PENALTY:-1.0}"
export MOSS_TEXT_TEMPERATURE="${MOSS_TEXT_TEMPERATURE:-}"
export MOSS_TEXT_TOP_P="${MOSS_TEXT_TOP_P:-}"
export MOSS_TEXT_TOP_K="${MOSS_TEXT_TOP_K:-}"
export MOSS_TEXT_REPETITION_PENALTY="${MOSS_TEXT_REPETITION_PENALTY:-}"
export MOSS_ATTN_IMPLEMENTATION="${MOSS_ATTN_IMPLEMENTATION:-auto}"
export MOSS_DTYPE="${MOSS_DTYPE:-auto}"
export MOSS_MAX_CHARS_PER_CHUNK="${MOSS_MAX_CHARS_PER_CHUNK:-300}"
export MOSS_PAUSE_MS="${MOSS_PAUSE_MS:-250}"
export MOSS_REQUEST_TIMEOUT="${MOSS_REQUEST_TIMEOUT:-600}"
export LONGCAT_MAX_CHARS_PER_CHUNK="${LONGCAT_MAX_CHARS_PER_CHUNK:-90}"
export LONGCAT_PAUSE_MS="${LONGCAT_PAUSE_MS:-250}"
export LONGCAT_NFE="${LONGCAT_NFE:-16}"
export LONGCAT_GUIDANCE_STRENGTH="${LONGCAT_GUIDANCE_STRENGTH:-4.0}"
export LONGCAT_GUIDANCE_METHOD="${LONGCAT_GUIDANCE_METHOD:-apg}"
export LONGCAT_SEED="${LONGCAT_SEED:-1024}"
export LONGCAT_DURATION_SCALE="${LONGCAT_DURATION_SCALE:-1.0}"
export LONGCAT_VAE_DTYPE="${LONGCAT_VAE_DTYPE:-float16}"
export LONGCAT_REQUEST_TIMEOUT="${LONGCAT_REQUEST_TIMEOUT:-600}"
export LONGCAT_TRIM_LEADING_SILENCE="${LONGCAT_TRIM_LEADING_SILENCE:-1}"
export LONGCAT_TRIM_LEADING_SILENCE_THRESHOLD_DB="${LONGCAT_TRIM_LEADING_SILENCE_THRESHOLD_DB:--42}"
export LONGCAT_TRIM_LEADING_SILENCE_MIN_MS="${LONGCAT_TRIM_LEADING_SILENCE_MIN_MS:-120}"
export LONGCAT_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS="${LONGCAT_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS:-30}"
export LONGCAT_TRIM_LEADING_SILENCE_PRE_ROLL_MS="${LONGCAT_TRIM_LEADING_SILENCE_PRE_ROLL_MS:-40}"
export LONGCAT_TRIM_LEADING_SILENCE_MAX_MS="${LONGCAT_TRIM_LEADING_SILENCE_MAX_MS:-8000}"
export LONGCAT_AUTO_PROMPT_TEXT="${LONGCAT_AUTO_PROMPT_TEXT:-1}"
export LONGCAT_ASR_MODEL_DIR="${LONGCAT_ASR_MODEL_DIR:-$HF_MIRROR_DIR/FunAudioLLM/SenseVoiceSmall}"
export LONGCAT_ASR_DEVICE="${LONGCAT_ASR_DEVICE:-cpu}"
export LONGCAT_ASR_LANGUAGE="${LONGCAT_ASR_LANGUAGE:-auto}"
export LONGCAT_ASR_TIMEOUT="${LONGCAT_ASR_TIMEOUT:-180}"
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
export DOTS_HOST="${DOTS_HOST:-$HOST}"
export DOTS_PORT="${DOTS_PORT:-8301}"
export LONGCAT_HOST="${LONGCAT_HOST:-$HOST}"
export LONGCAT_PORT="${LONGCAT_PORT:-8302}"
export MOSS_HOST="${MOSS_HOST:-$HOST}"
export MOSS_PORT="${MOSS_PORT:-8303}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUNTIME_CACHE_DIR/hf_modules}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$RUNTIME_CACHE_DIR/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_CACHE_DIR/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_CACHE_DIR/xdg}"
mkdir -p "$PROMPTS_DIR" "$HF_MODULES_CACHE" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$(dirname "$GPU_LOCK_FILE")"

if [[ -z "${MOSS_CODEC_PATH:-}" ]]; then
  default_moss_codec_path="$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
  if [[ -e "$default_moss_codec_path" ]]; then
    export MOSS_CODEC_PATH="$default_moss_codec_path"
  else
    export MOSS_CODEC_PATH="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
  fi
fi

echo "=================================================="
echo "   Unitale AI local backend"
echo "=================================================="
echo "Main conda env:      $CONDA_ENV"
echo "Qwen model:          $QWEN_MODEL_DIR"
echo "IndexTTS2 model:     $INDEXTTS_MODEL_DIR"
echo "IndexTTS2 code:      $INDEXTTS_CODE_DIR"
echo "dots.tts worker env: $DOTS_CONDA_ENV"
echo "dots.tts model:      $DOTS_MODEL_DIR"
echo "LongCat worker env:  $LONGCAT_CONDA_ENV"
echo "LongCat model:       $LONGCAT_MODEL_DIR"
echo "LongCat tokenizer:   $LONGCAT_TOKENIZER_PATH"
echo "LongCat repo path:   ${LONGCAT_REPO_PATH:-auto-detect}"
echo "MOSS worker env:     $MOSS_CONDA_ENV"
echo "MOSS model:          $MOSS_MODEL_DIR"
echo "MOSS codec:          $MOSS_CODEC_PATH"
echo "MOSS helper script:  $MOSS_HELPER_SCRIPT"
echo "LongCat auto prompt: $LONGCAT_AUTO_PROMPT_TEXT"
echo "LongCat trim lead:   $LONGCAT_TRIM_LEADING_SILENCE"
echo "LongCat trim thres:  $LONGCAT_TRIM_LEADING_SILENCE_THRESHOLD_DB dB"
echo "LongCat trim min:    $LONGCAT_TRIM_LEADING_SILENCE_MIN_MS ms"
echo "LongCat ASR model:   $LONGCAT_ASR_MODEL_DIR"
echo "LongCat ASR device:  $LONGCAT_ASR_DEVICE"
echo "LongCat ASR lang:    $LONGCAT_ASR_LANGUAGE"
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
echo "dots API:            http://$DOTS_HOST:$DOTS_PORT"
echo "dots health:         http://127.0.0.1:$DOTS_PORT/v1/health"
echo "LongCat API:         http://$LONGCAT_HOST:$LONGCAT_PORT"
echo "LongCat health:      http://127.0.0.1:$LONGCAT_PORT/v1/health"
echo "MOSS API:            http://$MOSS_HOST:$MOSS_PORT"
echo "MOSS health:         http://127.0.0.1:$MOSS_PORT/v1/health"
echo "Qwen design route:   http://127.0.0.1:$PORT/v1/qwen/design"
echo "MiMo design route:   http://127.0.0.1:$PORT/v1/mimo/design"
echo "dots synth route:    http://127.0.0.1:$DOTS_PORT/v2/synthesize"
echo "LongCat synth route: http://127.0.0.1:$LONGCAT_PORT/v2/synthesize"
echo "MOSS synth route:    http://127.0.0.1:$MOSS_PORT/v2/synthesize"
echo "=================================================="

cd "$PROJECT_DIR"

main_pid=""
dots_pid=""
longcat_pid=""
moss_pid=""

cleanup() {
  local status=$?
  trap - INT TERM EXIT

  for pid in "$main_pid" "$dots_pid" "$longcat_pid" "$moss_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  wait "$main_pid" 2>/dev/null || true
  wait "$dots_pid" 2>/dev/null || true
  wait "$longcat_pid" 2>/dev/null || true
  wait "$moss_pid" 2>/dev/null || true
  exit "$status"
}

trap cleanup INT TERM EXIT

conda run --no-capture-output -n "$CONDA_ENV" python api.py &
main_pid=$!
HOST="$DOTS_HOST" PORT="$DOTS_PORT" conda run --no-capture-output -n "$CONDA_ENV" python dots_api.py &
dots_pid=$!
HOST="$LONGCAT_HOST" PORT="$LONGCAT_PORT" conda run --no-capture-output -n "$CONDA_ENV" python longcat_api.py &
longcat_pid=$!
HOST="$MOSS_HOST" PORT="$MOSS_PORT" conda run --no-capture-output -n "$CONDA_ENV" python moss_api.py &
moss_pid=$!

wait -n "$main_pid" "$dots_pid" "$longcat_pid" "$moss_pid"
