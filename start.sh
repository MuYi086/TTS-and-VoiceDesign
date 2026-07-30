#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$PROJECT_DIR/api"
CONDA_ENV="${CONDA_ENV:-unitale-tts-local}"

export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
export INDEXTTS_MODEL_DIR="${INDEXTTS_MODEL_DIR:-$HF_MIRROR_DIR/IndexTeam/IndexTTS-2}"
export MOSS_SOUNDEFFECT_CONDA_ENV="${MOSS_SOUNDEFFECT_CONDA_ENV:-moss-soundEffect}"
export MOSS_SOUNDEFFECT_MODEL_DIR="${MOSS_SOUNDEFFECT_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0}"
export MOSS_SOUNDEFFECT_DEVICE="${MOSS_SOUNDEFFECT_DEVICE:-cuda}"
export MOSS_SOUNDEFFECT_DTYPE="${MOSS_SOUNDEFFECT_DTYPE:-bfloat16}"
export MOSS_SOUNDEFFECT_DEFAULT_SECONDS="${MOSS_SOUNDEFFECT_DEFAULT_SECONDS:-10}"
export MOSS_SOUNDEFFECT_DEFAULT_STEPS="${MOSS_SOUNDEFFECT_DEFAULT_STEPS:-100}"
export MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE="${MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE:-4.0}"
export MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT="${MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT:-5.0}"
export MOSS_SOUNDEFFECT_DEFAULT_SEED="${MOSS_SOUNDEFFECT_DEFAULT_SEED:-0}"
export MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO="${MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO:-1}"
export QWEN3_TTS_MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
export VOXCPM2_MODEL_DIR="${VOXCPM2_MODEL_DIR:-$HF_MIRROR_DIR/openbmb/VoxCPM2}"
export INDEXTTS_CFG_PATH="${INDEXTTS_CFG_PATH:-$INDEXTTS_MODEL_DIR/config.yaml}"
export INDEXTTS_CODE_DIR="${INDEXTTS_CODE_DIR:-$API_DIR/vendor/index-tts}"
export QWEN_LIBS="${QWEN_LIBS:-$API_DIR/vendor/qwen_libs}"
export PROMPTS_DIR="${PROMPTS_DIR:-$API_DIR/prompts}"
export RUNTIME_CACHE_DIR="${RUNTIME_CACHE_DIR:-$API_DIR/.cache/runtime}"
export GPU_LOCK_FILE="${GPU_LOCK_FILE:-$RUNTIME_CACHE_DIR/gpu-runtime.lock}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
export CLEAN_UNKNOWN_PYTHON_PROCESSES="${CLEAN_UNKNOWN_PYTHON_PROCESSES:-0}"
export INDEXTTS_DEVICE="${INDEXTTS_DEVICE:-}"
export INDEXTTS_USE_FP16="${INDEXTTS_USE_FP16:-1}"
export INDEXTTS_USE_CUDA_KERNEL="${INDEXTTS_USE_CUDA_KERNEL:-0}"
export INDEXTTS_NUM_BEAMS="${INDEXTTS_NUM_BEAMS:-1}"
export INDEXTTS_REQUEST_TIMEOUT="${INDEXTTS_REQUEST_TIMEOUT:-600}"
export INDEXTTS_CUDA_RETRY_COUNT="${INDEXTTS_CUDA_RETRY_COUNT:-1}"
export INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT="${INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT:-80}"
export INDEXTTS_MAX_MEL_TOKENS="${INDEXTTS_MAX_MEL_TOKENS:-1200}"
export INDEXTTS_CUDA_RETRY_MAX_TEXT_TOKENS="${INDEXTTS_CUDA_RETRY_MAX_TEXT_TOKENS:-50}"
export INDEXTTS_CUDA_RETRY_MAX_MEL_TOKENS="${INDEXTTS_CUDA_RETRY_MAX_MEL_TOKENS:-900}"
# VoxCPM2 的默认生成参数集中在 api/voxcpm2_api.py；此处不再写入默认值，
# 因此直接修改 API 顶部常量即可生效。外部显式设置的 VOXCPM2_* 环境变量会原样继承并覆盖 API 默认值。
# 该变量决定启动 8306 服务所使用的 Conda 环境，必须在脚本内解析；其余
# VoxCPM2 配置由 voxcpm2_api.py 在服务进程内统一处理。
export VOXCPM2_CONDA_ENV="${VOXCPM2_CONDA_ENV:-voxcpm2}"
export QWEN3_TTS_CONDA_ENV="${QWEN3_TTS_CONDA_ENV:-qwen3-tts}"
export QWEN3_TTS_DEVICE_MAP="${QWEN3_TTS_DEVICE_MAP:-cuda:0}"
export QWEN3_TTS_DTYPE="${QWEN3_TTS_DTYPE:-auto}"
export QWEN3_TTS_LANGUAGE="${QWEN3_TTS_LANGUAGE:-Chinese}"
export QWEN3_TTS_MAX_NEW_TOKENS="${QWEN3_TTS_MAX_NEW_TOKENS:-2048}"
export QWEN3_TTS_TOP_P="${QWEN3_TTS_TOP_P:-}"
export QWEN3_TTS_TEMPERATURE="${QWEN3_TTS_TEMPERATURE:-}"
export QWEN3_TTS_ATTN_IMPLEMENTATION="${QWEN3_TTS_ATTN_IMPLEMENTATION:-auto}"
export QWEN3_TTS_X_VECTOR_ONLY="${QWEN3_TTS_X_VECTOR_ONLY:-0}"
export QWEN3_TTS_USE_QWEN_LIBS="${QWEN3_TTS_USE_QWEN_LIBS:-0}"
export QWEN3_TTS_MAX_CHARS_PER_CHUNK="${QWEN3_TTS_MAX_CHARS_PER_CHUNK:-120}"
export QWEN3_TTS_PAUSE_MS="${QWEN3_TTS_PAUSE_MS:-250}"
export QWEN3_TTS_TRIM_LEADING_SILENCE="${QWEN3_TTS_TRIM_LEADING_SILENCE:-1}"
export QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB="${QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB:--42}"
export QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS="${QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS:-120}"
export QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS="${QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS:-30}"
export QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS="${QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS:-40}"
export QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS="${QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS:-8000}"
export QWEN3_TTS_REQUEST_TIMEOUT="${QWEN3_TTS_REQUEST_TIMEOUT:-600}"
export MOSS_SOUNDEFFECT_REQUEST_TIMEOUT="${MOSS_SOUNDEFFECT_REQUEST_TIMEOUT:-600}"
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
export SOUNDEFFECT_HOST="${SOUNDEFFECT_HOST:-$HOST}"
export SOUNDEFFECT_PORT="${SOUNDEFFECT_PORT:-8311}"
export QWEN3_TTS_HOST="${QWEN3_TTS_HOST:-$HOST}"
export QWEN3_TTS_PORT="${QWEN3_TTS_PORT:-8305}"
export VOXCPM2_HOST="${VOXCPM2_HOST:-$HOST}"
export VOXCPM2_PORT="${VOXCPM2_PORT:-8306}"

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
echo "SoundEffect env:     $MOSS_SOUNDEFFECT_CONDA_ENV"
echo "SoundEffect model:   $MOSS_SOUNDEFFECT_MODEL_DIR"
echo "SoundEffect device:  $MOSS_SOUNDEFFECT_DEVICE ($MOSS_SOUNDEFFECT_DTYPE)"
echo "Qwen3-TTS worker env: $QWEN3_TTS_CONDA_ENV"
echo "Qwen3-TTS model:     $QWEN3_TTS_MODEL_DIR"
echo "VoxCPM2 worker env:  $VOXCPM2_CONDA_ENV"
echo "VoxCPM2 model:       $VOXCPM2_MODEL_DIR"
echo "VoxCPM2 config:      managed by api/voxcpm2_api.py"
echo "Qwen3-TTS trim lead: $QWEN3_TTS_TRIM_LEADING_SILENCE"
echo "Qwen3-TTS trim thres:$QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB dB"
echo "Qwen3-TTS trim min:  $QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS ms"
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
echo "IndexTTS2 timeout:    $INDEXTTS_REQUEST_TIMEOUT s"
echo "IndexTTS2 CUDA retry: $INDEXTTS_CUDA_RETRY_COUNT"
echo "IndexTTS2 segment:    $INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT tokens"
echo "IndexTTS2 max mel:    $INDEXTTS_MAX_MEL_TOKENS tokens"
echo "Main API:            http://$HOST:$PORT"
echo "Main health:         http://127.0.0.1:$PORT/v1/health"
echo "SoundEffect API:     http://$SOUNDEFFECT_HOST:$SOUNDEFFECT_PORT"
echo "SoundEffect health:  http://127.0.0.1:$SOUNDEFFECT_PORT/v1/health"
echo "Qwen3-TTS API:       http://$QWEN3_TTS_HOST:$QWEN3_TTS_PORT"
echo "Qwen3-TTS health:    http://127.0.0.1:$QWEN3_TTS_PORT/v1/health"
echo "VoxCPM2 API:         http://$VOXCPM2_HOST:$VOXCPM2_PORT"
echo "VoxCPM2 health:      http://127.0.0.1:$VOXCPM2_PORT/v1/health"
echo "Qwen design route:   http://127.0.0.1:$PORT/v1/qwen/design"
echo "MiMo design route:   http://127.0.0.1:$PORT/v1/mimo/design"
echo "SoundEffect route:   http://127.0.0.1:$SOUNDEFFECT_PORT/v1/generate"
echo "Qwen3-TTS synth:     http://127.0.0.1:$QWEN3_TTS_PORT/v2/synthesize"
echo "VoxCPM2 synth:       http://127.0.0.1:$VOXCPM2_PORT/v2/synthesize"
echo "=================================================="

cd "$PROJECT_DIR"

main_pid=""
soundeffect_pid=""
qwen3_tts_pid=""
voxcpm2_pid=""

cleanup() {
  local status=$?
  trap - INT TERM EXIT

  for pid in "$main_pid" "$soundeffect_pid" "$qwen3_tts_pid" "$voxcpm2_pid"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done

  sleep 1

  for pid in "$main_pid" "$soundeffect_pid" "$qwen3_tts_pid" "$voxcpm2_pid"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  done

  wait "$main_pid" 2>/dev/null || true
  wait "$soundeffect_pid" 2>/dev/null || true
  wait "$qwen3_tts_pid" 2>/dev/null || true
  wait "$voxcpm2_pid" 2>/dev/null || true
  exit "$status"
}

trap cleanup INT TERM EXIT

setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/api.py" &
main_pid=$!
HOST="$SOUNDEFFECT_HOST" PORT="$SOUNDEFFECT_PORT" setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/soundeffect_api.py" &
soundeffect_pid=$!
HOST="$QWEN3_TTS_HOST" PORT="$QWEN3_TTS_PORT" setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/qwen3_tts_api.py" &
qwen3_tts_pid=$!
HOST="$VOXCPM2_HOST" PORT="$VOXCPM2_PORT" setsid conda run --no-capture-output -n "$VOXCPM2_CONDA_ENV" python "$API_DIR/voxcpm2_api.py" &
voxcpm2_pid=$!

wait -n "$main_pid" "$soundeffect_pid" "$qwen3_tts_pid" "$voxcpm2_pid"
