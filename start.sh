#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$PROJECT_DIR/api"
# The HTTP wrappers are intentionally lightweight and run in qwen3-tts, which
# already contains FastAPI/uvicorn; heavyweight inference is delegated to the
# model's dedicated Conda worker below.
CONDA_ENV="${CONDA_ENV:-qwen3-tts}"

export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
export QWEN_VOICEDESIGN_CONDA_ENV="${QWEN_VOICEDESIGN_CONDA_ENV:-qwen3-voiceDesign}"
export QWEN_VOICEDESIGN_WORKER_TIMEOUT="${QWEN_VOICEDESIGN_WORKER_TIMEOUT:-900}"
export MOSS_VOICEGENERATOR_CONDA_ENV="${MOSS_VOICEGENERATOR_CONDA_ENV:-moss-voiceGenerator}"
export MOSS_VOICEGENERATOR_MODEL_DIR="${MOSS_VOICEGENERATOR_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator}"
export MOSS_AUDIO_TOKENIZER_PATH="${MOSS_AUDIO_TOKENIZER_PATH:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer}"
export MOSS_VOICEGENERATOR_REQUEST_TIMEOUT="${MOSS_VOICEGENERATOR_REQUEST_TIMEOUT:-900}"
export MING_OMNI_TTS_CONDA_ENV="${MING_OMNI_TTS_CONDA_ENV:-Ming-omni-tts-0.5B}"
export MING_OMNI_TTS_MODEL_DIR="${MING_OMNI_TTS_MODEL_DIR:-$HF_MIRROR_DIR/inclusionAI/Ming-omni-tts-0.5B}"
export MING_OMNI_TTS_CODE_PATH="${MING_OMNI_TTS_CODE_PATH:-$HOME/tts-depency/Ming-omni-tts}"
export MING_OMNI_TTS_REQUEST_TIMEOUT="${MING_OMNI_TTS_REQUEST_TIMEOUT:-900}"
export STEP_AUDIO_EDITX_CONDA_ENV="${STEP_AUDIO_EDITX_CONDA_ENV:-Step-Audio-EditX}"
export STEP_AUDIO_EDITX_MODEL_DIR="${STEP_AUDIO_EDITX_MODEL_DIR:-$HF_MIRROR_DIR/stepfun-ai/Step-Audio-EditX}"
export STEP_AUDIO_TOKENIZER_PATH="${STEP_AUDIO_TOKENIZER_PATH:-$HF_MIRROR_DIR/stepfun-ai/Step-Audio-Tokenizer}"
export STEP_AUDIO_EDITX_CODE_PATH="${STEP_AUDIO_EDITX_CODE_PATH:-$HOME/tts-depency/Step-Audio-EditX}"
export STEP_AUDIO_EDITX_REQUEST_TIMEOUT="${STEP_AUDIO_EDITX_REQUEST_TIMEOUT:-900}"
export STEP_AUDIO_EDITX_DTYPE="${STEP_AUDIO_EDITX_DTYPE:-bfloat16}"
export STEP_AUDIO_EDITX_MAX_MODEL_LEN="${STEP_AUDIO_EDITX_MAX_MODEL_LEN:-3072}"
export STEP_AUDIO_EDITX_GPU_MEMORY_UTILIZATION="${STEP_AUDIO_EDITX_GPU_MEMORY_UTILIZATION:-0.5}"
export STEP_AUDIO_EDITX_MAX_NUM_SEQS="${STEP_AUDIO_EDITX_MAX_NUM_SEQS:-1}"
export STEP_AUDIO_EDITX_COSYVOICE_DTYPE="${STEP_AUDIO_EDITX_COSYVOICE_DTYPE:-bfloat16}"
export STEP_AUDIO_EDITX_ENFORCE_EAGER="${STEP_AUDIO_EDITX_ENFORCE_EAGER:-1}"
export STEP_AUDIO_EDITX_COSYVOICE_CUDA_GRAPH="${STEP_AUDIO_EDITX_COSYVOICE_CUDA_GRAPH:-0}"
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
export STABLE_AUDIO_3_SMALL_SFX_CONDA_ENV="${STABLE_AUDIO_3_SMALL_SFX_CONDA_ENV:-stable_audio_3_small_sfx}"
export STABLE_AUDIO_3_SMALL_SFX_MODEL_DIR="${STABLE_AUDIO_3_SMALL_SFX_MODEL_DIR:-$HF_MIRROR_DIR/stabilityai/stable-audio-3-small-sfx}"
export STABLE_AUDIO_3_REPO_PATH="${STABLE_AUDIO_3_REPO_PATH:-$HOME/tts-depency/stable-audio-3}"
export STABLE_AUDIO_3_SMALL_SFX_DEVICE="${STABLE_AUDIO_3_SMALL_SFX_DEVICE:-auto}"
export STABLE_AUDIO_3_SMALL_SFX_DTYPE="${STABLE_AUDIO_3_SMALL_SFX_DTYPE:-auto}"
export STABLE_AUDIO_3_SMALL_SFX_DEFAULT_SECONDS="${STABLE_AUDIO_3_SMALL_SFX_DEFAULT_SECONDS:-7}"
export STABLE_AUDIO_3_SMALL_SFX_DEFAULT_STEPS="${STABLE_AUDIO_3_SMALL_SFX_DEFAULT_STEPS:-8}"
export STABLE_AUDIO_3_SMALL_SFX_DEFAULT_CFG_SCALE="${STABLE_AUDIO_3_SMALL_SFX_DEFAULT_CFG_SCALE:-1.0}"
export STABLE_AUDIO_3_SMALL_SFX_DEFAULT_SEED="${STABLE_AUDIO_3_SMALL_SFX_DEFAULT_SEED:--1}"
export STABLE_AUDIO_3_SMALL_SFX_REQUEST_TIMEOUT="${STABLE_AUDIO_3_SMALL_SFX_REQUEST_TIMEOUT:-900}"
export STABLE_AUDIO_3_MEDIUM_CONDA_ENV="${STABLE_AUDIO_3_MEDIUM_CONDA_ENV:-stable_audio_3_medium}"
export STABLE_AUDIO_3_MEDIUM_MODEL_DIR="${STABLE_AUDIO_3_MEDIUM_MODEL_DIR:-$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium}"
export STABLE_AUDIO_3_MEDIUM_DEVICE="${STABLE_AUDIO_3_MEDIUM_DEVICE:-cuda}"
export STABLE_AUDIO_3_MEDIUM_DTYPE="${STABLE_AUDIO_3_MEDIUM_DTYPE:-float16}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS="${STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS:-7}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS="${STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS:-8}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE="${STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE:-1.0}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED="${STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED:--1}"
export STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT="${STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT:-900}"
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
# LongCat、dots.tts-soar、Qwen3-TTS 的克隆默认值集中在各自 API 顶部；
# 此处只保留环境、模型路径等启动路由配置，避免覆盖 API 内可直接调试的值。
# IndexTTS2 的克隆默认值集中在 api/api.py 顶部；此处不再写入默认参数，
# 这样直接修改 api/api.py 的全局值即可生效。外部显式设置的 INDEXTTS_* 变量仍会继承。
# VoxCPM2 的默认生成参数集中在 api/voxcpm2_api.py；此处不再写入默认值，
# 因此直接修改 API 顶部常量即可生效。外部显式设置的 VOXCPM2_* 环境变量会原样继承并覆盖 API 默认值。
# 该变量决定启动 8306 服务所使用的 Conda 环境，必须在脚本内解析；其余
# VoxCPM2 配置由 voxcpm2_api.py 在服务进程内统一处理。
export VOXCPM2_CONDA_ENV="${VOXCPM2_CONDA_ENV:-voxcpm2}"
export LONGCAT_AUDIODIT_CONDA_ENV="${LONGCAT_AUDIODIT_CONDA_ENV:-LongCat-AudioDiT-3.5B-bf16}"
export LONGCAT_AUDIODIT_MODEL_DIR="${LONGCAT_AUDIODIT_MODEL_DIR:-$HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16}"
export LONGCAT_AUDIODIT_REPO_PATH="${LONGCAT_AUDIODIT_REPO_PATH:-$HOME/tts-depency/LongCat-AudioDiT}"
export LONGCAT_AUDIODIT_TOKENIZER_PATH="${LONGCAT_AUDIODIT_TOKENIZER_PATH:-$HF_MIRROR_DIR/google/umt5-base}"
export DOTS_TTS_SOAR_CONDA_ENV="${DOTS_TTS_SOAR_CONDA_ENV:-dots_tts_soar}"
export DOTS_TTS_SOAR_MODEL_DIR="${DOTS_TTS_SOAR_MODEL_DIR:-$HF_MIRROR_DIR/rednote-hilab/dots.tts-soar}"
export QWEN3_TTS_CONDA_ENV="${QWEN3_TTS_CONDA_ENV:-qwen3-tts}"
export QWEN3_TTS_USE_QWEN_LIBS="${QWEN3_TTS_USE_QWEN_LIBS:-0}"
export MOSS_SOUNDEFFECT_REQUEST_TIMEOUT="${MOSS_SOUNDEFFECT_REQUEST_TIMEOUT:-600}"
export CUDA_RELEASE_DELAY="${CUDA_RELEASE_DELAY:-2.0}"
export QWEN_REQUEST_TIMEOUT="${QWEN_REQUEST_TIMEOUT:-900}"
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
export STABLE_AUDIO_3_SMALL_SFX_HOST="${STABLE_AUDIO_3_SMALL_SFX_HOST:-$HOST}"
export STABLE_AUDIO_3_SMALL_SFX_PORT="${STABLE_AUDIO_3_SMALL_SFX_PORT:-8312}"
export STABLE_AUDIO_3_MEDIUM_HOST="${STABLE_AUDIO_3_MEDIUM_HOST:-$HOST}"
export STABLE_AUDIO_3_MEDIUM_PORT="${STABLE_AUDIO_3_MEDIUM_PORT:-8313}"
export QWEN3_TTS_HOST="${QWEN3_TTS_HOST:-$HOST}"
export QWEN3_TTS_PORT="${QWEN3_TTS_PORT:-8305}"
export VOXCPM2_HOST="${VOXCPM2_HOST:-$HOST}"
export VOXCPM2_PORT="${VOXCPM2_PORT:-8306}"
export LONGCAT_AUDIODIT_HOST="${LONGCAT_AUDIODIT_HOST:-$HOST}"
export LONGCAT_AUDIODIT_PORT="${LONGCAT_AUDIODIT_PORT:-8307}"
export DOTS_TTS_SOAR_HOST="${DOTS_TTS_SOAR_HOST:-$HOST}"
export DOTS_TTS_SOAR_PORT="${DOTS_TTS_SOAR_PORT:-8308}"

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
echo "Qwen VoiceDesign env: $QWEN_VOICEDESIGN_CONDA_ENV"
echo "MOSS VoiceDesign env: $MOSS_VOICEGENERATOR_CONDA_ENV"
echo "MOSS VoiceGenerator:  $MOSS_VOICEGENERATOR_MODEL_DIR"
echo "Ming TTS env:         $MING_OMNI_TTS_CONDA_ENV"
echo "Ming TTS model:       $MING_OMNI_TTS_MODEL_DIR"
echo "Step-Audio-EditX env: $STEP_AUDIO_EDITX_CONDA_ENV"
echo "Step-Audio-EditX model: $STEP_AUDIO_EDITX_MODEL_DIR"
echo "Step-Audio tokenizer: $STEP_AUDIO_TOKENIZER_PATH"
echo "Step-Audio-EditX code: $STEP_AUDIO_EDITX_CODE_PATH"
echo "IndexTTS2 model:     $INDEXTTS_MODEL_DIR"
echo "IndexTTS2 code:      $INDEXTTS_CODE_DIR"
echo "SoundEffect env:     $MOSS_SOUNDEFFECT_CONDA_ENV"
echo "SoundEffect model:   $MOSS_SOUNDEFFECT_MODEL_DIR"
echo "SoundEffect device:  $MOSS_SOUNDEFFECT_DEVICE ($MOSS_SOUNDEFFECT_DTYPE)"
echo "Stable Audio 3 SFX env:    $STABLE_AUDIO_3_SMALL_SFX_CONDA_ENV"
echo "Stable Audio 3 SFX model:  $STABLE_AUDIO_3_SMALL_SFX_MODEL_DIR"
echo "Stable Audio 3 source:     $STABLE_AUDIO_3_REPO_PATH"
echo "Stable Audio 3 SFX device: $STABLE_AUDIO_3_SMALL_SFX_DEVICE ($STABLE_AUDIO_3_SMALL_SFX_DTYPE)"
echo "Stable Audio 3 Medium env:    $STABLE_AUDIO_3_MEDIUM_CONDA_ENV"
echo "Stable Audio 3 Medium model:  $STABLE_AUDIO_3_MEDIUM_MODEL_DIR"
echo "Stable Audio 3 Medium device: $STABLE_AUDIO_3_MEDIUM_DEVICE ($STABLE_AUDIO_3_MEDIUM_DTYPE)"
echo "Qwen3-TTS worker env: $QWEN3_TTS_CONDA_ENV"
echo "Qwen3-TTS model:     $QWEN3_TTS_MODEL_DIR"
echo "VoxCPM2 worker env:  $VOXCPM2_CONDA_ENV"
echo "VoxCPM2 model:       $VOXCPM2_MODEL_DIR"
echo "LongCat worker env:  $LONGCAT_AUDIODIT_CONDA_ENV"
echo "LongCat model:       $LONGCAT_AUDIODIT_MODEL_DIR"
echo "LongCat repo:        $LONGCAT_AUDIODIT_REPO_PATH"
echo "LongCat tokenizer:   $LONGCAT_AUDIODIT_TOKENIZER_PATH"
echo "LongCat config:      managed by api/longcat_audiodit_api.py"
echo "dots.tts-soar env:   $DOTS_TTS_SOAR_CONDA_ENV"
echo "dots.tts-soar model: $DOTS_TTS_SOAR_MODEL_DIR"
echo "dots.tts-soar config: managed by api/dots_tts_soar_api.py"
echo "VoxCPM2 config:      managed by api/voxcpm2_api.py"
echo "Qwen3-TTS config:    managed by api/qwen3_tts_api.py"
echo "Qwen sidecar libs:   $QWEN_LIBS"
echo "MiMo base URL:       $MIMO_BASE_URL"
echo "MiMo model:          $MIMO_MODEL"
echo "MiMo API key:        $([[ -n "${MIMO_API_KEY:-}" ]] && echo configured || echo missing)"
echo "Prompts dir:         $PROMPTS_DIR"
echo "HF modules cache:    $HF_MODULES_CACHE"
echo "GPU lock file:       $GPU_LOCK_FILE"
echo "IndexTTS2 config:    managed by api/api.py"
echo "Main API:            http://$HOST:$PORT"
echo "Main health:         http://127.0.0.1:$PORT/v1/health"
echo "SoundEffect API:     http://$SOUNDEFFECT_HOST:$SOUNDEFFECT_PORT"
echo "SoundEffect health:  http://127.0.0.1:$SOUNDEFFECT_PORT/v1/health"
echo "Stable Audio 3 SFX API:    http://$STABLE_AUDIO_3_SMALL_SFX_HOST:$STABLE_AUDIO_3_SMALL_SFX_PORT"
echo "Stable Audio 3 SFX health: http://127.0.0.1:$STABLE_AUDIO_3_SMALL_SFX_PORT/v1/health"
echo "Stable Audio 3 Medium API:    http://$STABLE_AUDIO_3_MEDIUM_HOST:$STABLE_AUDIO_3_MEDIUM_PORT"
echo "Stable Audio 3 Medium health: http://127.0.0.1:$STABLE_AUDIO_3_MEDIUM_PORT/v1/health"
echo "Qwen3-TTS API:       http://$QWEN3_TTS_HOST:$QWEN3_TTS_PORT"
echo "Qwen3-TTS health:    http://127.0.0.1:$QWEN3_TTS_PORT/v1/health"
echo "VoxCPM2 API:         http://$VOXCPM2_HOST:$VOXCPM2_PORT"
echo "VoxCPM2 health:      http://127.0.0.1:$VOXCPM2_PORT/v1/health"
echo "LongCat health:      http://127.0.0.1:$LONGCAT_AUDIODIT_PORT/v1/health"
echo "dots.tts-soar health: http://127.0.0.1:$DOTS_TTS_SOAR_PORT/v1/health"
echo "Qwen design route:   http://127.0.0.1:$PORT/v1/qwen/design"
echo "MOSS design route:   http://127.0.0.1:$PORT/v1/moss/design"
echo "Ming design route:   http://127.0.0.1:$PORT/v1/Ming/design"
echo "VoxCPM2 design route: http://127.0.0.1:$PORT/v1/voxcpm2/design"
echo "MiMo design route:   http://127.0.0.1:$PORT/v1/mimo/design"
echo "Step-Audio-EditX route: http://127.0.0.1:$PORT/v1/step-audio-editx/edit"
echo "SoundEffect route:   http://127.0.0.1:$SOUNDEFFECT_PORT/v1/generate"
echo "Stable Audio 3 SFX route: http://127.0.0.1:$STABLE_AUDIO_3_SMALL_SFX_PORT/v1/generate"
echo "Stable Audio 3 Medium route: http://127.0.0.1:$STABLE_AUDIO_3_MEDIUM_PORT/v1/generate"
echo "Qwen3-TTS synth:     http://127.0.0.1:$QWEN3_TTS_PORT/v2/synthesize"
echo "VoxCPM2/Ming synth:  http://127.0.0.1:$VOXCPM2_PORT/v2/synthesize (backend/model selects worker)"
echo "LongCat synth:       http://127.0.0.1:$LONGCAT_AUDIODIT_PORT/v2/synthesize"
echo "dots.tts-soar synth: http://127.0.0.1:$DOTS_TTS_SOAR_PORT/v2/synthesize"
echo "=================================================="

cd "$PROJECT_DIR"

main_pid=""
soundeffect_pid=""
stable_audio_3_small_sfx_pid=""
stable_audio_3_medium_pid=""
qwen3_tts_pid=""
voxcpm2_pid=""
longcat_audiodit_pid=""
dots_tts_soar_pid=""

cleanup() {
  local status=$?
  trap - INT TERM EXIT

  for pid in "$main_pid" "$soundeffect_pid" "$stable_audio_3_small_sfx_pid" "$stable_audio_3_medium_pid" "$qwen3_tts_pid" "$voxcpm2_pid" "$longcat_audiodit_pid" "$dots_tts_soar_pid"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done

  sleep 1

  for pid in "$main_pid" "$soundeffect_pid" "$stable_audio_3_small_sfx_pid" "$stable_audio_3_medium_pid" "$qwen3_tts_pid" "$voxcpm2_pid" "$longcat_audiodit_pid" "$dots_tts_soar_pid"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  done

  wait "$main_pid" 2>/dev/null || true
  wait "$soundeffect_pid" 2>/dev/null || true
  wait "$stable_audio_3_small_sfx_pid" 2>/dev/null || true
  wait "$stable_audio_3_medium_pid" 2>/dev/null || true
  wait "$qwen3_tts_pid" 2>/dev/null || true
  wait "$voxcpm2_pid" 2>/dev/null || true
  wait "$longcat_audiodit_pid" 2>/dev/null || true
  wait "$dots_tts_soar_pid" 2>/dev/null || true
  exit "$status"
}

trap cleanup INT TERM EXIT

setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/api.py" &
main_pid=$!
HOST="$SOUNDEFFECT_HOST" PORT="$SOUNDEFFECT_PORT" setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/soundeffect_api.py" &
soundeffect_pid=$!
HOST="$STABLE_AUDIO_3_SMALL_SFX_HOST" PORT="$STABLE_AUDIO_3_SMALL_SFX_PORT" setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/stable_audio_3_small_sfx_api.py" &
stable_audio_3_small_sfx_pid=$!
HOST="$STABLE_AUDIO_3_MEDIUM_HOST" PORT="$STABLE_AUDIO_3_MEDIUM_PORT" setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/stable_audio_3_medium_api.py" &
stable_audio_3_medium_pid=$!
HOST="$QWEN3_TTS_HOST" PORT="$QWEN3_TTS_PORT" setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/qwen3_tts_api.py" &
qwen3_tts_pid=$!
HOST="$VOXCPM2_HOST" PORT="$VOXCPM2_PORT" setsid conda run --no-capture-output -n "$VOXCPM2_CONDA_ENV" python "$API_DIR/voxcpm2_api.py" &
voxcpm2_pid=$!
HOST="$LONGCAT_AUDIODIT_HOST" PORT="$LONGCAT_AUDIODIT_PORT" setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/longcat_audiodit_api.py" &
longcat_audiodit_pid=$!
HOST="$DOTS_TTS_SOAR_HOST" PORT="$DOTS_TTS_SOAR_PORT" setsid conda run --no-capture-output -n "$CONDA_ENV" python "$API_DIR/dots_tts_soar_api.py" &
dots_tts_soar_pid=$!

wait -n "$main_pid" "$soundeffect_pid" "$stable_audio_3_small_sfx_pid" "$stable_audio_3_medium_pid" "$qwen3_tts_pid" "$voxcpm2_pid" "$longcat_audiodit_pid" "$dots_tts_soar_pid"
