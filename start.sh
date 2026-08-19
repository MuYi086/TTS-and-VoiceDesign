#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$PROJECT_DIR/main"
STORAGE_DIR="${STORAGE_DIR:-$PROJECT_DIR/storage}"
export STORAGE_DIR
MIMO_TTS_PROJECT_DIR="${MIMO_TTS_PROJECT_DIR:-$PROJECT_DIR/mimo_tts}"
# The control-plane wrappers use the Qwen3-TTS uv environment, while
# heavyweight workers use their model-specific environments below.
MOSS_SOUNDEFFECT_PROJECT_DIR="${MOSS_SOUNDEFFECT_PROJECT_DIR:-$PROJECT_DIR/moss_soundEffect}"
QWEN3_TTS_PROJECT_DIR="${QWEN3_TTS_PROJECT_DIR:-$PROJECT_DIR/qwen3_tts}"
QWEN3_VOICEDESIGN_PROJECT_DIR="${QWEN3_VOICEDESIGN_PROJECT_DIR:-$PROJECT_DIR/qwen3_voiceDesign}"
MOSS_VOICEGENERATOR_PROJECT_DIR="${MOSS_VOICEGENERATOR_PROJECT_DIR:-$PROJECT_DIR/moss_voiceGenerator}"
STEP_AUDIO_EDITX_PROJECT_DIR="${STEP_AUDIO_EDITX_PROJECT_DIR:-$PROJECT_DIR/Step_Audio_EditX}"
LONGCAT_AUDIODIT_PROJECT_DIR="${LONGCAT_AUDIODIT_PROJECT_DIR:-$PROJECT_DIR/LongCat_AudioDiT_3.5B_bf16}"
DOTS_TTS_SOAR_PROJECT_DIR="${DOTS_TTS_SOAR_PROJECT_DIR:-$PROJECT_DIR/dots_tts_soar}"
STABLE_AUDIO_3_MEDIUM_PROJECT_DIR="${STABLE_AUDIO_3_MEDIUM_PROJECT_DIR:-$PROJECT_DIR/stable_audio_3_medium}"
ACESTEP_PROJECT_DIR="${ACESTEP_PROJECT_DIR:-$PROJECT_DIR/ace_step_1_5}"
VOXCPM2_PROJECT_DIR="${VOXCPM2_PROJECT_DIR:-$PROJECT_DIR/voxcpm2}"

export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export QWEN_VOICEDESIGN_MODEL_DIR="${QWEN_VOICEDESIGN_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
export MOSS_VOICEGENERATOR_MODEL_DIR="${MOSS_VOICEGENERATOR_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator}"
export MOSS_AUDIO_TOKENIZER_PATH="${MOSS_AUDIO_TOKENIZER_PATH:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer}"
export MOSS_VOICEGENERATOR_REQUEST_TIMEOUT="${MOSS_VOICEGENERATOR_REQUEST_TIMEOUT:-900}"
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
export MOSS_SOUNDEFFECT_CODE_PATH="${MOSS_SOUNDEFFECT_CODE_PATH:-$HOME/tts-depency/MOSS-TTS}"
export MOSS_SOUNDEFFECT_MODEL_DIR="${MOSS_SOUNDEFFECT_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0}"
export MOSS_SOUNDEFFECT_DEVICE="${MOSS_SOUNDEFFECT_DEVICE:-cuda}"
export MOSS_SOUNDEFFECT_DTYPE="${MOSS_SOUNDEFFECT_DTYPE:-bfloat16}"
export MOSS_SOUNDEFFECT_DEFAULT_SECONDS="${MOSS_SOUNDEFFECT_DEFAULT_SECONDS:-10}"
export MOSS_SOUNDEFFECT_DEFAULT_STEPS="${MOSS_SOUNDEFFECT_DEFAULT_STEPS:-100}"
export MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE="${MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE:-4.0}"
export MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT="${MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT:-5.0}"
export MOSS_SOUNDEFFECT_DEFAULT_SEED="${MOSS_SOUNDEFFECT_DEFAULT_SEED:-0}"
export MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO="${MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO:-1}"
export STABLE_AUDIO_3_REPO_PATH="${STABLE_AUDIO_3_REPO_PATH:-$HOME/tts-depency/stable-audio-3}"
export STABLE_AUDIO_3_MEDIUM_MODEL_DIR="${STABLE_AUDIO_3_MEDIUM_MODEL_DIR:-$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium}"
export STABLE_AUDIO_3_MEDIUM_DEVICE="${STABLE_AUDIO_3_MEDIUM_DEVICE:-cuda}"
export STABLE_AUDIO_3_MEDIUM_DTYPE="${STABLE_AUDIO_3_MEDIUM_DTYPE:-float16}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS="${STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS:-7}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS="${STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS:-8}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE="${STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE:-1.0}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED="${STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED:--1}"
export STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT="${STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT:-900}"
export STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN="${STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN:-0}"
export ACESTEP_MODEL_DIR="${ACESTEP_MODEL_DIR:-$HF_MIRROR_DIR/ACE-Step/acestep-v15-xl-turbo-diffusers}"
export BGM_STORAGE_DIR="${BGM_STORAGE_DIR:-$STORAGE_DIR/bgm}"
export ACESTEP_OUTPUT_DIR="${ACESTEP_OUTPUT_DIR:-$BGM_STORAGE_DIR}"
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-cuda}"
export ACESTEP_DTYPE="${ACESTEP_DTYPE:-bfloat16}"
export ACESTEP_OFFLOAD="${ACESTEP_OFFLOAD:-model}"
export ACESTEP_VAE_TILING="${ACESTEP_VAE_TILING:-1}"
export ACESTEP_DEFAULT_SECONDS="${ACESTEP_DEFAULT_SECONDS:-60}"
export ACESTEP_DEFAULT_STEPS="${ACESTEP_DEFAULT_STEPS:-8}"
export ACESTEP_DEFAULT_SEED="${ACESTEP_DEFAULT_SEED:--1}"
export ACESTEP_REQUEST_TIMEOUT="${ACESTEP_REQUEST_TIMEOUT:-1800}"
export QWEN3_TTS_MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
export VOXCPM2_MODEL_DIR="${VOXCPM2_MODEL_DIR:-$HF_MIRROR_DIR/openbmb/VoxCPM2}"
export QWEN_LIBS="${QWEN_LIBS:-$QWEN3_TTS_PROJECT_DIR/vendor/qwen_libs}"
export TIMBRE_STORAGE_DIR="${TIMBRE_STORAGE_DIR:-$STORAGE_DIR/timbre}"
export SOUNDEFFECT_STORAGE_DIR="${SOUNDEFFECT_STORAGE_DIR:-$STORAGE_DIR/soundEffect}"
export CLONE_STORAGE_DIR="${CLONE_STORAGE_DIR:-$STORAGE_DIR/clone}"
export STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR="${STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR:-$SOUNDEFFECT_STORAGE_DIR}"
export QWEN3_TTS_OUTPUT_DIR="${QWEN3_TTS_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export VOXCPM2_OUTPUT_DIR="${VOXCPM2_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export LONGCAT_AUDIODIT_OUTPUT_DIR="${LONGCAT_AUDIODIT_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export DOTS_TTS_SOAR_OUTPUT_DIR="${DOTS_TTS_SOAR_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export STEP_AUDIO_EDITX_OUTPUT_DIR="${STEP_AUDIO_EDITX_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export PROMPTS_DIR="${PROMPTS_DIR:-$CLONE_STORAGE_DIR}"
export RUNTIME_CACHE_DIR="${RUNTIME_CACHE_DIR:-$STORAGE_DIR/.cache/runtime}"
export GPU_LOCK_FILE="${GPU_LOCK_FILE:-$RUNTIME_CACHE_DIR/gpu-runtime.lock}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
# LongCat 的克隆默认值集中在对应 API 顶部；Qwen3-TTS 的默认值
# 集中在 qwen3_tts/main.py；
# 此处只保留环境、模型路径等启动路由配置，避免覆盖 API 内可直接调试的值。
# VoxCPM2 的默认生成参数集中在 voxcpm2/main.py；此处不再写入默认值，
# 因此直接修改 API 顶部常量即可生效。外部显式设置的 VOXCPM2_* 环境变量会原样继承并覆盖 API 默认值。
# VoxCPM2 只使用仓库内独立 uv 项目，旧 Conda wrapper 已移除。
export LONGCAT_AUDIODIT_MODEL_DIR="${LONGCAT_AUDIODIT_MODEL_DIR:-$HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16}"
export LONGCAT_AUDIODIT_REPO_PATH="${LONGCAT_AUDIODIT_REPO_PATH:-$HOME/tts-depency/LongCat-AudioDiT}"
export LONGCAT_AUDIODIT_TOKENIZER_PATH="${LONGCAT_AUDIODIT_TOKENIZER_PATH:-$HF_MIRROR_DIR/google/umt5-base}"
export DOTS_TTS_SOAR_MODEL_DIR="${DOTS_TTS_SOAR_MODEL_DIR:-$HF_MIRROR_DIR/rednote-hilab/dots.tts-soar}"
export QWEN3_TTS_USE_QWEN_LIBS="${QWEN3_TTS_USE_QWEN_LIBS:-0}"
export MOSS_SOUNDEFFECT_REQUEST_TIMEOUT="${MOSS_SOUNDEFFECT_REQUEST_TIMEOUT:-600}"
export CUDA_RELEASE_DELAY="${CUDA_RELEASE_DELAY:-2.0}"
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
export MIMO_TTS_HOST="${MIMO_TTS_HOST:-$HOST}"
export MIMO_TTS_PORT="${MIMO_TTS_PORT:-8303}"
export MIMO_TTS_PROXY_URL="${MIMO_TTS_PROXY_URL:-http://127.0.0.1:$MIMO_TTS_PORT/v1/mimo/timbre}"
export STEP_AUDIO_EDITX_HOST="${STEP_AUDIO_EDITX_HOST:-$HOST}"
export STEP_AUDIO_EDITX_PORT="${STEP_AUDIO_EDITX_PORT:-8331}"
export SOUNDEFFECT_HOST="${SOUNDEFFECT_HOST:-$HOST}"
export SOUNDEFFECT_PORT="${SOUNDEFFECT_PORT:-8312}"
export STABLE_AUDIO_3_MEDIUM_HOST="${STABLE_AUDIO_3_MEDIUM_HOST:-$HOST}"
export STABLE_AUDIO_3_MEDIUM_PORT="${STABLE_AUDIO_3_MEDIUM_PORT:-8311}"
export QWEN3_TTS_HOST="${QWEN3_TTS_HOST:-$HOST}"
export QWEN3_TTS_PORT="${QWEN3_TTS_PORT:-8321}"
export VOXCPM2_HOST="${VOXCPM2_HOST:-$HOST}"
export VOXCPM2_PORT="${VOXCPM2_PORT:-8322}"
export LONGCAT_AUDIODIT_HOST="${LONGCAT_AUDIODIT_HOST:-$HOST}"
export LONGCAT_AUDIODIT_PORT="${LONGCAT_AUDIODIT_PORT:-8323}"
export DOTS_TTS_SOAR_HOST="${DOTS_TTS_SOAR_HOST:-$HOST}"
export DOTS_TTS_SOAR_PORT="${DOTS_TTS_SOAR_PORT:-8324}"
export QWEN_VOICEDESIGN_HOST="${QWEN_VOICEDESIGN_HOST:-$HOST}"
export QWEN_VOICEDESIGN_PORT="${QWEN_VOICEDESIGN_PORT:-8301}"
export MOSS_VOICEGENERATOR_HOST="${MOSS_VOICEGENERATOR_HOST:-$HOST}"
export MOSS_VOICEGENERATOR_PORT="${MOSS_VOICEGENERATOR_PORT:-8302}"
export ACESTEP_HOST="${ACESTEP_HOST:-$HOST}"
export ACESTEP_PORT="${ACESTEP_PORT:-8313}"

export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUNTIME_CACHE_DIR/hf_modules}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$RUNTIME_CACHE_DIR/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_CACHE_DIR/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_CACHE_DIR/xdg}"
mkdir -p "$TIMBRE_STORAGE_DIR" "$SOUNDEFFECT_STORAGE_DIR" "$BGM_STORAGE_DIR" "$CLONE_STORAGE_DIR" "$PROMPTS_DIR" "$HF_MODULES_CACHE" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$(dirname "$GPU_LOCK_FILE")"

echo "=================================================="
echo "   Unitale AI local backend"
echo "=================================================="
echo "Control-plane uv project: $QWEN3_TTS_PROJECT_DIR"
echo "MiMo TTS uv project:      $MIMO_TTS_PROJECT_DIR"
echo "MOSS VoiceGenerator:  $MOSS_VOICEGENERATOR_MODEL_DIR"
echo "Step-Audio-EditX uv project: $STEP_AUDIO_EDITX_PROJECT_DIR"
echo "Step-Audio-EditX API: http://$STEP_AUDIO_EDITX_HOST:$STEP_AUDIO_EDITX_PORT"
echo "Step-Audio-EditX model: $STEP_AUDIO_EDITX_MODEL_DIR"
echo "Step-Audio tokenizer: $STEP_AUDIO_TOKENIZER_PATH"
echo "Step-Audio-EditX code: $STEP_AUDIO_EDITX_CODE_PATH"
echo "SoundEffect uv project: $MOSS_SOUNDEFFECT_PROJECT_DIR"
echo "SoundEffect source:  $MOSS_SOUNDEFFECT_CODE_PATH"
echo "SoundEffect model:   $MOSS_SOUNDEFFECT_MODEL_DIR"
echo "SoundEffect device:  $MOSS_SOUNDEFFECT_DEVICE ($MOSS_SOUNDEFFECT_DTYPE)"
echo "Stable Audio 3 Medium project: $STABLE_AUDIO_3_MEDIUM_PROJECT_DIR"
echo "Stable Audio 3 Medium model:  $STABLE_AUDIO_3_MEDIUM_MODEL_DIR"
echo "Stable Audio 3 source:        $STABLE_AUDIO_3_REPO_PATH"
echo "Stable Audio 3 Medium device: $STABLE_AUDIO_3_MEDIUM_DEVICE ($STABLE_AUDIO_3_MEDIUM_DTYPE)"
echo "ACE-Step uv project:           $ACESTEP_PROJECT_DIR"
echo "ACE-Step model:                $ACESTEP_MODEL_DIR"
echo "ACE-Step BGM storage:          $BGM_STORAGE_DIR"
echo "ACE-Step runtime:              $ACESTEP_DTYPE / offload=$ACESTEP_OFFLOAD"
echo "Qwen3-TTS uv project:  $QWEN3_TTS_PROJECT_DIR"
echo "Qwen3-TTS model:     $QWEN3_TTS_MODEL_DIR"
echo "Qwen VoiceDesign uv project: $QWEN3_VOICEDESIGN_PROJECT_DIR"
echo "Qwen VoiceDesign model:      $QWEN_VOICEDESIGN_MODEL_DIR"
echo "MOSS VoiceGenerator uv project: $MOSS_VOICEGENERATOR_PROJECT_DIR"
echo "MOSS VoiceGenerator model:      $MOSS_VOICEGENERATOR_MODEL_DIR"
echo "MOSS Audio tokenizer:           $MOSS_AUDIO_TOKENIZER_PATH"
echo "VoxCPM2 uv project:  $VOXCPM2_PROJECT_DIR"
echo "VoxCPM2 model:       $VOXCPM2_MODEL_DIR"
echo "LongCat uv project:  $LONGCAT_AUDIODIT_PROJECT_DIR"
echo "LongCat model:       $LONGCAT_AUDIODIT_MODEL_DIR"
echo "LongCat repo:        $LONGCAT_AUDIODIT_REPO_PATH"
echo "LongCat tokenizer:   $LONGCAT_AUDIODIT_TOKENIZER_PATH"
echo "LongCat config:      managed by $LONGCAT_AUDIODIT_PROJECT_DIR/main.py"
echo "dots.tts-soar uv project: $DOTS_TTS_SOAR_PROJECT_DIR"
echo "dots.tts-soar model: $DOTS_TTS_SOAR_MODEL_DIR"
echo "dots.tts-soar config: managed by $DOTS_TTS_SOAR_PROJECT_DIR/main.py"
echo "VoxCPM2 config:      managed by $VOXCPM2_PROJECT_DIR/main.py"
echo "Qwen3-TTS config:    managed by qwen3_tts/main.py"
echo "Qwen sidecar libs:   $QWEN_LIBS"
echo "MiMo base URL:       $MIMO_BASE_URL"
echo "MiMo model:          $MIMO_MODEL"
echo "MiMo API key:        $([[ -n "${MIMO_API_KEY:-}" ]] && echo configured || echo missing)"
echo "Storage root:        $STORAGE_DIR"
echo "Timbre storage:      $TIMBRE_STORAGE_DIR"
echo "SoundEffect storage: $SOUNDEFFECT_STORAGE_DIR"
echo "Clone storage:       $CLONE_STORAGE_DIR"
echo "Reference audio dir: $PROMPTS_DIR"
echo "HF modules cache:    $HF_MODULES_CACHE"
echo "GPU lock file:       $GPU_LOCK_FILE"
echo "Main API:            http://$HOST:$PORT"
echo "Control route:       http://127.0.0.1:$PORT/v1/control"
echo "MiMo TTS API:        http://$MIMO_TTS_HOST:$MIMO_TTS_PORT"
echo "MiMo TTS health:     http://127.0.0.1:$MIMO_TTS_PORT/v1/health"
echo "SoundEffect API:     http://$SOUNDEFFECT_HOST:$SOUNDEFFECT_PORT"
echo "SoundEffect health:  http://127.0.0.1:$SOUNDEFFECT_PORT/v1/health"
echo "Stable Audio 3 Medium API:    http://$STABLE_AUDIO_3_MEDIUM_HOST:$STABLE_AUDIO_3_MEDIUM_PORT"
echo "Stable Audio 3 Medium health: http://127.0.0.1:$STABLE_AUDIO_3_MEDIUM_PORT/v1/health"
echo "ACE-Step BGM API:             http://$ACESTEP_HOST:$ACESTEP_PORT"
echo "ACE-Step BGM health:           http://127.0.0.1:$ACESTEP_PORT/v1/health"
echo "Qwen3-TTS API:       http://$QWEN3_TTS_HOST:$QWEN3_TTS_PORT"
echo "Qwen3-TTS health:    http://127.0.0.1:$QWEN3_TTS_PORT/v1/health"
echo "Qwen VoiceDesign API: http://$QWEN_VOICEDESIGN_HOST:$QWEN_VOICEDESIGN_PORT"
echo "Qwen VoiceDesign health: http://127.0.0.1:$QWEN_VOICEDESIGN_PORT/v1/health"
echo "MOSS VoiceGenerator API: http://$MOSS_VOICEGENERATOR_HOST:$MOSS_VOICEGENERATOR_PORT"
echo "MOSS VoiceGenerator health: http://127.0.0.1:$MOSS_VOICEGENERATOR_PORT/v1/health"
echo "VoxCPM2 API:         http://$VOXCPM2_HOST:$VOXCPM2_PORT"
echo "VoxCPM2 health:      http://127.0.0.1:$VOXCPM2_PORT/v1/health"
echo "LongCat health:      http://127.0.0.1:$LONGCAT_AUDIODIT_PORT/v1/health"
echo "dots.tts-soar health: http://127.0.0.1:$DOTS_TTS_SOAR_PORT/v1/health"
echo "MOSS timbre route:   http://127.0.0.1:$MOSS_VOICEGENERATOR_PORT/v1/moss/timbre"
echo "MiMo timbre route:   http://127.0.0.1:$MIMO_TTS_PORT/v1/mimo/timbre"
echo "Step-Audio-EditX route: http://127.0.0.1:$STEP_AUDIO_EDITX_PORT/v1/stepAudioEditx/edit"
echo "MOSS sound-effect route: http://127.0.0.1:$SOUNDEFFECT_PORT/v1/moss/soundEffect"
echo "Stable Audio 3 Medium route: http://127.0.0.1:$STABLE_AUDIO_3_MEDIUM_PORT/v1/stableAudio/soundEffect"
echo "ACE-Step BGM route:     http://127.0.0.1:$ACESTEP_PORT/v1/aceStep/bgm"
echo "Qwen timbre route:   http://127.0.0.1:$QWEN_VOICEDESIGN_PORT/v1/qwen/timbre"
echo "Qwen3-TTS clone:     http://127.0.0.1:$QWEN3_TTS_PORT/v1/qwen/clone"
echo "VoxCPM2 clone:       http://127.0.0.1:$VOXCPM2_PORT/v1/voxcpm2/clone"
echo "LongCat clone:       http://127.0.0.1:$LONGCAT_AUDIODIT_PORT/v1/longCat/clone"
echo "dots.tts-soar clone: http://127.0.0.1:$DOTS_TTS_SOAR_PORT/v2/dotsTTS/clone"
echo ""
echo "模型接口与端口映射"
printf '%-24s %-6s %s\n' '服务' '端口' '最终接口'
printf '%-24s %-6s %s\n' '控制面' "$PORT" '/v1/control'
printf '%-24s %-6s %s\n' 'Qwen3-TTS VoiceDesign' "$QWEN_VOICEDESIGN_PORT" '/v1/qwen/timbre'
printf '%-24s %-6s %s\n' 'MOSS VoiceGenerator' "$MOSS_VOICEGENERATOR_PORT" '/v1/moss/timbre'
printf '%-24s %-6s %s\n' 'MiMo TTS VoiceDesign' "$MIMO_TTS_PORT" '/v1/mimo/timbre'
printf '%-24s %-6s %s\n' 'Stable Audio 3 Medium' "$STABLE_AUDIO_3_MEDIUM_PORT" '/v1/stableAudio/soundEffect'
printf '%-24s %-6s %s\n' 'ACE-Step 1.5 XL Turbo' "$ACESTEP_PORT" '/v1/aceStep/bgm'
printf '%-24s %-6s %s\n' 'MOSS-SoundEffect v2' "$SOUNDEFFECT_PORT" '/v1/moss/soundEffect'
printf '%-24s %-6s %s\n' 'Qwen3-TTS Base' "$QWEN3_TTS_PORT" '/v1/qwen/clone'
printf '%-24s %-6s %s\n' 'VoxCPM2' "$VOXCPM2_PORT" '/v1/voxcpm2/clone'
printf '%-24s %-6s %s\n' 'LongCat-AudioDiT' "$LONGCAT_AUDIODIT_PORT" '/v1/longCat/clone'
printf '%-24s %-6s %s\n' 'dots.tts-soar' "$DOTS_TTS_SOAR_PORT" '/v2/dotsTTS/clone'
printf '%-24s %-6s %s\n' 'Step-Audio-EditX' "$STEP_AUDIO_EDITX_PORT" '/v1/stepAudioEditx/edit'
echo "=================================================="

cd "$PROJECT_DIR"

main_pid=""
mimo_tts_pid=""
soundeffect_pid=""
stable_audio_3_medium_pid=""
acestep_pid=""
qwen3_tts_pid=""
voxcpm2_pid=""
longcat_audiodit_pid=""
dots_tts_soar_pid=""
qwen_voicedesign_pid=""
moss_voicegenerator_pid=""
step_audio_editx_pid=""

cleanup() {
  local status=$?
  trap - INT TERM EXIT

  for pid in "$main_pid" "$mimo_tts_pid" "$soundeffect_pid" "$stable_audio_3_medium_pid" "$acestep_pid" "$qwen3_tts_pid" "$qwen_voicedesign_pid" "$moss_voicegenerator_pid" "$step_audio_editx_pid" "$voxcpm2_pid" "$longcat_audiodit_pid" "$dots_tts_soar_pid"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done

  sleep 1

  for pid in "$main_pid" "$mimo_tts_pid" "$soundeffect_pid" "$stable_audio_3_medium_pid" "$acestep_pid" "$qwen3_tts_pid" "$qwen_voicedesign_pid" "$moss_voicegenerator_pid" "$step_audio_editx_pid" "$voxcpm2_pid" "$longcat_audiodit_pid" "$dots_tts_soar_pid"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  done

  wait "$main_pid" 2>/dev/null || true
  wait "$mimo_tts_pid" 2>/dev/null || true
  wait "$soundeffect_pid" 2>/dev/null || true
  wait "$stable_audio_3_medium_pid" 2>/dev/null || true
  wait "$acestep_pid" 2>/dev/null || true
  wait "$qwen3_tts_pid" 2>/dev/null || true
  wait "$qwen_voicedesign_pid" 2>/dev/null || true
  wait "$moss_voicegenerator_pid" 2>/dev/null || true
  wait "$step_audio_editx_pid" 2>/dev/null || true
  wait "$voxcpm2_pid" 2>/dev/null || true
  wait "$longcat_audiodit_pid" 2>/dev/null || true
  wait "$dots_tts_soar_pid" 2>/dev/null || true
  exit "$status"
}

trap cleanup INT TERM EXIT

# The main control plane owns the 8300 control routes and uses the
# Qwen3-TTS uv environment only for its lightweight HTTP dependencies.
setsid uv run --no-sync --project "$QWEN3_TTS_PROJECT_DIR" python "$MAIN_DIR/main.py" &
main_pid=$!
# MiMo is a cloud-backed VoiceDesign service with its own health and cache.
MIMO_TTS_HOST="$MIMO_TTS_HOST" MIMO_TTS_PORT="$MIMO_TTS_PORT" \
  HOST="$MIMO_TTS_HOST" PORT="$MIMO_TTS_PORT" \
  setsid uv run --no-sync --project "$MIMO_TTS_PROJECT_DIR" \
  python "$MIMO_TTS_PROJECT_DIR/main.py" &
mimo_tts_pid=$!
# MOSS-SoundEffect uv 服务：使用最终约定的 8312 端口和路由。
HOST="$SOUNDEFFECT_HOST" PORT="$SOUNDEFFECT_PORT" \
  setsid uv run --no-sync --project "$MOSS_SOUNDEFFECT_PROJECT_DIR" \
  python "$MOSS_SOUNDEFFECT_PROJECT_DIR/main.py" &
soundeffect_pid=$!
# Stable Audio 3 Medium is fully migrated to its standalone uv project.
HOST="$STABLE_AUDIO_3_MEDIUM_HOST" PORT="$STABLE_AUDIO_3_MEDIUM_PORT" \
  setsid uv run --no-sync --project "$STABLE_AUDIO_3_MEDIUM_PROJECT_DIR" \
  python "$STABLE_AUDIO_3_MEDIUM_PROJECT_DIR/main.py" &
stable_audio_3_medium_pid=$!
# ACE-Step 1.5 is a separate one-shot-worker uv service.  Dependencies must be
# synchronized manually before startup; --no-sync keeps startup offline.
HOST="$ACESTEP_HOST" PORT="$ACESTEP_PORT" \
  setsid uv run --no-sync --project "$ACESTEP_PROJECT_DIR" \
  python "$ACESTEP_PROJECT_DIR/main.py" &
acestep_pid=$!
# Qwen3-TTS uv 服务：使用最终约定的 8321 端口和克隆路由。
HOST="$QWEN3_TTS_HOST" PORT="$QWEN3_TTS_PORT" setsid uv run --project "$QWEN3_TTS_PROJECT_DIR" python "$QWEN3_TTS_PROJECT_DIR/main.py" &
qwen3_tts_pid=$!
# Qwen3-TTS VoiceDesign 独立 uv 服务：使用最终约定的 8301 端口。
QWEN_VOICEDESIGN_HOST="$QWEN_VOICEDESIGN_HOST" QWEN_VOICEDESIGN_PORT="$QWEN_VOICEDESIGN_PORT" \
  HOST="$QWEN_VOICEDESIGN_HOST" PORT="$QWEN_VOICEDESIGN_PORT" \
  setsid uv run --project "$QWEN3_VOICEDESIGN_PROJECT_DIR" \
  python "$QWEN3_VOICEDESIGN_PROJECT_DIR/main.py" &
qwen_voicedesign_pid=$!
# MOSS VoiceGenerator 独立 uv 服务：使用最终约定的 8302 端口。
MOSS_VOICEGENERATOR_HOST="$MOSS_VOICEGENERATOR_HOST" MOSS_VOICEGENERATOR_PORT="$MOSS_VOICEGENERATOR_PORT" \
  HOST="$MOSS_VOICEGENERATOR_HOST" PORT="$MOSS_VOICEGENERATOR_PORT" \
  setsid uv run --project "$MOSS_VOICEGENERATOR_PROJECT_DIR" \
  python "$MOSS_VOICEGENERATOR_PROJECT_DIR/main.py" &
moss_voicegenerator_pid=$!
# Step-Audio-EditX 独立 uv 服务：完整提供上传、检查和编辑接口。
# 依赖由部署前手动执行 `uv sync --project Step_Audio_EditX --locked`；启动阶段不再联网解析。
STEP_AUDIO_EDITX_HOST="$STEP_AUDIO_EDITX_HOST" STEP_AUDIO_EDITX_PORT="$STEP_AUDIO_EDITX_PORT" \
  HOST="$STEP_AUDIO_EDITX_HOST" PORT="$STEP_AUDIO_EDITX_PORT" \
  setsid uv run --no-sync --project "$STEP_AUDIO_EDITX_PROJECT_DIR" \
  python "$STEP_AUDIO_EDITX_PROJECT_DIR/main.py" &
step_audio_editx_pid=$!
HOST="$VOXCPM2_HOST" PORT="$VOXCPM2_PORT" \
  setsid uv run --no-sync --project "$VOXCPM2_PROJECT_DIR" \
  python "$VOXCPM2_PROJECT_DIR/main.py" &
voxcpm2_pid=$!
# LongCat-AudioDiT uv 服务：使用最终约定的 8323 端口。
HOST="$LONGCAT_AUDIODIT_HOST" PORT="$LONGCAT_AUDIODIT_PORT" \
  setsid uv run --no-sync --project "$LONGCAT_AUDIODIT_PROJECT_DIR" \
  python "$LONGCAT_AUDIODIT_PROJECT_DIR/main.py" &
longcat_audiodit_pid=$!
# dots.tts-soar 使用独立 uv 项目，使用最终约定的 8324 端口。
HOST="$DOTS_TTS_SOAR_HOST" PORT="$DOTS_TTS_SOAR_PORT" \
  setsid uv run --no-sync --project "$DOTS_TTS_SOAR_PROJECT_DIR" \
    python "$DOTS_TTS_SOAR_PROJECT_DIR/main.py" &
dots_tts_soar_pid=$!

wait -n "$main_pid" "$mimo_tts_pid" "$soundeffect_pid" "$stable_audio_3_medium_pid" "$acestep_pid" "$qwen3_tts_pid" "$qwen_voicedesign_pid" "$moss_voicegenerator_pid" "$step_audio_editx_pid" "$voxcpm2_pid" "$longcat_audiodit_pid" "$dots_tts_soar_pid"
