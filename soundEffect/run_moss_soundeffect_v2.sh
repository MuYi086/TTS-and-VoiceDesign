#!/usr/bin/env bash
# 一键启动本地 MOSS-SoundEffect v2.0 GPU smoke test。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_PROJECT_DIR="${MOSS_SOUNDEFFECT_PROJECT_DIR:-${PROJECT_DIR}/moss_soundEffect}"
CODE_PATH="${MOSS_SOUNDEFFECT_CODE_PATH:-${HOME}/tts-depency/MOSS-TTS}"
MODEL_DIR="${MOSS_SOUNDEFFECT_MODEL_DIR:-/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0}"

# smoke test 只允许使用已经存在的本地权重，避免脚本偷偷下载大文件。
if [[ ! -f "${MODEL_DIR}/model_index.json" || ! -f "${MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors" || ! -f "${MODEL_DIR}/vae/vae_128d_48k.pth" ]]; then
  echo "[ERROR] Local MOSS-SoundEffect v2.0 weights are incomplete or missing: ${MODEL_DIR}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is not on PATH." >&2
  exit 1
fi

# 运行时依赖必须来自仓库约定的本地 MOSS-TTS checkout。
if [[ ! -f "${CODE_PATH}/moss_soundeffect_v2/__init__.py" ]]; then
  echo "[ERROR] MOSS-TTS source checkout is incomplete: ${CODE_PATH}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
export MOSS_SOUNDEFFECT_MODEL_DIR="${MODEL_DIR}"
export MOSS_SOUNDEFFECT_CODE_PATH="${CODE_PATH}"

# --no-sync 确保 smoke test 不修改锁定环境，只验证当前可用安装。
echo "[INFO] uv project: ${UV_PROJECT_DIR}"
echo "[INFO] MOSS-TTS source: ${CODE_PATH}"
echo "[INFO] Local model directory: ${MODEL_DIR}"
exec uv run --no-sync --project "${UV_PROJECT_DIR}" \
  python "${SCRIPT_DIR}/test_moss_soundeffect_v2.py"
