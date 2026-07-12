#!/usr/bin/env bash
# One-command launcher for the local MOSS-SoundEffect v2.0 test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-moss-soundEffect}"
MODEL_DIR="${MOSS_SOUNDEFFECT_MODEL_DIR:-/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0}"

if [[ ! -f "${MODEL_DIR}/model_index.json" || ! -f "${MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors" || ! -f "${MODEL_DIR}/vae/vae_128d_48k.pth" ]]; then
  echo "[ERROR] Local MOSS-SoundEffect v2.0 weights are incomplete or missing: ${MODEL_DIR}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not on PATH. Run this script from a Conda-enabled shell." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
export MOSS_SOUNDEFFECT_MODEL_DIR="${MODEL_DIR}"

echo "[INFO] Conda environment: ${CONDA_ENV_NAME}"
echo "[INFO] Local model directory: ${MODEL_DIR}"
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
  python "${SCRIPT_DIR}/test_moss_soundeffect_v2.py"
