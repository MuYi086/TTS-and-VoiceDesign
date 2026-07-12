#!/usr/bin/env python3
"""Generate one sound effect with OpenMOSS MOSS-SoundEffect v2.0.

Edit the configuration constants below, then run:

    conda run -n moss-soundEffect python soundEffect/test_moss_soundeffect_v2.py

The first invocation downloads the model weights into the Hugging Face cache and
may compile CUDA kernels.  It therefore takes substantially longer than later
runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Debug configuration — change these values directly for an individual test.
# ---------------------------------------------------------------------------
# Use the downloaded local model by default. Set MOSS_SOUNDEFFECT_MODEL_DIR
# before launching to override it, or edit LOCAL_MODEL_DIR for another disk.
LOCAL_MODEL_DIR = Path("/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0")
MODEL_ID = os.environ.get("MOSS_SOUNDEFFECT_MODEL_DIR", str(LOCAL_MODEL_DIR))
PROMPT = "门吱吱作响的声音，刺耳急促"
SECONDS = 10.0
NUM_INFERENCE_STEPS = 100
CFG_SCALE = 4.0
SIGMA_SHIFT = 5.0
SEED = 0
DEVICE = "cuda"
TORCH_DTYPE = "bfloat16"
OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "dog_barking_park.wav"
DISABLE_TORCHDYNAMO = True

# MOSS-SoundEffect v2.0 supports a maximum output duration of 30 seconds.
MAX_SECONDS = 30.0


def validate_configuration() -> None:
    """Fail before the expensive model load when a setting is invalid."""
    if not PROMPT.strip():
        raise ValueError("PROMPT must not be empty.")
    if not 0 < SECONDS <= MAX_SECONDS:
        raise ValueError(f"SECONDS must be in (0, {MAX_SECONDS}], got {SECONDS!r}.")
    if NUM_INFERENCE_STEPS <= 0:
        raise ValueError("NUM_INFERENCE_STEPS must be greater than zero.")
    if CFG_SCALE < 0:
        raise ValueError("CFG_SCALE must be greater than or equal to zero.")

    model_path = Path(MODEL_ID).expanduser()
    if model_path.is_absolute() and not model_path.is_dir():
        raise FileNotFoundError(
            f"Local MOSS-SoundEffect v2.0 model directory does not exist: {model_path}"
        )


def main() -> None:
    validate_configuration()

    # Upstream recommends disabling TorchDynamo when Triton/CUDA graph
    # compilation is unstable. Keep it configurable with the other test knobs.
    if DISABLE_TORCHDYNAMO:
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    try:
        import soundfile as sf
        import torch
        from moss_soundeffect_v2 import MossSoundEffectPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Could not import MOSS-SoundEffect v2.0 or one of its dependencies. "
            "Run soundEffect/run_moss_soundeffect_v2.sh so the correct Conda "
            f"environment is used. Active interpreter: {sys.executable}. "
            f"Original import error: {exc}"
        ) from exc

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE is set to 'cuda', but CUDA is not available. "
            "Use a CUDA-capable PyTorch installation, or explicitly change DEVICE "
            "and TORCH_DTYPE at the top of this file for an unsupported CPU test."
        )

    try:
        torch_dtype = getattr(torch, TORCH_DTYPE)
    except AttributeError as exc:
        raise ValueError(f"Unknown TORCH_DTYPE: {TORCH_DTYPE!r}") from exc

    print(f"[INFO] Loading {MODEL_ID} on {DEVICE} with {TORCH_DTYPE} ...")
    pipe = MossSoundEffectPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device=DEVICE,
        local_files_only=Path(MODEL_ID).expanduser().is_dir(),
    )

    # Keep the requested call arguments together and editable at the top.
    audio = pipe(
        prompt=PROMPT,
        seconds=SECONDS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
        sigma_shift=SIGMA_SHIFT,
        seed=SEED,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # torchaudio.save() requires TorchCodec in newer Torchaudio releases.
    # SoundFile is already a core MOSS v2 dependency and writes the generated
    # (B, C, T) tensor without that optional runtime dependency.
    waveform = audio[0].detach().to(torch.float32).cpu().transpose(0, 1).numpy()
    sf.write(str(OUTPUT_PATH), waveform, pipe.sample_rate)
    print(f"[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
