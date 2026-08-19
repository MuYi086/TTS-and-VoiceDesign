#!/usr/bin/env python3
"""使用 OpenMOSS MOSS-SoundEffect v2.0 生成一段音效。

修改下方配置常量后运行：

    uv run --project moss_soundEffect python soundEffect/test_moss_soundeffect_v2.py

首次运行会将模型权重下载到 Hugging Face 缓存，并可能编译 CUDA 内核，
因此耗时通常明显长于后续运行。
"""

# 这是唯一面向真实 GPU 的示例脚本；普通回归测试不会调用它。

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 调试配置：单独测试时可以直接修改这些值。
# ---------------------------------------------------------------------------
# 默认使用已下载的本地模型。启动前设置 MOSS_SOUNDEFFECT_MODEL_DIR 可覆盖路径，
# 也可以修改 LOCAL_MODEL_DIR 指向其他磁盘。
LOCAL_MODEL_DIR = Path("/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0")
MODEL_ID = os.environ.get("MOSS_SOUNDEFFECT_MODEL_DIR", str(LOCAL_MODEL_DIR))
LOCAL_CODE_DIR = Path.home() / "tts-depency/MOSS-TTS"
CODE_DIR = Path(os.environ.get("MOSS_SOUNDEFFECT_CODE_PATH", str(LOCAL_CODE_DIR))).expanduser()
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

# MOSS-SoundEffect v2.0 支持的最大输出时长为 30 秒。
MAX_SECONDS = 30.0


def validate_configuration() -> None:
    """配置无效时在加载耗时模型前立即失败。"""
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
    if not (CODE_DIR / "moss_soundeffect_v2" / "__init__.py").is_file():
        raise FileNotFoundError(f"Local MOSS-TTS source checkout is incomplete: {CODE_DIR}")


def main() -> None:
    validate_configuration()
    sys.path.insert(0, str(CODE_DIR.resolve()))

    # 当 Triton/CUDA Graph 编译不稳定时，上游建议禁用 TorchDynamo；这里与其他
    # 测试开关保持一致，允许通过配置控制。
    if DISABLE_TORCHDYNAMO:
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    try:
        import soundfile as sf
        import torch
        from moss_soundeffect_v2 import MossSoundEffectPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Could not import MOSS-SoundEffect v2.0 or one of its dependencies. "
            "Run soundEffect/run_moss_soundeffect_v2.sh so the correct uv "
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

    # 将本次调用参数集中放在上方，便于直接编辑。
    audio = pipe(
        prompt=PROMPT,
        seconds=SECONDS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
        sigma_shift=SIGMA_SHIFT,
        seed=SEED,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 新版 Torchaudio 的 torchaudio.save() 需要 TorchCodec；SoundFile 已是 MOSS v2
    # 的核心依赖，可以直接写出生成的 (B, C, T) 张量，不依赖这个可选运行时。
    waveform = audio[0].detach().to(torch.float32).cpu().transpose(0, 1).numpy()
    sf.write(str(OUTPUT_PATH), waveform, pipe.sample_rate)
    print(f"[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
