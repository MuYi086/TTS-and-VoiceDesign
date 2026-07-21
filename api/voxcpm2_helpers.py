"""Shared VoxCPM2 runtime helpers used by the one-shot worker.

Runtime-heavy dependencies are imported lazily so the API process and unit
tests do not need the VoxCPM2 Conda environment.
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path
from typing import Any


def import_runtime():
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from voxcpm import VoxCPM
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM2 runtime 无法导入。请确认 voxcpm2 Conda 环境已安装 "
            f"voxcpm、torch、numpy 和 soundfile。缺少依赖：{exc.name or exc}"
        ) from exc
    return VoxCPM, np, sf, torch


def set_seed(seed: int, np: Any, torch: Any) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def from_pretrained_kwargs(VoxCPM: Any, args: Any) -> dict[str, Any]:
    """Build loading options supported by the installed voxcpm version."""
    signature = inspect.signature(VoxCPM.from_pretrained)
    options = {
        "load_denoiser": args.load_denoiser,
        "local_files_only": args.local_files_only,
        "optimize": args.optimize,
    }
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return options
    return {key: value for key, value in options.items() if key in signature.parameters}


def generate_kwargs(
    model: Any,
    args: Any,
    chunk: str,
    ref_audio: Path,
    prompt_text: str | None,
) -> dict[str, Any]:
    """Build voice-cloning arguments supported by the installed version."""
    options = {
        "text": chunk,
        "reference_wav_path": str(ref_audio),
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
    }
    if prompt_text is not None:
        options["prompt_text"] = prompt_text
        options["prompt_wav_path"] = str(ref_audio)

    signature = inspect.signature(model.generate)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return options
    return {key: value for key, value in options.items() if key in signature.parameters}


def to_mono_float32(waveform: Any, np: Any):
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, np: Any):
    if not waveforms:
        raise RuntimeError("VoxCPM2 未返回任何音频片段。")

    segments = [to_mono_float32(waveform, np) for waveform in waveforms]
    pause_samples = int(sample_rate * max(pause_ms, 0) / 1000)
    if pause_samples <= 0 or len(segments) == 1:
        return np.concatenate(segments)

    pause = np.zeros(pause_samples, dtype=np.float32)
    joined = []
    for index, segment in enumerate(segments):
        joined.append(segment)
        if index < len(segments) - 1:
            joined.append(pause)
    return np.concatenate(joined)


def resolve_sample_rate(model: Any) -> int:
    tts_model = getattr(model, "tts_model", None)
    sample_rate = getattr(tts_model, "sample_rate", None)
    if sample_rate is None:
        raise RuntimeError("无法从 model.tts_model.sample_rate 获取 VoxCPM2 采样率。")
    return int(sample_rate)
