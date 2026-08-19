#!/usr/bin/env python3
"""MOSS-SoundEffect v2 一次性推理 worker。"""

from __future__ import annotations

# 声效模型只在子进程中加载；进程退出是释放 MOSS CUDA 上下文的最后保障。
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

MAX_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    """解析一次声效请求的 JSON、输出 WAV 和本地源码路径。"""
    parser = argparse.ArgumentParser(description="Run one MOSS-SoundEffect v2 request")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def read_payload(path: str) -> dict[str, Any]:
    """读取 JSON 对象，并拒绝结构错误的请求。"""
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Worker input must be a JSON object.")
    return payload


def required_text(payload: dict[str, Any], key: str) -> str:
    """读取必须非空的文本字段。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def positive_float(
    payload: dict[str, Any],
    key: str,
    *,
    maximum: float | None = None,
) -> float:
    """读取正数参数，并在需要时施加最大值限制。"""
    value = float(payload[key])
    if value <= 0 or (maximum is not None and value > maximum):
        bound = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{key} must be > 0{bound}.")
    return value


def add_upstream_source(code_path: Path) -> None:
    """把本地 MOSS-TTS 源码放到 import 路径，避免运行时下载依赖。"""
    package_dir = code_path / "moss_soundeffect_v2"
    if package_dir.is_dir():
        import_root = code_path
    elif code_path.name == "moss_soundeffect_v2" and code_path.is_dir():
        import_root = code_path.parent
    else:
        raise FileNotFoundError(f"上游源码目录中没有 moss_soundeffect_v2 包: {code_path}")
    sys.path.insert(0, str(import_root))


def cleanup_cuda(torch: Any) -> None:
    """尽力清空 CUDA 缓存；最终释放仍依赖 worker 进程退出。"""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"[SoundEffect] CUDA synchronize skipped: {exc}")
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[SoundEffect] CUDA cache cleanup skipped: {exc}")


def main() -> None:
    """加载 MOSS-SoundEffect、生成 WAV，并在 finally 中清理资源。"""
    args = parse_args()
    payload = read_payload(args.input_json)

    if payload.get("disable_torchdynamo", True):
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    model_path = Path(required_text(payload, "model_path")).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local MOSS-SoundEffect model directory is missing: {model_path}")

    code_path = Path(required_text(payload, "code_path")).expanduser().resolve()
    if not (code_path / "moss_soundeffect_v2").is_dir() and code_path.name != "moss_soundeffect_v2":
        raise FileNotFoundError(f"MOSS-TTS source checkout is missing: {code_path}")
    add_upstream_source(code_path)

    prompt = required_text(payload, "prompt")
    seconds = positive_float(payload, "seconds", maximum=MAX_SECONDS)
    num_inference_steps = int(payload["num_inference_steps"])
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be greater than zero.")
    cfg_scale = float(payload["cfg_scale"])
    if cfg_scale < 0:
        raise ValueError("cfg_scale must be greater than or equal to zero.")
    sigma_shift = positive_float(payload, "sigma_shift")
    seed = int(payload["seed"])
    device = required_text(payload, "device")
    torch_dtype_name = required_text(payload, "torch_dtype")
    local_files_only = bool(payload.get("local_files_only", True))

    import soundfile as sf
    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for SoundEffect, but PyTorch cannot see a CUDA device."
        )
    try:
        torch_dtype = getattr(torch, torch_dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown TORCH_DTYPE: {torch_dtype_name!r}") from exc

    output_path = Path(args.output_wav).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipe = None
    audio = None
    try:
        print(
            "[SoundEffect] Loading model for one request: "
            f"model={model_path}, device={device}, dtype={torch_dtype_name}"
        )
        pipe = MossSoundEffectPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device=device,
            local_files_only=local_files_only,
        )
        audio = pipe(
            prompt=prompt,
            seconds=seconds,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            sigma_shift=sigma_shift,
            seed=seed,
        )
        waveform = audio[0].detach().to(torch.float32).cpu().transpose(0, 1).numpy()
        sf.write(str(output_path), waveform, pipe.sample_rate)
        print(f"[SoundEffect] Saved {output_path}")
    finally:
        if pipe is not None:
            del pipe
        if audio is not None:
            del audio
        cleanup_cuda(torch)


if __name__ == "__main__":
    main()
