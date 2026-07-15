#!/usr/bin/env python3
"""One-shot MOSS-SoundEffect v2.0 inference worker.

This process deliberately owns the model lifecycle.  The HTTP wrapper creates
it for one request and waits for it to exit, which guarantees the model is no
longer resident in GPU memory after the request completes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any


MAX_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MOSS-SoundEffect v2.0 request")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def read_payload(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Worker input must be a JSON object.")
    return payload


def required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def positive_float(payload: dict[str, Any], key: str, *, maximum: float | None = None) -> float:
    value = float(payload[key])
    if value <= 0 or (maximum is not None and value > maximum):
        bound = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{key} must be > 0{bound}.")
    return value


def cleanup_cuda(torch: Any) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # Cleanup must not hide the inference result.
        print(f"[SoundEffect] CUDA synchronize skipped: {exc}")
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[SoundEffect] CUDA cache cleanup skipped: {exc}")


def main() -> None:
    args = parse_args()
    payload = read_payload(args.input_json)

    if payload.get("disable_torchdynamo", True):
        # Upstream recommends this opt-out when Triton/CUDA-graph compilation
        # is unstable.  Preserve an explicit caller setting if one already
        # exists in the service environment.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    model_path = Path(required_text(payload, "model_path")).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local MOSS-SoundEffect model directory is missing: {model_path}")

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
        raise RuntimeError("CUDA was requested for SoundEffect, but PyTorch cannot see a CUDA device.")
    try:
        torch_dtype = getattr(torch, torch_dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown TORCH_DTYPE: {torch_dtype_name!r}") from exc

    output_path = Path(args.output_wav)
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
        # The wrapper also waits for this process to terminate.  These explicit
        # releases reduce the handover delay to the next GPU workload.
        if pipe is not None:
            del pipe
        if audio is not None:
            del audio
        cleanup_cuda(torch)


if __name__ == "__main__":
    main()
