#!/usr/bin/env python3
"""One-shot ACE-Step 1.5 XL Turbo inference worker."""

from __future__ import annotations

import argparse
import gc
import json
import os
import secrets
from pathlib import Path
from typing import Any

MIN_SECONDS = 10.0
MAX_SECONDS = 600.0
REQUIRED_MODEL_PATHS = (
    "model_index.json",
    "condition_encoder",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ACE-Step 1.5 BGM request")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--metadata-json", required=True)
    return parser.parse_args()


def read_payload(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Worker input must be a JSON object.")
    return payload


def configure_runtime_cache(payload: dict[str, Any]) -> None:
    """Set offline/cache controls before importing Hugging Face libraries."""
    cache_dir = payload.get("runtime_cache_dir")
    if cache_dir:
        cache_path = Path(str(cache_dir)).expanduser()
        os.environ.setdefault("HF_MODULES_CACHE", str(cache_path / "hf_modules"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_path / "xdg"))

    hf_mirror_dir = payload.get("hf_mirror_dir")
    if hf_mirror_dir:
        os.environ.setdefault("HF_HOME", str(Path(str(hf_mirror_dir)).expanduser()))

    if payload.get("local_files_only", True):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


def validate_model(model_path: Path) -> Path:
    """Validate model components without assuming a single weight filename."""
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local ACE-Step model directory is missing: {model_path}")

    missing = [
        name
        for name in REQUIRED_MODEL_PATHS
        if not (
            (model_path / name).is_file()
            if name.endswith(".json")
            else (model_path / name).is_dir()
        )
    ]
    transformer_weights = tuple((model_path / "transformer").glob("*.safetensors"))
    if missing or not transformer_weights:
        details = []
        if missing:
            details.append(f"missing components: {', '.join(missing)}")
        if not transformer_weights:
            details.append("transformer has no .safetensors weights")
        raise FileNotFoundError(
            f"Local ACE-Step model directory is incomplete: {model_path}; " + "; ".join(details)
        )
    return model_path


def load_pipeline(
    model_path: Path,
    device_name: str,
    dtype_name: str,
    offload: str,
    vae_tiling: bool,
):
    """Load ACE-Step only inside the short-lived worker process."""
    import torch
    from diffusers import AceStepPipeline

    if device_name != "cuda":
        raise ValueError("当前 ACE-Step 部署只支持 ACESTEP_DEVICE=cuda。")
    if not torch.cuda.is_available():
        raise RuntimeError("ACE-Step requires CUDA on this deployment.")
    if dtype_name != "bfloat16":
        raise ValueError("当前 ACE-Step 生产配置只允许 bfloat16。")
    if offload not in {"model", "sequential", "none"}:
        raise ValueError(f"Unsupported ACE-Step offload mode: {offload}")

    print("[ACE-Step] before load:", torch.cuda.mem_get_info())
    pipe = AceStepPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if vae_tiling:
        pipe.vae.enable_tiling()

    if offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)
    return pipe, torch


def run_generation(pipe: Any, torch: Any, payload: dict[str, Any]):
    """Generate pure instrumental music with the ACE-Step text2music task."""
    seed = int(payload.get("seed", -1))
    if seed < 0:
        seed = secrets.randbelow(2**31 - 1)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs: dict[str, Any] = {
        "prompt": payload["prompt"],
        "lyrics": "",
        "audio_duration": float(payload["seconds"]),
        "num_inference_steps": int(payload.get("steps", 8)),
        "generator": generator,
        "task_type": "text2music",
    }

    if payload.get("bpm") is not None:
        kwargs["bpm"] = int(payload["bpm"])
    if payload.get("keyscale"):
        kwargs["keyscale"] = str(payload["keyscale"])
    if payload.get("timesignature"):
        kwargs["timesignature"] = str(payload["timesignature"])

    with torch.inference_mode():
        result = pipe(**kwargs)
    print("[ACE-Step] after generation:", torch.cuda.mem_get_info())
    return result, seed


def write_audio(result: Any, pipe: Any, output_path: Path) -> tuple[int, int]:
    """Write the pipeline's native 48 kHz stereo output as a float WAV."""
    import numpy as np
    import soundfile as sf

    audio = result.audios[0]
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        elif audio.shape[1] > 8:
            raise ValueError(f"Unexpected ACE-Step audio shape: {audio.shape}")
    else:
        raise ValueError(f"Unexpected ACE-Step audio dimensions: {audio.shape}")

    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[1] != 2:
        raise ValueError(f"ACE-Step must return stereo audio, got shape {audio.shape}")

    sample_rate = int(getattr(pipe, "sample_rate", 48_000))
    if sample_rate != 48_000:
        raise ValueError(f"ACE-Step pipeline returned unexpected sample rate: {sample_rate}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate, subtype="FLOAT")
    return sample_rate, int(audio.shape[1])


def clear_cuda(torch: Any) -> None:
    """Release allocator state before the worker process exits."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def write_metadata(path: Path, *, seed: int, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": "acestep-v15-xl-turbo-diffusers",
                "seed": seed,
                "sample_rate": sample_rate,
                "channels": channels,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    payload = read_payload(args.input_json)
    configure_runtime_cache(payload)

    seconds = float(payload["seconds"])
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ValueError(f"seconds must be between {MIN_SECONDS:g} and {MAX_SECONDS:g}.")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    model_path = validate_model(Path(str(payload["model_path"])))
    output_path = Path(args.output_wav).expanduser().resolve()
    metadata_path = Path(args.metadata_json).expanduser().resolve()

    pipe = None
    result = None
    torch = None
    try:
        pipe, torch = load_pipeline(
            model_path=model_path,
            device_name=str(payload.get("device", "cuda")),
            dtype_name=str(payload.get("dtype", "bfloat16")),
            offload=str(payload.get("offload", "model")),
            vae_tiling=bool(payload.get("vae_tiling", True)),
        )
        result, seed = run_generation(pipe, torch, payload)
        sample_rate, channels = write_audio(result, pipe, output_path)
        write_metadata(metadata_path, seed=seed, sample_rate=sample_rate, channels=channels)
        print(f"[ACE-Step] generation done seed={seed} path={output_path}")
    finally:
        if result is not None:
            del result
        if pipe is not None:
            del pipe
        if torch is not None:
            clear_cuda(torch)


if __name__ == "__main__":
    main()
