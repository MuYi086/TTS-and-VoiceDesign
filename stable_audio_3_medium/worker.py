#!/usr/bin/env python3
"""One-shot Stable Audio 3 Medium inference worker.

The worker owns the heavyweight model lifecycle.  It runs inside the same
Python 3.12 uv project as the API, writes one WAV, clears CUDA allocations and
exits.  The upstream runtime can use flex-attention/SDPA when FlashAttention
is absent; strict FlashAttention validation is available by configuration.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any


MAX_SECONDS = 380.0
REQUIRED_MODEL_FILES = (
    "model_config.json",
    "model.safetensors",
    "t5gemma-b-b-ul2/config.json",
    "t5gemma-b-b-ul2/model.safetensors",
    "t5gemma-b-b-ul2/tokenizer.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Stable Audio 3 Medium request")
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


def require_model_path(path: str) -> Path:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Local Stable Audio 3 Medium model directory is missing: {model_path}"
        )
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Local Stable Audio 3 Medium model directory is incomplete: "
            f"{model_path}; missing: {', '.join(missing)}"
        )
    return model_path


def maybe_add_upstream_path(upstream_path: Path) -> None:
    upstream_path = upstream_path.expanduser().resolve()
    if not (upstream_path / "stable_audio_3").is_dir():
        raise FileNotFoundError(
            f"Official stable-audio-3 source directory is missing or incomplete: {upstream_path}"
        )
    if str(upstream_path) not in sys.path:
        sys.path.insert(0, str(upstream_path))


def import_runtime(upstream_path: Path):
    maybe_add_upstream_path(upstream_path)
    try:
        import soundfile as sf
        import torch
        from stable_audio_3.loading_utils import load_diffusion_cond
        from stable_audio_3.model import StableAudioModel
    except ImportError as exc:
        raise RuntimeError(
            "Stable Audio 3 runtime is not importable. Install the uv project "
            "dependencies and check STABLE_AUDIO_3_REPO_PATH. "
            f"Missing import: {exc.name or exc}"
        ) from exc
    return StableAudioModel, load_diffusion_cond, sf, torch


def check_flash_attention(torch: Any, required: bool) -> bool:
    """Validate FlashAttention when strict mode is enabled, otherwise report fallback."""
    try:
        import flash_attn
        from flash_attn import flash_attn_func
    except ImportError as exc:
        if required:
            raise RuntimeError(
                "Stable Audio 3 Medium strict mode requires Flash Attention 2. "
                "Install a cp312 wheel matching Torch/CUDA or set "
                "STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN=0 to use the upstream "
                f"flex-attention/SDPA fallback. Missing import: {exc}"
            ) from exc
        print(
            "[Stable Audio 3 Medium] flash-attn unavailable; using the upstream "
            "flex-attention/SDPA fallback"
        )
        return False

    if flash_attn_func is None:
        raise RuntimeError("Flash Attention 2 imported but flash_attn_func is unavailable.")
    print(f"[Stable Audio 3 Medium] flash-attn: {getattr(flash_attn, '__version__', 'installed')}")
    return True


def require_cuda(torch: Any) -> None:
    """Apply Stable Audio 3 Medium's GPU requirement."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stable Audio 3 Medium requires an NVIDIA CUDA GPU, but "
            "torch.cuda.is_available() is False."
        )

    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise RuntimeError(
            "Stable Audio 3 Medium requires an Ampere-or-newer GPU; "
            f"detected compute capability {capability[0]}.{capability[1]}."
        )


def resolve_device(requested: str) -> str:
    if requested != "cuda":
        raise ValueError("Stable Audio 3 Medium device must be cuda.")
    return requested


def resolve_model_half(dtype: str) -> bool:
    if dtype != "float16":
        raise ValueError("Stable Audio 3 Medium dtype must be float16.")
    return True


def patch_local_text_encoder_path(model_config: dict[str, Any], model_path: Path) -> dict[str, Any]:
    """Replace the upstream Hub text-encoder path with the local checkpoint."""
    local_config = copy.deepcopy(model_config)
    conditioning = local_config.get("model", {}).get("conditioning", {})
    prompt_configs = [
        item for item in conditioning.get("configs", []) if item.get("id") == "prompt"
    ]
    if not prompt_configs:
        raise ValueError("model_config.json does not contain a prompt conditioner.")

    local_text_encoder = (model_path / "t5gemma-b-b-ul2").resolve()
    for item in prompt_configs:
        config = item.setdefault("config", {})
        config["model_path"] = str(local_text_encoder)
        config.pop("repo_id", None)
        config.pop("subfolder", None)
    return local_config


def load_local_model(
    model_path: Path,
    upstream_path: Path,
    device: str,
    model_half: bool,
):
    StableAudioModel, load_diffusion_cond, _, _ = import_runtime(upstream_path)
    model_config = json.loads((model_path / "model_config.json").read_text(encoding="utf-8"))
    model_config = patch_local_text_encoder_path(model_config, model_path)
    diffusion_model = load_diffusion_cond(
        model_config,
        str(model_path / "model.safetensors"),
        device=device,
        model_half=model_half,
    )
    diffusion_model.use_lora = False
    diffusion_model.lora_names = []
    return StableAudioModel(diffusion_model, model_config, device, model_half)


def model_sample_rate(model: Any) -> int:
    sample_rate = model.model_config.get("sample_rate")
    if sample_rate is None:
        sample_rate = getattr(model.model, "sample_rate", None)
    if sample_rate is None:
        raise RuntimeError("Could not resolve Stable Audio 3 sample rate.")
    return int(sample_rate)


def model_sample_size(model: Any) -> int:
    sample_size = model.model_config.get("sample_size")
    if sample_size is None:
        raise RuntimeError("Could not resolve Stable Audio 3 maximum sample size.")
    return int(sample_size)


def audio_to_numpy(audio: Any, torch: Any):
    waveform = audio.detach().to(torch.float32).cpu().numpy()
    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise RuntimeError(f"Expected batch size 1, got waveform shape {waveform.shape}.")
        waveform = waveform[0]
    if waveform.ndim == 1:
        return waveform[:, None]
    if waveform.ndim == 2:
        # Stable Audio returns (channels, samples); soundfile writes (samples, channels).
        return waveform.T if waveform.shape[0] <= 8 else waveform
    raise RuntimeError(f"Unexpected generated waveform shape: {waveform.shape}.")


def clear_cuda_cache(torch: Any) -> None:
    """Release the worker allocator without hiding inference outcomes."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"[Stable Audio 3 Medium] CUDA synchronize skipped: {exc}")
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[Stable Audio 3 Medium] CUDA cache cleanup skipped: {exc}")


def configure_runtime_cache(payload: dict[str, Any]) -> None:
    """Set isolated Hugging Face and XDG caches before model imports."""
    cache_dir = payload.get("runtime_cache_dir")
    if isinstance(cache_dir, str) and cache_dir.strip():
        base = Path(cache_dir).expanduser()
        os.environ.setdefault("HF_MODULES_CACHE", str(base / "hf_modules"))
        os.environ.setdefault("XDG_CACHE_HOME", str(base / "xdg"))
        Path(os.environ["HF_MODULES_CACHE"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    mirror_dir = payload.get("hf_mirror_dir")
    if isinstance(mirror_dir, str) and mirror_dir.strip():
        os.environ.setdefault("HF_HOME", str(Path(mirror_dir).expanduser()))
    if payload.get("local_files_only", True):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
    )
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def main() -> None:
    args = parse_args()
    payload = read_payload(args.input_json)
    configure_runtime_cache(payload)

    prompt = required_text(payload, "prompt")
    seconds = positive_float(payload, "seconds", maximum=MAX_SECONDS)
    steps = int(payload["steps"])
    if steps < 1:
        raise ValueError("steps must be greater than or equal to one.")
    cfg_scale = float(payload["cfg_scale"])
    if cfg_scale < 0:
        raise ValueError("cfg_scale must be greater than or equal to zero.")
    seed = int(payload["seed"])
    requested_device = resolve_device(required_text(payload, "device"))
    requested_dtype = resolve_model_half(required_text(payload, "dtype"))
    model_path = require_model_path(required_text(payload, "model_path"))
    upstream_path = Path(required_text(payload, "upstream_path"))
    require_flash_attn = bool(payload.get("require_flash_attn", False))

    _, _, sf, torch = import_runtime(upstream_path)
    require_cuda(torch)
    check_flash_attention(torch, require_flash_attn)
    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = None
    audio = None
    try:
        print(
            "[Stable Audio 3 Medium] Loading model for one request: "
            f"model={model_path}, device={requested_device}, dtype=float16"
        )
        model = load_local_model(model_path, upstream_path, requested_device, requested_dtype)
        with torch.inference_mode():
            audio = model.generate(
                prompt=prompt,
                duration=seconds,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                batch_size=1,
                sample_size=model_sample_size(model),
            )
        waveform = audio_to_numpy(audio, torch)
        sf.write(str(output_path), waveform, model_sample_rate(model), subtype="FLOAT")
        print(f"[Stable Audio 3 Medium] Saved {output_path}")
    finally:
        if audio is not None:
            del audio
        if model is not None:
            del model
        clear_cuda_cache(torch)


if __name__ == "__main__":
    main()
