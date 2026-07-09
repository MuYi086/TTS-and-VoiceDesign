#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot OmniVoice worker")
    parser.add_argument("--input-json", required=True, help="Request JSON file path")
    parser.add_argument("--output-wav", required=True, help="Output wav file path")
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_path(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label}不存在：{resolved}")
    return resolved


def normalize_text(text: str) -> str:
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise RuntimeError("text 不能为空。")
    return normalized


def split_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    sentences = re.findall(r".+?[。！？；;!?]|.+$", text, flags=re.S)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_sentence(sentence, max_chars))
            continue

        candidate = current + sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def split_long_sentence(text: str, max_chars: int) -> list[str]:
    parts = re.findall(r".+?[，,、：:]|.+$", text, flags=re.S)
    chunks: list[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(part[index : index + max_chars] for index in range(0, len(part), max_chars))
            continue
        candidate = current + part
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def resolve_dtype(torch: Any, dtype_name: Any) -> Any:
    normalized = str(dtype_name or "float16").strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized == "auto":
        return "auto"
    raise ValueError(f"不支持的 dtype：{dtype_name}")


def seed_everything(torch: Any, np: Any, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_mono_float32(audio: Any, np: Any) -> Any:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0 if waveform.shape[0] <= 2 else 1)
    return waveform.reshape(-1)


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, np: Any) -> Any:
    if not waveforms:
        raise RuntimeError("OmniVoice 未返回音频片段。")

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


def clear_cuda_cache(torch: Any) -> None:
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


def prepare_environment(request: dict[str, Any]) -> None:
    runtime_cache_dir = str(request.get("runtime_cache_dir") or Path.cwd() / ".cache/runtime")
    hf_mirror_dir = str(request.get("hf_mirror_dir") or Path.home() / "hf-mirror")
    local_files_only = bool(request.get("local_files_only", True))

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("HF_HOME", hf_mirror_dir)
    os.environ.setdefault("HF_MODULES_CACHE", os.path.join(runtime_cache_dir, "hf_modules"))
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(runtime_cache_dir, "numba"))
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(runtime_cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(runtime_cache_dir, "xdg"))

    for key in ("HF_MODULES_CACHE", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        os.makedirs(os.environ[key], exist_ok=True)

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def import_runtime():
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise RuntimeError(
            "OmniVoice 运行时不可导入。请确认 omnivoice conda 环境已安装 omnivoice、"
            f"torch、numpy、soundfile。缺失导入：{exc.name or exc}"
        ) from exc
    return OmniVoice, np, sf, torch


def build_generation_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_step": int(request.get("num_step") or 32),
        "guidance_scale": float(request.get("guidance_scale") or 2.0),
        "t_shift": float(request.get("t_shift") or 0.1),
        "denoise": bool(request.get("denoise", True)),
        "postprocess_output": bool(request.get("postprocess_output", True)),
        "layer_penalty_factor": float(request.get("layer_penalty_factor") or 5.0),
        "position_temperature": float(request.get("position_temperature") or 5.0),
        "class_temperature": float(request.get("class_temperature") or 0.0),
        "audio_chunk_duration": float(request.get("audio_chunk_duration") or 15.0),
        "audio_chunk_threshold": float(request.get("audio_chunk_threshold") or 30.0),
        "pad_duration": float(request.get("pad_duration") or 0.1),
        "fade_duration": float(request.get("fade_duration") or 0.1),
    }

    speed = request.get("speed")
    if speed is not None:
        kwargs["speed"] = float(speed)

    duration = request.get("duration")
    if duration is not None:
        kwargs["duration"] = float(duration)

    return kwargs


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    prepare_environment(request)
    OmniVoice, np, sf, torch = import_runtime()

    model_path = require_path(str(request.get("model_path") or ""), "模型路径")
    ref_audio_path = require_path(str(request.get("ref_audio_path") or ""), "参考音频")
    text = normalize_text(str(request.get("text") or ""))
    ref_text = normalize_optional_text(request.get("ref_text"))
    language = normalize_optional_text(request.get("language"))
    device_map = normalize_optional_text(request.get("device_map")) or "cuda:0"
    dtype = resolve_dtype(torch, request.get("dtype") or "float16")
    seed = int(request.get("seed") or 42)
    preprocess_prompt = bool(request.get("preprocess_prompt", True))
    max_chars_per_chunk = int(request.get("max_chars_per_chunk") or 120)
    pause_ms = int(request.get("pause_ms") or 250)
    local_files_only = bool(request.get("local_files_only", True))

    if not torch.cuda.is_available():
        raise RuntimeError("OmniVoice 合成需要 CUDA GPU。")

    seed_everything(torch, np, seed)
    chunks = split_text(text, max_chars_per_chunk)

    model = None
    voice_clone_prompt = None
    started = time.perf_counter()
    try:
        print(f"[OmniVoice worker] 模型目录: {model_path}")
        print(f"[OmniVoice worker] 参考音频: {ref_audio_path}")
        print(f"[OmniVoice worker] 参考文本: {'provided' if ref_text else 'not provided'}")
        print(f"[OmniVoice worker] 文本长度: {len(text)} 字, chunks={len(chunks)}")
        print(f"[OmniVoice worker] device_map={device_map}, dtype={dtype}")

        model = OmniVoice.from_pretrained(
            str(model_path),
            device_map=device_map,
            dtype=dtype,
            local_files_only=local_files_only,
        )
        sample_rate = int(getattr(model, "sampling_rate", 24000))
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt,
        )

        waveforms = []
        generation_kwargs = build_generation_kwargs(request)
        for index, chunk in enumerate(chunks, start=1):
            print(f"[OmniVoice worker] 合成 chunk {index}/{len(chunks)} ({len(chunk)} chars)")
            audios = model.generate(
                text=chunk,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                **generation_kwargs,
            )
            if not audios:
                raise RuntimeError("OmniVoice 未返回音频片段。")
            waveforms.append(audios[0])

        waveform = join_waveforms(waveforms, sample_rate, pause_ms, np)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, sample_rate)
        elapsed = time.perf_counter() - started
        print(f"[OmniVoice worker] 完成: sample_rate={sample_rate}, elapsed={elapsed:.2f}s, output={output_wav}")
    finally:
        if voice_clone_prompt is not None:
            try:
                del voice_clone_prompt
            except Exception:
                pass
        if model is not None:
            try:
                del model
            except Exception:
                pass
        clear_cuda_cache(torch)


def main() -> int:
    args = parse_args()
    request = load_request(args.input_json)
    try:
        synthesize(request, Path(args.output_wav).expanduser().resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
