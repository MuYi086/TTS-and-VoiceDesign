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

from audio_trim import trim_leading_silence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot dots.tts worker")
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


def normalize_language(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def to_mono_float32(audio: Any, np: Any) -> Any:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0 if waveform.shape[0] <= 2 else 1)
    return waveform.reshape(-1)


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, np: Any) -> Any:
    if not waveforms:
        raise RuntimeError("dots.tts 未返回音频片段。")

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
    runtime_cache_dir = str(
        request.get("runtime_cache_dir") or Path(__file__).resolve().parent / ".cache/runtime"
    )
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
        from dots_tts.runtime import DotsTtsRuntime
        from dots_tts.utils.logging import configure_logging
        from dots_tts.utils.util import seed_everything
    except ImportError as exc:
        raise RuntimeError(
            "dots.tts 运行时不可导入。请确认 dots_tts conda 环境已安装 rednote-hilab/dots.tts、"
            f"torch、numpy、soundfile。缺失导入：{exc.name or exc}"
        ) from exc
    return DotsTtsRuntime, configure_logging, np, seed_everything, sf, torch


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    prepare_environment(request)
    DotsTtsRuntime, configure_logging, np, seed_everything, sf, torch = import_runtime()

    model_path = require_path(str(request.get("model_path") or ""), "模型路径")
    ref_audio_path = require_path(str(request.get("ref_audio_path") or ""), "参考音频")
    text = normalize_text(str(request.get("text") or ""))
    prompt_text = str(request["prompt_text"]).strip() if request.get("prompt_text") else None
    language = normalize_language(request.get("language"))
    template_name = str(request["template_name"]).strip() if request.get("template_name") else None

    precision = str(request.get("precision") or "bfloat16")
    seed = int(request.get("seed") or 42)
    ode_method = str(request.get("ode_method") or "euler")
    num_steps = int(request.get("num_steps") or 10)
    guidance_scale = float(request.get("guidance_scale") or 1.2)
    speaker_scale = float(request.get("speaker_scale") or 1.5)
    max_generate_length = int(request.get("max_generate_length") or 500)
    max_chars_per_chunk = int(request.get("max_chars_per_chunk") or 120)
    pause_ms = int(request.get("pause_ms") or 250)
    normalize_text_flag = bool(request.get("normalize_text", False))
    profile_inference = bool(request.get("profile_inference", False))

    if not torch.cuda.is_available():
        raise RuntimeError("dots.tts 合成需要 CUDA GPU。")

    chunks = split_text(text, max_chars_per_chunk)
    configure_logging()
    seed_everything(seed)

    runtime = None
    started = time.perf_counter()
    try:
        print(f"[dots.tts worker] 模型目录: {model_path}")
        print(f"[dots.tts worker] 参考音频: {ref_audio_path}")
        print(f"[dots.tts worker] 文本长度: {len(text)} 字, chunks={len(chunks)}")
        print(f"[dots.tts worker] prompt_text: {'provided' if prompt_text else 'not provided'}")
        runtime = DotsTtsRuntime.from_pretrained(
            str(model_path),
            precision=precision,
            max_generate_length=max_generate_length,
        )

        waveforms = []
        for index, chunk in enumerate(chunks, start=1):
            print(f"[dots.tts worker] 合成 chunk {index}/{len(chunks)} ({len(chunk)} chars)")
            result = runtime.generate(
                text=chunk,
                prompt_audio_path=str(ref_audio_path),
                prompt_text=prompt_text,
                language=language,
                template_name=template_name,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                normalize_text=normalize_text_flag,
                profile_inference=profile_inference,
            )
            waveforms.append(result["audio"].float().cpu().squeeze().numpy())

        sample_rate = int(runtime.sample_rate)
        waveform = join_waveforms(waveforms, sample_rate, pause_ms, np)
        waveform, trimmed_samples = trim_leading_silence(waveform, sample_rate, np)
        if trimmed_samples > 0:
            print(f"[dots.tts worker] 裁掉前导空白 {trimmed_samples / sample_rate:.2f}s")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, sample_rate)
        elapsed = time.perf_counter() - started
        print(f"[dots.tts worker] 完成: sample_rate={sample_rate}, elapsed={elapsed:.2f}s, output={output_wav}")
    finally:
        if runtime is not None:
            try:
                del runtime
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
