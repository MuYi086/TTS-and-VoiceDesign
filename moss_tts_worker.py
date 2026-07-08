#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot MOSS-TTS worker")
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


def normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.lower() == "none":
        return None
    return normalized


def parse_optional_int(value: Any) -> int | None:
    normalized = normalize_optional_str(value)
    return int(normalized) if normalized is not None else None


def parse_optional_float(value: Any) -> float | None:
    normalized = normalize_optional_str(value)
    return float(normalized) if normalized is not None else None


def parse_codec_path(value: Any) -> str:
    normalized = normalize_optional_str(value)
    if normalized is None:
        raise RuntimeError("codec_path 不能为空。")
    expanded = os.path.expandvars(os.path.expanduser(normalized))
    if os.path.exists(expanded):
        return os.path.abspath(expanded)
    return normalized


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, torch: Any) -> Any:
    if not waveforms:
        raise RuntimeError("MOSS-TTS 未返回音频片段。")

    normalized: list[Any] = []
    channels: int | None = None
    for waveform in waveforms:
        tensor = waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
        tensor = tensor.to(torch.float32).cpu()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise RuntimeError(f"音频张量维度不合法：{tuple(tensor.shape)}")

        current_channels = int(tensor.shape[0])
        if channels is None:
            channels = current_channels
        elif channels != current_channels:
            raise RuntimeError("MOSS-TTS 返回了不一致的声道数。")
        normalized.append(tensor)

    if len(normalized) == 1 or pause_ms <= 0:
        return torch.cat(normalized, dim=-1)

    pause_samples = int(sample_rate * pause_ms / 1000)
    if pause_samples <= 0:
        return torch.cat(normalized, dim=-1)

    pause = torch.zeros((channels or 1, pause_samples), dtype=torch.float32)
    segments: list[Any] = []
    for index, waveform in enumerate(normalized):
        segments.append(waveform)
        if index < len(normalized) - 1:
            segments.append(pause)
    return torch.cat(segments, dim=-1)


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
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    for key in ("HF_MODULES_CACHE", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        os.makedirs(os.environ[key], exist_ok=True)

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_moss_helpers(script_path: str) -> Any:
    helper_file = require_path(script_path, "MOSS-TTS 辅助脚本")
    spec = importlib.util.spec_from_file_location("timbre_moss_tts", helper_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 MOSS-TTS 辅助脚本：{helper_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    prepare_environment(request)

    helper_script = str(request.get("moss_helper_script") or "")
    helpers = load_moss_helpers(helper_script)
    AutoModel, AutoProcessor, torch, torchaudio = helpers.import_runtime()

    model_path = require_path(str(request.get("model_path") or ""), "模型路径")
    ref_audio_path = require_path(str(request.get("ref_audio_path") or ""), "参考音频")
    codec_path = parse_codec_path(request.get("codec_path"))
    text = normalize_text(str(request.get("text") or ""))
    language = normalize_optional_str(request.get("language")) or "Chinese"
    instruction = normalize_optional_str(request.get("instruction"))
    quality = normalize_optional_str(request.get("quality"))
    tokens = parse_optional_int(request.get("tokens"))
    max_new_tokens = int(request.get("max_new_tokens") or 4096)
    n_vq_for_inference = parse_optional_int(request.get("n_vq_for_inference"))
    audio_temperature = float(request.get("audio_temperature") or 1.7)
    audio_top_p = float(request.get("audio_top_p") or 0.8)
    audio_top_k = int(request.get("audio_top_k") or 25)
    audio_repetition_penalty = float(request.get("audio_repetition_penalty") or 1.0)
    text_temperature = parse_optional_float(request.get("text_temperature"))
    text_top_p = parse_optional_float(request.get("text_top_p"))
    text_top_k = parse_optional_int(request.get("text_top_k"))
    text_repetition_penalty = parse_optional_float(request.get("text_repetition_penalty"))
    attn_implementation = normalize_optional_str(request.get("attn_implementation")) or "auto"
    dtype = normalize_optional_str(request.get("dtype")) or "auto"
    max_chars_per_chunk = int(request.get("max_chars_per_chunk") or 300)
    pause_ms = int(request.get("pause_ms") or 250)

    device = helpers.resolve_device(torch)
    resolved_dtype = helpers.resolve_dtype(torch, dtype, device)
    resolved_attn_implementation = helpers.resolve_attn_implementation(
        torch,
        attn_implementation,
        device,
        resolved_dtype,
    )
    helpers.patch_pad_sequence_padding_side(torch)
    helpers.patch_autocast_enabled_device_arg(torch)

    chunks = split_text(text, max_chars_per_chunk)
    processor = None
    model = None
    started = time.perf_counter()

    try:
        print(f"[MOSS worker] 模型目录: {model_path}")
        print(f"[MOSS worker] codec: {codec_path}")
        print(f"[MOSS worker] 参考音频: {ref_audio_path}")
        print(f"[MOSS worker] 文本长度: {len(text)} 字, chunks={len(chunks)}")
        print(f"[MOSS worker] language={language}, device={device}, dtype={resolved_dtype}")
        print(f"[MOSS worker] attn_implementation={resolved_attn_implementation}")

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            codec_path=codec_path,
        )
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            attn_implementation=resolved_attn_implementation,
            dtype=resolved_dtype,
            local_files_only=bool(request.get("local_files_only", True)),
        ).to(device)
        model.eval()

        waveforms = []
        with torch.no_grad():
            for index, chunk in enumerate(chunks, start=1):
                print(f"[MOSS worker] 合成 chunk {index}/{len(chunks)} ({len(chunk)} chars)")
                conversation = [
                    processor.build_user_message(
                        text=chunk,
                        reference=[str(ref_audio_path)],
                        instruction=instruction,
                        tokens=tokens,
                        quality=quality,
                        language=language,
                    )
                ]
                batch = processor([conversation], mode="generation")
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "audio_temperature": audio_temperature,
                    "audio_top_p": audio_top_p,
                    "audio_top_k": audio_top_k,
                    "audio_repetition_penalty": audio_repetition_penalty,
                }
                if n_vq_for_inference is not None:
                    generation_kwargs["n_vq_for_inference"] = n_vq_for_inference
                if text_temperature is not None:
                    generation_kwargs["text_temperature"] = text_temperature
                if text_top_p is not None:
                    generation_kwargs["text_top_p"] = text_top_p
                if text_top_k is not None:
                    generation_kwargs["text_top_k"] = text_top_k
                if text_repetition_penalty is not None:
                    generation_kwargs["text_repetition_penalty"] = text_repetition_penalty

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
                waveforms.append(helpers.collect_audio(processor.decode(outputs), torch))

        sample_rate = int(processor.model_config.sampling_rate)
        waveform = join_waveforms(waveforms, sample_rate, pause_ms, torch)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_wav), waveform, sample_rate)
        elapsed = time.perf_counter() - started
        print(f"[MOSS worker] 完成: sample_rate={sample_rate}, elapsed={elapsed:.2f}s, output={output_wav}")
    finally:
        if model is not None:
            try:
                del model
            except Exception:
                pass
        if processor is not None:
            try:
                del processor
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
