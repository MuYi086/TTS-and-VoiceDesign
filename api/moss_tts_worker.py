#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from audio_trim import trim_leading_silence


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


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def generation_frame_budget(
    text: str,
    max_new_tokens: int,
    *,
    auto_limit: bool,
    min_new_tokens: int,
    new_tokens_per_char: float,
    requested_tokens: int | None = None,
) -> int:
    """Bound a generation chunk so a missing EOS cannot grow the KV cache indefinitely."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0。")
    if not auto_limit:
        return max_new_tokens
    if min_new_tokens <= 0:
        raise ValueError("min_new_tokens 必须大于 0。")
    if new_tokens_per_char <= 0:
        raise ValueError("new_tokens_per_char 必须大于 0。")

    estimated = math.ceil(max(1, len(text.strip())) * new_tokens_per_char)
    if requested_tokens is not None:
        estimated = max(estimated, requested_tokens + 32)
    return min(max_new_tokens, max(min_new_tokens, estimated))


def generated_frame_counts(outputs: Any) -> list[int]:
    counts: list[int] = []
    for output in outputs:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            continue
        start_length, generation_ids = output[0], output[1]
        shape = getattr(generation_ids, "shape", ())
        if not shape:
            continue
        counts.append(max(0, int(shape[0]) - int(start_length) - 1))
    return counts


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


def trim_generated_audio(waveform: Any, sample_rate: int, np: Any, torch: Any) -> tuple[Any, int]:
    """Trim only the generated prefix, preserving MOSS channel layout."""
    trimmed, trimmed_samples = trim_leading_silence(
        waveform.detach().cpu().numpy(), sample_rate, np
    )
    return torch.from_numpy(trimmed).to(dtype=torch.float32), trimmed_samples


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
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    for key in ("HF_MODULES_CACHE", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        os.makedirs(os.environ[key], exist_ok=True)

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def import_runtime():
    try:
        import torch
        import torchaudio
        import transformers
        import transformers.processing_utils as processing_utils
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "MOSS-TTS 运行时不可导入。请确认 moss-tts-py310 环境已安装 "
            f"torch、torchaudio 和官方 transformers 依赖。缺失导入：{exc.name or exc}"
        ) from exc

    if not hasattr(processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING"):
        raise RuntimeError(
            "当前 transformers 版本过旧，缺少 MOSS-TTS remote code 所需的 "
            "processing_utils.MODALITY_TO_BASE_CLASS_MAPPING。"
            f"当前版本：{transformers.__version__}"
        )
    return AutoModel, AutoProcessor, torch, torchaudio


def resolve_device(torch: Any) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("MOSS-TTS 合成需要 CUDA GPU。")
    return "cuda"


def resolve_dtype(torch: Any, dtype: str, device: str) -> Any:
    if dtype == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtypes:
        raise ValueError(f"不支持的 MOSS dtype: {dtype}")
    return dtypes[dtype]


def resolve_attn_implementation(
    torch: Any,
    requested: str,
    device: str,
    dtype: Any,
) -> str:
    if requested != "auto":
        return requested
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _minor = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


def configure_sdpa_backend(torch: Any, requested: str) -> str:
    normalized = (requested or "math").strip().lower()
    if normalized not in {"auto", "math"}:
        raise ValueError(f"不支持的 MOSS SDPA backend: {requested}")

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if normalized == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    else:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    return normalized


def patch_pad_sequence_padding_side(torch: Any) -> None:
    original = torch.nn.utils.rnn.pad_sequence
    if "padding_side" in inspect.signature(original).parameters:
        return

    def pad_sequence_compat(
        sequences,
        batch_first=False,
        padding_value=0.0,
        padding_side="right",
    ):
        if padding_side == "right":
            return original(
                sequences,
                batch_first=batch_first,
                padding_value=padding_value,
            )
        if padding_side != "left":
            raise ValueError(
                f"padding_side must be 'right' or 'left', got {padding_side!r}"
            )

        flipped = [sequence.flip(0) for sequence in sequences]
        padded = original(
            flipped,
            batch_first=batch_first,
            padding_value=padding_value,
        )
        sequence_dim = 1 if batch_first else 0
        return padded.flip(sequence_dim)

    torch.nn.utils.rnn.pad_sequence = pad_sequence_compat


def patch_autocast_enabled_device_arg(torch: Any) -> None:
    original = torch.is_autocast_enabled
    try:
        original("cuda")
    except TypeError:
        def is_autocast_enabled_compat(device_type=None):
            if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
                return torch.is_autocast_cpu_enabled()
            return original()

        torch.is_autocast_enabled = is_autocast_enabled_compat

    if not hasattr(torch, "get_autocast_dtype"):
        def get_autocast_dtype_compat(device_type=None):
            if device_type == "cpu" and hasattr(torch, "get_autocast_cpu_dtype"):
                return torch.get_autocast_cpu_dtype()
            if hasattr(torch, "get_autocast_gpu_dtype"):
                return torch.get_autocast_gpu_dtype()
            return torch.float32

        torch.get_autocast_dtype = get_autocast_dtype_compat


def collect_audio(decoded_messages: Any, torch: Any) -> Any:
    waveforms = []
    channels = None
    for message in decoded_messages:
        if message is None:
            continue
        for audio in message.audio_codes_list:
            if not isinstance(audio, torch.Tensor):
                continue
            waveform = audio.to(torch.float32).cpu()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim != 2:
                raise RuntimeError(
                    "MOSS-TTS 解码音频必须是 [samples] 或 [channels, samples]，"
                    f"实际为 {tuple(waveform.shape)}。"
                )
            if channels is None:
                channels = int(waveform.shape[0])
            elif int(waveform.shape[0]) != channels:
                raise RuntimeError("MOSS-TTS 返回了不一致的声道数。")
            waveforms.append(waveform)

    if not waveforms:
        raise RuntimeError("MOSS-TTS 未返回解码音频。")
    return torch.cat(waveforms, dim=-1)


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    prepare_environment(request)

    AutoModel, AutoProcessor, torch, torchaudio = import_runtime()
    import numpy as np

    model_path = require_path(str(request.get("model_path") or ""), "模型路径")
    ref_audio_path = require_path(str(request.get("ref_audio_path") or ""), "参考音频")
    codec_path = parse_codec_path(request.get("codec_path"))
    text = normalize_text(str(request.get("text") or ""))
    language = normalize_optional_str(request.get("language")) or "Chinese"
    instruction = normalize_optional_str(request.get("instruction"))
    quality = normalize_optional_str(request.get("quality"))
    tokens = parse_optional_int(request.get("tokens"))
    max_new_tokens = int(request.get("max_new_tokens") or 4096)
    auto_limit_max_new_tokens = parse_bool(
        request.get("auto_limit_max_new_tokens"),
        default=True,
    )
    min_new_tokens = int(request.get("min_new_tokens") or 256)
    new_tokens_per_char = float(request.get("new_tokens_per_char") or 10.0)
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
    sdpa_backend = normalize_optional_str(request.get("sdpa_backend")) or "math"
    dtype = normalize_optional_str(request.get("dtype")) or "auto"
    max_chars_per_chunk = int(request.get("max_chars_per_chunk") or 80)
    pause_ms = int(request.get("pause_ms") or 250)

    device = resolve_device(torch)
    resolved_dtype = resolve_dtype(torch, dtype, device)
    resolved_attn_implementation = resolve_attn_implementation(
        torch,
        attn_implementation,
        device,
        resolved_dtype,
    )
    resolved_sdpa_backend = configure_sdpa_backend(torch, sdpa_backend)
    patch_pad_sequence_padding_side(torch)
    patch_autocast_enabled_device_arg(torch)

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
        print(
            f"[MOSS worker] attn_implementation={resolved_attn_implementation}, "
            f"sdpa_backend={resolved_sdpa_backend}"
        )
        print(
            f"[MOSS worker] generation_limit=auto:{auto_limit_max_new_tokens}, "
            f"floor:{min_new_tokens}, per_char:{new_tokens_per_char:g}, hard_cap:{max_new_tokens}"
        )

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
                chunk_frame_budget = generation_frame_budget(
                    chunk,
                    max_new_tokens,
                    auto_limit=auto_limit_max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    new_tokens_per_char=new_tokens_per_char,
                    requested_tokens=tokens,
                )
                print(
                    f"[MOSS worker] 合成 chunk {index}/{len(chunks)} "
                    f"({len(chunk)} chars, frame_budget={chunk_frame_budget})"
                )
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
                    "max_new_tokens": chunk_frame_budget,
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
                frame_counts = generated_frame_counts(outputs)
                if frame_counts:
                    print(
                        f"[MOSS worker] chunk {index} 生成帧数={frame_counts}"
                        + (
                            "（达到安全上限）"
                            if any(count >= chunk_frame_budget for count in frame_counts)
                            else ""
                        )
                    )
                waveforms.append(collect_audio(processor.decode(outputs), torch))

        sample_rate = int(processor.model_config.sampling_rate)
        waveform = join_waveforms(waveforms, sample_rate, pause_ms, torch)
        waveform, trimmed_samples = trim_generated_audio(waveform, sample_rate, np, torch)
        if trimmed_samples > 0:
            print(f"[MOSS worker] 裁掉前导空白 {trimmed_samples / sample_rate:.2f}s")
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
