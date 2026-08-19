#!/usr/bin/env python3
"""One-shot MOSS-VoiceGenerator inference worker."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from moss_voice_design_compat import (
    install_moss_decode_compatibility,
    validate_moss_codec_compatibility,
    validate_moss_codec_path,
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="One-shot MOSS VoiceGenerator worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def require_path(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label}不存在：{resolved}")
    return resolved


def normalize_text(value: Any, label: str) -> str:
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", str(value or "").strip())
    if not text:
        raise ValueError(f"{label} 不能为空。")
    return text


def split_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for sentence in re.findall(r".+?[。！？；;!?]|.+$", text, flags=re.S):
        sentence = sentence.strip()
        if not sentence:
            continue
        pieces = [
            sentence[index : index + max_chars] for index in range(0, len(sentence), max_chars)
        ]
        for piece in pieces:
            if current and len(current) + len(piece) > max_chars:
                chunks.append(current)
                current = ""
            current += piece
    if current:
        chunks.append(current)
    return chunks


def resolve_dtype(torch, value: str):
    requested = str(value or "auto").lower()
    if requested == "auto":
        return torch.bfloat16
    try:
        return getattr(torch, requested)
    except AttributeError as exc:
        raise ValueError(f"不支持的 dtype：{value}") from exc


def resolve_attention(torch, requested: str, dtype: Any) -> str:
    value = str(requested or "auto")
    if value != "auto":
        return value
    if importlib.util.find_spec("flash_attn") is not None and dtype in {
        torch.float16,
        torch.bfloat16,
    }:
        major, _minor = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa"


def decode_message(processor, outputs):
    install_moss_decode_compatibility(processor)
    messages = processor.decode(outputs)
    if not messages or messages[0] is None or not messages[0].audio_codes_list:
        raise RuntimeError("MOSS-VoiceGenerator 解码结果不包含音频。")
    return messages[0].audio_codes_list[0]


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, torch) -> np.ndarray:
    if not waveforms:
        raise RuntimeError("MOSS-VoiceGenerator 未返回音频。")
    segments = []
    for waveform in waveforms:
        audio = (
            waveform.detach().float().cpu().numpy()
            if isinstance(waveform, torch.Tensor)
            else np.asarray(waveform)
        )
        if audio.ndim == 2:
            audio = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
        segments.append(audio.reshape(-1).astype(np.float32, copy=False))
    pause = np.zeros(int(sample_rate * max(pause_ms, 0) / 1000), dtype=np.float32)
    result: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        result.append(segment)
        if index + 1 < len(segments) and len(pause):
            result.append(pause)
    return np.concatenate(result)


def load_moss_processor(auto_processor, model_path: Path, codec_path: Path):
    """Load the MOSS custom processor without leaking generic HF-only kwargs."""
    return auto_processor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        normalize_inputs=True,
        codec_path=str(codec_path),
    )


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    try:
        import torch
        from transformers import AutoModel, AutoProcessor, processing_utils
    except ImportError as exc:
        raise RuntimeError(f"MOSS-VoiceGenerator 运行时不可导入：{exc.name or exc}") from exc

    if not hasattr(processing_utils, "MODALITY_TO_BASE_CLASS_MAPPING"):
        processing_utils.MODALITY_TO_BASE_CLASS_MAPPING = {}
    if request.get("local_files_only", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not torch.cuda.is_available():
        raise RuntimeError("MOSS-VoiceGenerator 需要 CUDA GPU。")

    model_path = require_path(str(request.get("model_path") or ""), "MOSS-VoiceGenerator 模型目录")
    codec_path = require_path(str(request.get("codec_path") or ""), "MOSS 音频 tokenizer 目录")
    validate_moss_codec_path(codec_path)
    text = normalize_text(request.get("text"), "text")
    instruction = normalize_text(request.get("voice_description"), "voice_description")
    dtype = resolve_dtype(torch, request.get("dtype", "auto"))
    attention = resolve_attention(torch, request.get("attn_implementation", "auto"), dtype)
    processor = None
    model = None
    try:
        # MOSS 的自定义 Processor 不接受 local_files_only；离线加载由
        # HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE 环境变量控制，不能把该参数
        # 继续传入其 ProcessorMixin 构造函数。
        processor = load_moss_processor(AutoProcessor, model_path, codec_path)
        validate_moss_codec_compatibility(processor)
        processor.audio_tokenizer = processor.audio_tokenizer.to("cuda")
        model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            attn_implementation=attention,
            dtype=dtype,
            local_files_only=bool(request.get("local_files_only", True)),
        ).to("cuda")
        model.eval()
        sample_rate = int(processor.model_config.sampling_rate)
        waveforms = []
        chunks = split_text(text, int(request.get("max_chars_per_chunk") or 0))
        for chunk in chunks:
            conversation = [[processor.build_user_message(text=chunk, instruction=instruction)]]
            batch = processor(conversation, mode="generation")
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(request.get("max_new_tokens") or 4096),
                    audio_temperature=float(request.get("audio_temperature") or 1.5),
                    audio_top_p=float(request.get("audio_top_p") or 0.6),
                    audio_top_k=int(request.get("audio_top_k") or 50),
                    audio_repetition_penalty=float(request.get("audio_repetition_penalty") or 1.1),
                )
            waveforms.append(decode_message(processor, outputs))
        waveform = join_waveforms(
            waveforms, sample_rate, int(request.get("pause_ms") or 250), torch
        )
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, sample_rate)
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    try:
        synthesize(load_request(args.input_json), Path(args.output_wav).expanduser().resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
