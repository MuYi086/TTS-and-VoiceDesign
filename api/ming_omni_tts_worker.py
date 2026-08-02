#!/usr/bin/env python3
"""One-shot worker for Ming-omni-tts-0.5B design and reference cloning."""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="One-shot Ming-omni-tts worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
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


def build_instruction(request: dict[str, Any]) -> str | None:
    raw = request.get("instruction_json")
    values: dict[str, Any] = {}
    if raw:
        try:
            values.update(json.loads(raw) if isinstance(raw, str) else raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"instruction_json 不是合法 JSON：{exc}") from exc
    for request_key, instruction_key in (
        ("voice_description", "风格"),
        ("style", "风格"),
        ("emotion", "情感"),
        ("dialect", "方言"),
        ("speed", "语速"),
        ("pitch", "基频"),
        ("volume", "音量"),
    ):
        if request.get(request_key) is not None and str(request[request_key]).strip():
            values[instruction_key] = request[request_key]
    if not values:
        return None
    if "audio_sequence" in values:
        return json.dumps(values, ensure_ascii=False)
    item = {
        "序号": 1,
        "说话人": "speaker_1",
        "方言": None,
        "风格": None,
        "语速": None,
        "基频": None,
        "音量": None,
        "情感": None,
        "BGM": {"Genre": None, "Mood": None, "Instrument": None, "Theme": None, "ENV": None, "SNR": None},
        "IP": None,
    }
    item.update({key: value for key, value in values.items() if key in item})
    return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)


def load_upstream(code_path: Path):
    code_path = require_path(str(code_path), "Ming-omni-tts 源码目录")
    required = ("modeling_bailingmm.py", "spkemb_extractor.py", "audio_tokenizer", "fm")
    missing = [name for name in required if not (code_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Ming-omni-tts 源码不完整，缺少 {', '.join(missing)}：{code_path}")
    sys.path.insert(0, str(code_path))
    try:
        import torch
        import torchaudio
        from transformers import AutoTokenizer
        from modeling_bailingmm import BailingMMNativeForConditionalGeneration
        from spkemb_extractor import SpkembExtractor
    except ImportError as exc:
        raise RuntimeError(f"Ming-omni-tts 运行时不可导入：{exc.name or exc}") from exc
    return BailingMMNativeForConditionalGeneration, SpkembExtractor, AutoTokenizer, torch, torchaudio


def prepare_prompt(torchaudio, torch, ref_audio: Path | None, sample_rate: int, extractor):
    if ref_audio is None:
        return None, None
    waveform, source_rate = torchaudio.load(str(ref_audio))
    original = waveform
    if source_rate != sample_rate:
        waveform = torchaudio.transforms.Resample(source_rate, sample_rate)(waveform)
    if source_rate != 16000:
        original = torchaudio.transforms.Resample(source_rate, 16000)(original)
    speaker_embedding = extractor(original)
    pad_align = int(1 / 12.5 * 4 * sample_rate)
    new_len = (waveform.size(-1) + pad_align - 1) // pad_align * pad_align
    if new_len != waveform.size(-1):
        padded = torch.zeros(1, new_len, dtype=waveform.dtype)
        padded[:, : waveform.size(-1)] = waveform
        waveform = padded
    return waveform, [speaker_embedding]


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    if request.get("local_files_only", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    code_path = Path(str(request.get("code_path") or "")).expanduser().resolve()
    model_path = require_path(str(request.get("model_path") or ""), "Ming-omni-tts 模型目录")
    BailingModel, SpkembExtractor, AutoTokenizer, torch, torchaudio = load_upstream(code_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Ming-omni-tts 需要 CUDA GPU。")

    operation = str(request.get("operation") or "clone")
    ref_audio = request.get("ref_audio_path")
    ref_path = require_path(str(ref_audio), "参考音频") if ref_audio else None
    model = None
    try:
        model = BailingModel.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=bool(request.get("local_files_only", True)),
        ).eval().to(torch.bfloat16).to("cuda")
        model.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=bool(request.get("local_files_only", True))
        )
        sample_rate = int(model.config.audio_tokenizer_config.sample_rate)
        extractor = SpkembExtractor(str(model_path / "campplus.onnx")) if ref_path else None
        prompt_waveform, speaker_embeddings = prepare_prompt(
            torchaudio, torch, ref_path, sample_rate, extractor
        )
        text = normalize_text(request.get("text"), "text")
        generation_kwargs = {
            "prompt": request.get("prompt") or "Please generate speech based on the following description.\n",
            "text": text,
            "spk_emb": speaker_embeddings,
            "instruction": build_instruction(request),
            "prompt_waveform": prompt_waveform,
            "prompt_text": request.get("prompt_text") or request.get("ref_text"),
            "max_decode_steps": int(request.get("max_decode_steps") or 200),
            "cfg": float(request.get("cfg") if request.get("cfg") is not None else 2.0),
            "sigma": float(request.get("sigma") if request.get("sigma") is not None else 0.25),
            "temperature": float(request.get("temperature") if request.get("temperature") is not None else 0.0),
            "use_zero_spk_emb": ref_path is None,
        }
        with torch.inference_mode():
            waveform = model.generate(**generation_kwargs)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_wav), waveform.detach().cpu(), sample_rate)
    finally:
        if model is not None:
            del model
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
