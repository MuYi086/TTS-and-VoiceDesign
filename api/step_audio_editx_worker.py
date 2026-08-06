#!/usr/bin/env python3
"""在独立 Conda 环境中执行一次 Step-Audio-EditX 音频编辑。"""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="One-shot Step-Audio-EditX worker")
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


def require_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label}不能为空。")
    return text


def load_upstream(code_path: Path):
    code_path = require_path(str(code_path), "Step-Audio-EditX 源码目录")
    missing = [name for name in ("tts.py", "tokenizer.py") if not (code_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Step-Audio-EditX 源码目录缺少 {', '.join(missing)}：{code_path}"
        )
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))
    try:
        import torch
        import torchaudio
        from tokenizer import StepAudioTokenizer
        from tts import StepAudioTTS
    except ImportError as exc:
        raise RuntimeError(
            "Step-Audio-EditX 运行时不可导入："
            f"{exc.name or exc}。请确认 {code_path} 对应 Conda 环境已安装官方依赖和可运行的 vLLM。"
        ) from exc
    return StepAudioTokenizer, StepAudioTTS, torch, torchaudio


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    if request.get("local_files_only", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    StepAudioTokenizer, StepAudioTTS, torch, torchaudio = load_upstream(
        Path(str(request.get("code_path") or ""))
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Step-Audio-EditX 需要 CUDA GPU。")

    model_path = require_path(str(request.get("model_path") or ""), "Step-Audio-EditX 模型目录")
    tokenizer_path = require_path(str(request.get("tokenizer_path") or ""), "Step-Audio-Tokenizer 模型目录")
    prompt_wav_path = require_path(str(request.get("prompt_wav_path") or ""), "prompt 音频")
    edit_type = require_text(request.get("edit_type"), "edit_type")
    prompt_text = request.get("prompt_text")
    generated_text = request.get("generated_text")
    if edit_type not in {"denoise", "vad"}:
        prompt_text = require_text(prompt_text, "prompt_text")
        generated_text = require_text(generated_text or prompt_text, "generated_text")

    model = None
    try:
        tokenizer = StepAudioTokenizer(str(tokenizer_path), model_source="local")
        model = StepAudioTTS(
            str(model_path),
            tokenizer,
            model_source="local",
            quantization=None,
            tensor_parallel_size=1,
            gpu_memory_utilization=float(request.get("gpu_memory_utilization") or 0.5),
            max_model_len=int(request.get("max_model_len") or 3072),
            enforce_eager=bool(request.get("enforce_eager", True)),
            dtype=str(request.get("dtype") or "bfloat16"),
            max_num_seqs=int(request.get("max_num_seqs") or 1),
            cosyvoice_dtype=str(request.get("cosyvoice_dtype") or "bfloat16"),
            cosyvoice_cuda_graph=bool(request.get("cosyvoice_cuda_graph", False)),
        )
        output_audio, sample_rate = model.edit(
            prompt_wav_path=str(prompt_wav_path),
            prompt_text=prompt_text,
            target_text=generated_text,
            edit_type=edit_type,
            edit_info=str(request.get("edit_info") or ""),
        )
        waveform = output_audio if isinstance(output_audio, torch.Tensor) else torch.as_tensor(output_audio)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_wav), waveform.detach().cpu(), int(sample_rate))
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
