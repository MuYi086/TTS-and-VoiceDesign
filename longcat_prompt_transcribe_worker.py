#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LongCat prompt audio transcription worker")
    parser.add_argument("--input-audio", required=True, help="Prompt audio path")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--model-dir", required=True, help="SenseVoiceSmall local model path")
    parser.add_argument("--device", default="cpu", help="FunASR device, e.g. cpu or cuda:0")
    parser.add_argument("--language", default="auto", help="ASR language")
    return parser.parse_args()


def require_path(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label}不存在：{resolved}")
    return resolved


def transcribe(audio_path: Path, model_dir: Path, device: str, language: str) -> str:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    model = AutoModel(
        model=str(model_dir),
        trust_remote_code=False,
        disable_update=True,
        disable_pbar=True,
        log_level="ERROR",
        device=device,
        hub="hf",
    )
    result = model.generate(
        input=str(audio_path),
        cache={},
        language=language,
        use_itn=True,
        batch_size=1,
    )

    texts: list[str] = []
    for item in result or []:
        raw_text = str((item or {}).get("text") or "").strip()
        if not raw_text:
            continue
        normalized = rich_transcription_postprocess(raw_text).strip()
        if normalized:
            texts.append(normalized)
    merged = " ".join(texts).strip()
    if not merged:
        raise RuntimeError("SenseVoiceSmall 未返回可用转写文本。")
    return merged


def main() -> int:
    args = parse_args()
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        audio_path = require_path(args.input_audio, "参考音频")
        model_dir = require_path(args.model_dir, "转写模型目录")
        text = transcribe(audio_path, model_dir, args.device, args.language)

        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"text": text}, f, ensure_ascii=False)

        print(f"[LongCat ASR worker] 完成: audio={audio_path}, text_len={len(text)}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
