#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from audio_trim import trim_leading_silence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot IndexTTS2 worker")
    parser.add_argument("--input-json", required=True, help="Request JSON file path")
    parser.add_argument("--output-wav", required=True, help="Output wav file path")
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_directory(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label}不存在：{resolved}")
    return resolved


def require_file(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}不存在：{resolved}")
    return resolved


def prepare_environment(request: dict[str, Any]) -> None:
    runtime_cache_dir = Path(str(request["runtime_cache_dir"])).expanduser().resolve()
    hf_mirror_dir = Path(str(request["hf_mirror_dir"])).expanduser().resolve()
    runtime_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("HF_HOME", str(hf_mirror_dir))
    os.environ.setdefault("HF_MODULES_CACHE", str(runtime_cache_dir / "hf_modules"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(runtime_cache_dir / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(runtime_cache_dir / "xdg"))
    if bool(request.get("local_files_only", True)):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    for env_name in ("HF_MODULES_CACHE", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)

    code_dir = require_directory(str(request.get("code_dir") or ""), "IndexTTS2 代码目录")
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def import_runtime():
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from indextts.infer_v2 import IndexTTS2
    except ImportError as exc:
        raise RuntimeError(
            "IndexTTS2 运行时不可导入。请确认 unitale-tts-local 环境已安装 "
            f"IndexTTS2、torch、numpy、soundfile 所需依赖。缺失导入：{exc.name or exc}"
        ) from exc
    return IndexTTS2, np, sf, torch


def assert_cuda_ready(torch: Any, device: Any) -> None:
    requested_device = str(device or "").strip().lower()
    if requested_device == "cpu":
        return
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 {device}，但当前 worker 无法使用 CUDA。")
    if not torch.cuda.is_available():
        return

    probe_device = device if requested_device.startswith("cuda") else "cuda"
    try:
        probe = torch.empty(1, device=probe_device)
        probe.fill_(1)
        del probe
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"IndexTTS2 worker CUDA 自检失败: {exc}") from exc


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


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    prepare_environment(request)
    IndexTTS2, np, sf, torch = import_runtime()

    model_dir = require_directory(str(request.get("model_dir") or ""), "IndexTTS2 模型目录")
    cfg_path = require_file(str(request.get("cfg_path") or ""), "IndexTTS2 配置")
    ref_audio_path = require_file(str(request.get("ref_audio_path") or ""), "参考音频")
    aux_paths = dict(request.get("aux_paths") or {})
    text = str(request.get("text") or "").strip()
    if not text:
        raise RuntimeError("text 不能为空。")

    device = request.get("device") or None
    use_fp16 = bool(request.get("use_fp16", True))
    use_cuda_kernel = bool(request.get("use_cuda_kernel", False))
    num_beams = int(request.get("num_beams") or 1)
    emo_text = request.get("emo_text") or None
    emo_vector = request.get("emo_vector")

    assert_cuda_ready(torch, device)
    model = None
    started = time.perf_counter()
    try:
        print(f"[IndexTTS2 worker] 模型目录: {model_dir}")
        print(f"[IndexTTS2 worker] 参考音频: {ref_audio_path}")
        print(
            f"[IndexTTS2 worker] device={device or 'auto'}, fp16={use_fp16}, "
            f"cuda_kernel={use_cuda_kernel}, num_beams={num_beams}"
        )
        model = IndexTTS2(
            model_dir=str(model_dir),
            cfg_path=str(cfg_path),
            aux_paths=aux_paths,
            device=device,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
        )
        print(
            "[IndexTTS2 worker] 模型就绪: "
            f"device={model.device}, fp16={model.use_fp16}, cuda_kernel={model.use_cuda_kernel}"
        )
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        model.infer(
            spk_audio_prompt=str(ref_audio_path),
            text=text,
            output_path=str(output_wav),
            emo_vector=emo_vector,
            emo_text=emo_text,
            use_emo_text=bool(emo_text),
            emo_alpha=0.6,
            num_beams=num_beams,
        )

        waveform, sample_rate = sf.read(str(output_wav), dtype="float32", always_2d=True)
        waveform, trimmed_samples = trim_leading_silence(waveform, sample_rate, np)
        if trimmed_samples > 0:
            print(f"[IndexTTS2 worker] 裁掉前导空白 {trimmed_samples / sample_rate:.2f}s")
        sf.write(str(output_wav), waveform, sample_rate, format="WAV")
        elapsed = time.perf_counter() - started
        print(
            f"[IndexTTS2 worker] 完成: sample_rate={sample_rate}, "
            f"elapsed={elapsed:.2f}s, output={output_wav}"
        )
    finally:
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
