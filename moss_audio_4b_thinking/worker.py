#!/usr/bin/env python3
"""MOSS-Audio-4B-Thinking 一次性音频理解 worker。"""

from __future__ import annotations

# 模型、Torch 与 MOSS-Audio 上游源码只在这个一次性进程中导入，退出后释放显存。
import gc
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args():
    """解析 worker JSON 输入和文本结果输出路径。"""
    import argparse

    parser = argparse.ArgumentParser(description="One-shot MOSS-Audio-4B-Thinking worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取并确认 worker 输入 JSON 顶层为对象。"""
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("worker input JSON 必须是对象。")
    return payload


def require_path(path: str, label: str, *, directory: bool = False) -> Path:
    """验证文件或目录存在，并返回规范化绝对路径。"""
    resolved = Path(path).expanduser().resolve()
    expected = resolved.is_dir() if directory else resolved.is_file()
    if not expected:
        kind = "目录" if directory else "文件"
        raise FileNotFoundError(f"{label}{kind}不存在: {resolved}")
    return resolved


def require_text(value: Any, label: str) -> str:
    """提取非空文本字段并去除首尾空白。"""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} 不能为空。")
    return text


def parse_bool(value: Any, default: bool = False) -> bool:
    """兼容 JSON 布尔值与环境变量式字符串。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_runtime(dependency_path: Path):
    """从指定 MOSS-Audio 源码目录加载模型、处理器与音频读取函数。"""
    dependency_root = require_path(str(dependency_path), "MOSS-Audio 依赖源码", directory=True)
    source_path = dependency_root / "src"
    required = ("audio_io.py", "modeling_moss_audio.py", "processing_moss_audio.py")
    missing = [name for name in required if not (source_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MOSS-Audio 依赖源码目录缺少 {', '.join(missing)}: {source_path}")
    dependency_string = str(dependency_root)
    if dependency_string not in sys.path:
        sys.path.insert(0, dependency_string)

    try:
        import torch
        from src.audio_io import load_audio
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor
    except ImportError as exc:
        raise RuntimeError(
            "MOSS-Audio 运行时导入失败，请先为 moss_audio_4b_thinking 执行 uv sync，"
            "并确认 MOSS_AUDIO_4B_THINKING_DEPENDENCY_PATH 指向官方源码。"
            f"缺少或无法导入: {exc.name or exc}"
        ) from exc
    return torch, load_audio, MossAudioModel, MossAudioProcessor


def resolve_device(torch, device: str) -> str:
    """验证请求设备可用性，避免加载 4B 模型后才暴露配置错误。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，MOSS-Audio-4B-Thinking 默认要求 GPU。")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("请求使用 MPS，但当前 PyTorch 未检测到可用的 MPS 设备。")
    return device


def build_generation_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    """构造生成参数；非采样模式不传入无效的采样字段。"""
    do_sample = parse_bool(request.get("do_sample"), False)
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(request.get("max_new_tokens") or 1024),
        "num_beams": 1,
        "use_cache": True,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs.update(
            temperature=float(request.get("temperature") or 1.0),
            top_p=float(request.get("top_p") or 1.0),
            top_k=int(request.get("top_k") or 50),
        )
    return kwargs


def load_audio_compat(load_audio, audio_path: Path, sample_rate: int):
    """调用官方读取器，并仅在 TorchCodec 缺失时使用 ffmpeg 解码回退。"""
    try:
        return load_audio(str(audio_path), sample_rate=sample_rate)
    except (ImportError, RuntimeError) as exc:
        if "torchcodec" not in str(exc).lower():
            raise

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("ffmpeg 音频读取回退需要 numpy。") from exc

    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "torchaudio 需要 TorchCodec，且系统中找不到 ffmpeg 音频读取回退工具。"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg 无法读取音频 {audio_path}: {detail}") from exc

    audio = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg 未输出音频采样: {audio_path}")
    print("notice: TorchCodec 不可用，已使用 ffmpeg 音频读取回退。")
    return audio


def strip_thinking(text: str) -> str:
    """移除完整 ``<think>...</think>`` 段，仅保留最终回答。"""
    match = re.search(r"</think>\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip() if match is None else match.group(1).strip()


def write_result(path: Path, payload: dict[str, Any]) -> None:
    """写入 worker 结果 JSON，供父 HTTP 进程在 worker 退出后读取。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())


def understand_audio(request: dict[str, Any]) -> dict[str, Any]:
    """加载 MOSS-Audio-4B-Thinking，理解一个音频文件并返回文本结果。"""
    if parse_bool(request.get("local_files_only"), True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_path = require_path(
        str(request.get("model_path") or ""),
        "MOSS-Audio-4B-Thinking 模型",
        directory=True,
    )
    audio_path = require_path(str(request.get("audio_path") or ""), "待理解音频")
    prompt = require_text(request.get("prompt"), "prompt")
    torch, load_audio, MossAudioModel, MossAudioProcessor = load_runtime(
        Path(str(request.get("dependency_path") or ""))
    )
    device = resolve_device(torch, str(request.get("device") or "cuda:0"))
    model = None
    processor = None
    started = time.perf_counter()
    try:
        model = MossAudioModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            dtype=str(request.get("dtype") or "auto"),
            device_map=device,
            local_files_only=parse_bool(request.get("local_files_only"), True),
        )
        model.eval()
        processor = MossAudioProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            enable_time_marker=parse_bool(request.get("enable_time_marker"), True),
            local_files_only=parse_bool(request.get("local_files_only"), True),
        )
        raw_audio = load_audio_compat(load_audio, audio_path, int(processor.config.mel_sr))
        inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
        inputs = inputs.to(model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **build_generation_kwargs(request))

        input_length = inputs["input_ids"].shape[1]
        answer = processor.decode(
            generated_ids[0, input_length:],
            skip_special_tokens=True,
        ).strip()
        if parse_bool(request.get("strip_thinking"), False):
            answer = strip_thinking(answer)
        if not answer:
            raise RuntimeError("MOSS-Audio-4B-Thinking 返回了空文本。")
        return {
            "text": answer,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                print(f"[MOSS-Audio-4B-Thinking worker] CUDA synchronize 跳过: {exc}")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> int:
    """执行一次音频理解任务，并将成功结果写入输出 JSON。"""
    args = parse_args()
    try:
        result = understand_audio(load_request(args.input_json))
        write_result(Path(args.output_json).expanduser().resolve(), result)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
