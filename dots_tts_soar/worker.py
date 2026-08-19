#!/usr/bin/env python3
"""dots.tts-soar 一次性语音克隆 worker。

HTTP 服务刻意不导入重型 dots.tts 运行时，而是将单个请求序列化后交给本项目
uv 环境执行；请求完成后进程退出，从而释放模型的 CUDA 上下文。
"""

from __future__ import annotations

# dots.tts 的重型运行时只在此进程导入；worker 退出后释放模型 CUDA 上下文。
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from audio_trim import trim_leading_silence


def parse_args() -> argparse.Namespace:
    """解析一次 dots 请求的 JSON 和输出 WAV 路径。"""
    parser = argparse.ArgumentParser(description="One-shot dots.tts-soar worker")
    parser.add_argument("--input-json", required=True, help="Request JSON file path")
    parser.add_argument("--output-wav", required=True, help="Output WAV file path")
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取并校验 JSON 对象请求。"""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def require_path(path: str, label: str) -> Path:
    """确认参考音频、模型或脚本路径存在。"""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label}不存在：{resolved}")
    return resolved


def normalize_optional_text(value: Any) -> str | None:
    """把可选字段规范化为字符串或 ``None``。"""
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def normalize_text(text: str) -> str:
    """清理待合成文本，避免空片段进入模型。"""
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise RuntimeError("text 不能为空。")
    return normalized


def split_text(text: str, max_chars: int) -> list[str]:
    """优先按标点切分长文本，控制 dots 模型单次输入长度。"""
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
    """对没有标点的超长句做硬切分。"""
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
            chunks.extend(
                part[index : index + max_chars] for index in range(0, len(part), max_chars)
            )
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


def import_runtime():
    """延迟导入官方 dots runtime，保持 API 进程无模型依赖。"""
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from dots_tts.runtime import DotsTtsRuntime
        from dots_tts.utils.logging import configure_logging
        from dots_tts.utils.util import seed_everything
    except ImportError as exc:
        raise RuntimeError(
            "dots.tts-soar runtime 无法导入。请确认 dots_tts_soar uv 环境已安装官方 "
            f"dots.tts 包。缺少导入：{exc.name or exc}"
        ) from exc
    return DotsTtsRuntime, configure_logging, np, seed_everything, sf, torch


def clear_cuda_cache(torch: Any) -> None:
    """清空 CUDA cache；最终释放由 worker 进程退出保证。"""
    gc.collect()
    try:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[dots.tts-soar worker] CUDA cache 已清理")
    except Exception as exc:
        print(f"[dots.tts-soar worker] CUDA cache 清理失败: {exc}", file=sys.stderr)


def prepare_environment(request: dict[str, Any]) -> None:
    """设置离线模式、模型缓存和 CUDA allocator 环境变量。"""
    runtime_cache_dir = str(
        request.get("runtime_cache_dir") or Path(__file__).resolve().parent / ".cache/runtime"
    )
    hf_mirror_dir = str(request.get("hf_mirror_dir") or Path.home() / "hf-mirror")
    local_files_only = bool(request.get("local_files_only", True))

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


def to_mono_float32(audio: Any, np: Any) -> Any:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0 if waveform.shape[0] <= 2 else 1)
    return waveform.reshape(-1)


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, np: Any) -> Any:
    """在分段波形之间插入静音并拼接。"""
    if not waveforms:
        raise RuntimeError("dots.tts-soar 未返回音频片段。")

    segments = [to_mono_float32(waveform, np) for waveform in waveforms]
    pause_samples = int(sample_rate * max(pause_ms, 0) / 1000)
    if pause_samples <= 0 or len(segments) == 1:
        return np.concatenate(segments)

    pause = np.zeros(pause_samples, dtype=np.float32)
    joined: list[Any] = []
    for index, segment in enumerate(segments):
        joined.append(segment)
        if index < len(segments) - 1:
            joined.append(pause)
    return np.concatenate(joined)


def trim_generated_waveform(
    waveform: Any,
    sample_rate: int,
    np: Any,
    *,
    label: str,
) -> Any:
    """在拼接片段前裁剪模型生成的前导静音。"""
    trimmed_waveform, trimmed_samples = trim_leading_silence(
        waveform,
        sample_rate=sample_rate,
        np=np,
    )
    if trimmed_samples > 0:
        print(f"[dots.tts-soar worker] {label} 裁掉前导静音 {trimmed_samples / sample_rate:.2f}s")
    return trimmed_waveform


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    """加载 dots 模型，按文本片段生成克隆语音并写出 WAV。"""
    if str(request.get("operation") or "clone") != "clone":
        raise RuntimeError("dots.tts-soar worker 只接受 operation=clone。")

    prepare_environment(request)
    DotsTtsRuntime, configure_logging, np, seed_everything, sf, torch = import_runtime()

    model_path = require_path(str(request.get("model_path") or ""), "模型路径")
    ref_audio_path = require_path(str(request.get("ref_audio_path") or ""), "参考音频")
    text = normalize_text(str(request.get("text") or ""))
    prompt_text = normalize_optional_text(request.get("prompt_text"))
    language_value = normalize_optional_text(request.get("language"))
    language = language_value
    if language_value and language_value.lower() == "auto_detect":
        language = "auto_detect"

    if not torch.cuda.is_available():
        raise RuntimeError("dots.tts-soar 合成需要 CUDA GPU。")

    configure_logging()
    seed_everything(int(request.get("seed", 42)))
    chunks = split_text(text, int(request.get("max_chars_per_chunk", 120)))
    output_wav = output_wav.expanduser().resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    precision = str(request.get("precision") or "bfloat16")
    num_steps = int(request.get("num_steps", 10))
    guidance_scale = float(request.get("guidance_scale", 1.2))
    speaker_scale = float(request.get("speaker_scale", 1.5))
    max_generate_length = int(request.get("max_generate_length", 500))
    pause_ms = int(request.get("pause_ms", 250))
    template_name = normalize_optional_text(request.get("template_name"))
    ode_method = str(request.get("ode_method") or "euler")
    normalize_text_flag = bool(request.get("normalize_text", False))
    profile_inference = bool(request.get("profile_inference", False))

    print(f"[dots.tts-soar worker] 模型目录: {model_path}")
    print(f"[dots.tts-soar worker] 参考音频: {ref_audio_path}")
    print(
        f"[dots.tts-soar worker] 参考文本: {'provided' if prompt_text else 'not provided; x-vector-only cloning mode'}"
    )
    print(f"[dots.tts-soar worker] 文本长度: {len(text)} 字, chunks={len(chunks)}")
    print(
        f"[dots.tts-soar worker] precision={precision}, num_steps={num_steps}, "
        f"guidance_scale={guidance_scale}, speaker_scale={speaker_scale}, language={language or 'none'}"
    )

    runtime = None
    waveforms: list[Any] = []
    try:
        started = time.perf_counter()
        runtime = DotsTtsRuntime.from_pretrained(
            str(model_path),
            precision=precision,
            max_generate_length=max_generate_length,
        )

        with torch.inference_mode():
            for index, chunk in enumerate(chunks, start=1):
                print(
                    f"[dots.tts-soar worker] synthesizing chunk {index}/{len(chunks)} ({len(chunk)} chars)"
                )
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
                waveform = result["audio"].float().cpu().squeeze().numpy()
                waveforms.append(
                    trim_generated_waveform(
                        waveform,
                        int(runtime.sample_rate),
                        np,
                        label=f"chunk {index}/{len(chunks)}",
                    )
                )

        sample_rate = int(runtime.sample_rate)
        waveform = join_waveforms(waveforms, sample_rate, pause_ms, np)
        waveform, trimmed_samples = trim_leading_silence(
            waveform,
            sample_rate=sample_rate,
            np=np,
        )
        if trimmed_samples > 0:
            print(
                f"[dots.tts-soar worker] 最终音频裁掉前导静音 {trimmed_samples / sample_rate:.2f}s"
            )
        sf.write(str(output_wav), waveform, sample_rate)
        print(
            f"[dots.tts-soar worker] elapsed={time.perf_counter() - started:.2f}s, "
            f"sample_rate={sample_rate}, output={output_wav}"
        )
    finally:
        runtime = None
        waveforms.clear()
        clear_cuda_cache(torch)


def main() -> int:
    """执行一次 dots worker，并在失败时返回非零退出码。"""
    args = parse_args()
    try:
        synthesize(load_request(args.input_json), Path(args.output_wav))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
