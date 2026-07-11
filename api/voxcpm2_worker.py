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
from types import SimpleNamespace
from typing import Any

from audio_trim import trim_leading_silence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot VoxCPM2 worker")
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


def normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def resolve_device(request: dict[str, Any]) -> str:
    return (normalize_optional_text(request.get("device")) or "cuda").lower()


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

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.pop("CUDA_MODULE_LOADING", None)
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


def load_voxcpm2_helpers(script_path: str) -> Any:
    helper_file = require_path(script_path, "VoxCPM2 辅助脚本")
    spec = importlib.util.spec_from_file_location("timbre_voxcpm2", helper_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 VoxCPM2 辅助脚本：{helper_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_helper_args(request: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        # /v2/synthesize is a cloning endpoint.  Never let a style prompt be
        # prepended to the target text by the shared VoxCPM helper.
        style_prompt="",
        cfg_value=float(request.get("cfg_value") or 2.0),
        inference_timesteps=int(request.get("inference_timesteps") or 10),
        load_denoiser=bool(request.get("load_denoiser", False)),
        local_files_only=bool(request.get("local_files_only", True)),
        optimize=bool(request.get("optimize", False)),
    )


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    prepare_environment(request)

    helper_script = str(request.get("voxcpm2_helper_script") or "")
    helpers = load_voxcpm2_helpers(helper_script)
    VoxCPM, np, sf, torch = helpers.import_runtime()
    requested_device = resolve_device(request)
    if not requested_device.startswith("cuda"):
        raise RuntimeError(f"VoxCPM2 仅支持 GPU 设备，当前 device={requested_device}")
    if not torch.cuda.is_available():
        raise RuntimeError("VoxCPM2 合成需要 CUDA GPU。")

    model_path = require_path(str(request.get("model_path") or ""), "模型路径")
    ref_audio_path = require_path(str(request.get("ref_audio_path") or ""), "参考音频")
    text = normalize_text(str(request.get("text") or ""))
    prompt_text = normalize_optional_text(request.get("prompt_text"))
    seed = int(request.get("seed") or 20260614)
    max_chars_per_chunk = int(request.get("max_chars_per_chunk") or 0)
    pause_ms = int(request.get("pause_ms") or 250)
    helper_args = build_helper_args(request)
    chunks = split_text(text, max_chars_per_chunk)

    model = None
    started = time.perf_counter()
    try:
        helpers.set_seed(seed, np, torch)
        print(f"[VoxCPM2 worker] 模型目录: {model_path}")
        print(f"[VoxCPM2 worker] 参考音频: {ref_audio_path}")
        print(f"[VoxCPM2 worker] 参考文本: {'provided' if prompt_text else 'not provided; reference-only cloning mode'}")
        print(f"[VoxCPM2 worker] 文本长度: {len(text)} 字, chunks={len(chunks)}")
        print(
            f"[VoxCPM2 worker] cfg_value={helper_args.cfg_value}, "
            f"inference_timesteps={helper_args.inference_timesteps}"
        )
        print(
            f"[VoxCPM2 worker] seed={seed}, load_denoiser={helper_args.load_denoiser}, "
            f"optimize={helper_args.optimize}, local_files_only={helper_args.local_files_only}, "
            f"device={requested_device}"
        )

        model = VoxCPM.from_pretrained(
            str(model_path),
            device=requested_device,
            **helpers.from_pretrained_kwargs(VoxCPM, helper_args),
        )
        sample_rate = int(helpers.resolve_sample_rate(model))

        waveforms = []
        with torch.inference_mode():
            for index, chunk in enumerate(chunks, start=1):
                print(f"[VoxCPM2 worker] 合成 chunk {index}/{len(chunks)} ({len(chunk)} chars)")
                waveforms.append(
                    model.generate(
                        **helpers.generate_kwargs(
                            model,
                            helper_args,
                            chunk,
                            ref_audio_path,
                            prompt_text,
                        )
                    )
                )

        waveform = helpers.join_waveforms(waveforms, sample_rate, pause_ms, np)
        waveform, trimmed_samples = trim_leading_silence(waveform, sample_rate, np)
        if trimmed_samples > 0:
            print(f"[VoxCPM2 worker] 裁掉前导空白 {trimmed_samples / sample_rate:.2f}s")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, sample_rate)
        elapsed = time.perf_counter() - started
        print(
            f"[VoxCPM2 worker] 完成: sample_rate={sample_rate}, "
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
