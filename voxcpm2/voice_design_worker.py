#!/usr/bin/env python3
"""VoxCPM2 无参考音频音色设计的一次性 worker。

此文件刻意不处理参考音频、prompt_text 或克隆模式；对应能力由
``worker.py`` 单独维护。
"""

from __future__ import annotations

import inspect
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from audio_trim import trim_leading_silence
from worker import (
    build_helper_args,
    clear_cuda_cache,
    load_request,
    load_voxcpm2_helpers,
    normalize_optional_text,
    normalize_text,
    parse_args,
    prepare_environment,
    require_path,
    resolve_device,
)


def build_voice_design_generate_kwargs(
    model: Any,
    args: Any,
    text: str,
    seed: int,
) -> dict[str, Any]:
    """按已安装 VoxCPM 版本的签名构造官方音色设计 ``generate`` 调用。"""
    options = {
        "text": text,
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
        "normalize": getattr(args, "normalize", False),
        "denoise": getattr(args, "denoise", False),
        "retry_badcase": getattr(args, "retry_badcase", True),
    }
    signature = inspect.signature(model.generate)
    implementation = getattr(model, "_generate", None)
    implementation_signature = inspect.signature(implementation) if implementation else signature
    # worker 已调用 set_seed()；仅在当前版本公开支持时再显式传入，兼容旧包装器。
    if seed >= 0 and (
        "seed" in signature.parameters or "seed" in implementation_signature.parameters
    ):
        options["seed"] = seed
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return options
    return {key: value for key, value in options.items() if key in signature.parameters}


def synthesize_voice_design(request: dict[str, Any], output_wav: Path) -> None:
    """生成单段无参考音频的 VoxCPM2 设计音色。"""
    prepare_environment(request)

    helper_script = str(request.get("voxcpm2_helper_script") or "")
    helpers = load_voxcpm2_helpers(helper_script)
    VoxCPM, np, sf, torch = helpers.import_runtime()
    requested_device = resolve_device(request)
    if not requested_device.startswith("cuda"):
        raise RuntimeError(f"VoxCPM2 音色设计仅支持 GPU 设备，当前 device={requested_device}")
    if not torch.cuda.is_available():
        raise RuntimeError("VoxCPM2 音色设计需要 CUDA GPU。")

    if str(request.get("operation") or "voice_design") != "voice_design":
        raise RuntimeError("VoxCPM2 音色设计 worker 只接受 operation=voice_design。")

    model_path = require_path(str(request.get("model_path") or ""), "模型路径")
    text = normalize_text(str(request.get("text") or ""))
    voice_description = normalize_optional_text(request.get("voice_description"))
    if not voice_description:
        raise RuntimeError("VoxCPM2 音色设计需要 voice_description。")
    seed_value = request.get("seed")
    seed = int(seed_value if seed_value is not None else -1)
    seed_label = str(seed) if seed >= 0 else "random"
    helper_args = build_helper_args(request)
    model_text = f"({voice_description}){text}"

    model = None
    started = time.perf_counter()
    try:
        helpers.set_seed(seed, np, torch)
        print(f"[VoxCPM2 VoiceDesign worker] 模型目录: {model_path}")
        print(
            f"[VoxCPM2 VoiceDesign worker] cfg_value={helper_args.cfg_value}, "
            f"inference_timesteps={helper_args.inference_timesteps}, seed={seed_label}"
        )
        print(f"[VoxCPM2 VoiceDesign worker] 最终模型文本: {model_text}")

        model = VoxCPM.from_pretrained(
            str(model_path),
            device=requested_device,
            **helpers.from_pretrained_kwargs(VoxCPM, helper_args),
        )
        sample_rate = int(helpers.resolve_sample_rate(model))
        with torch.inference_mode():
            generate_options = build_voice_design_generate_kwargs(
                model,
                helper_args,
                model_text,
                seed,
            )
            waveform = model.generate(**generate_options)

        waveform, trimmed_samples = trim_leading_silence(waveform, sample_rate, np)
        if trimmed_samples > 0:
            print(f"[VoxCPM2 VoiceDesign worker] 裁掉前导空白 {trimmed_samples / sample_rate:.2f}s")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, sample_rate)
        print(
            f"[VoxCPM2 VoiceDesign worker] 完成: sample_rate={sample_rate}, "
            f"elapsed={time.perf_counter() - started:.2f}s, output={output_wav}"
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
        synthesize_voice_design(request, Path(args.output_wav).expanduser().resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
