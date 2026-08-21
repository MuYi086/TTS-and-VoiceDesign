#!/usr/bin/env python3
"""FireRedTTS3 一次性推理 worker。

只有该进程导入官方源码、Torch 和 Transformers。请求结束时显式释放模型和 CUDA
缓存，然后进程组被父进程回收，以防 Base/Instruct 或其它本地模型长期驻留显存。
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析请求 JSON 和输出 WAV 路径。"""
    parser = argparse.ArgumentParser(description="One-shot FireRedTTS3 worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def load_request(path: str | Path) -> dict[str, Any]:
    """读取请求并确认顶层结构为对象。"""
    with open(path, encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("FireRedTTS3 worker request must be a JSON object.")
    return value


def require_directory(path: str | Path, label: str) -> Path:
    """校验本地源码或权重目录存在。"""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} 不存在或不是目录: {resolved}")
    return resolved


def require_model(path: str | Path) -> Path:
    """校验 FireRedTTS3 镜像的核心权重结构。"""
    model_path = require_directory(path, "FireRedTTS3 模型目录")
    required = (
        "fireredtts3_base/config.json",
        "fireredtts3_base/model.safetensors",
        "fireredtts3_instruct/config.json",
        "fireredtts3_instruct/model.safetensors",
        "redae/config.json",
        "redae/model.safetensors",
        "campp/campplus_voxceleb.bin",
        "text_tokenizer/tokenizer.json",
        "text_tokenizer/tokenizer_config.json",
        "text_tokenizer/vocab.json",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError("FireRedTTS3 模型目录缺少文件: " + ", ".join(missing))
    return model_path


def flash_attention_available() -> bool:
    """检查 FlashAttention 扩展能否真正导入，而不只检查包目录是否存在。"""
    try:
        importlib.import_module("flash_attn")
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"[FireRedTTS3 worker] FlashAttention 不可用，回退到 SDPA: {exc}", file=sys.stderr)
        return False
    return True


def resolve_attention_implementation(requested: Any) -> str:
    """选择注意力后端，避免损坏的 FlashAttention 二进制阻断模型加载。"""
    implementation = str(requested or "auto").strip().lower()
    if implementation == "auto":
        implementation = "flash_attention_2"
    if implementation not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError(
            "FireRedTTS3 attention implementation 必须是 auto、flash_attention_2、sdpa 或 eager。"
        )
    if implementation == "flash_attention_2" and not flash_attention_available():
        return "sdpa"
    return implementation


def install_attention_compatibility(transformers_module: Any, implementation: str) -> None:
    """把官方源码硬编码的 FlashAttention 配置改为已验证的后端。

    FireRedTTS3 官方源码直接构造 Qwen3Config，未暴露 attention 参数；在这里做进程内
    兼容补丁，避免修改外部 editable 源码，同时让损坏的 CUDA 扩展自动回退到 SDPA。
    """
    qwen3_config = transformers_module.Qwen3Config
    original_init = qwen3_config.__init__
    if getattr(original_init, "_firered_tts3_attention_compatibility", False):
        return

    @wraps(original_init)
    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = implementation
        original_init(self, *args, **kwargs)

    patched_init._firered_tts3_attention_compatibility = True
    qwen3_config.__init__ = patched_init


def normalize_audio(audio: Any):
    """将上游返回的波形移到 CPU，并确保具有声道维度。"""
    if not hasattr(audio, "detach"):
        raise TypeError("FireRedTTS3 未返回 Torch waveform。")
    waveform = audio.detach().float().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"FireRedTTS3 waveform 维度异常: {tuple(waveform.shape)}")
    return waveform


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    """加载一个 FireRed 变体、生成 WAV 并在 finally 中清理 CUDA。"""
    operation = str(request.get("operation") or "").strip().lower()
    if operation not in {"clone", "timbre"}:
        raise ValueError(f"不支持的 FireRedTTS3 operation: {operation}")
    model_path = require_model(str(request.get("model_path") or ""))
    code_path = require_directory(str(request.get("code_path") or ""), "FireRedTTS3 官方源码目录")
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

    # 官方 core.py 只在 worker 中导入；API 父进程因此不会创建 CUDA 上下文。
    import torch
    import torchaudio
    import transformers

    attention_implementation = resolve_attention_implementation(request.get("attn_implementation"))
    install_attention_compatibility(transformers, attention_implementation)
    from fireredtts3.core import FireRedTTS3, FireRedTTS3Instruct

    model = None
    try:
        use_fasttext = bool(request.get("use_fasttext", False))
        use_wetext = bool(request.get("use_wetext", True))
        language = str(request.get("language") or "Chinese")
        common_kwargs = {
            "use_fasttext": use_fasttext,
            "use_llm_tn": False,
            "use_wetext": use_wetext,
        }
        print(
            f"[FireRedTTS3 worker] attention implementation: {attention_implementation}",
            flush=True,
        )
        # 官方 `_flow_one_step` 未加 autocast；模型权重是 bfloat16，需在整个生成调用外
        # 保持 CUDA autocast，避免时间步 embedding 以 float32 进入 bfloat16 Linear。
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if operation == "clone":
                ref_audio_path = (
                    Path(str(request.get("ref_audio_path") or "")).expanduser().resolve()
                )
                if not ref_audio_path.is_file():
                    raise FileNotFoundError(f"参考音频不存在: {ref_audio_path}")
                prompt_text = str(request.get("prompt_text") or "").strip()
                text = str(request.get("text") or "").strip()
                if not prompt_text or not text:
                    raise ValueError("FireRedTTS3 clone 需要非空 prompt_text 和 text。")
                prompt_audio, prompt_audio_sr = torchaudio.load(str(ref_audio_path))
                print(f"[FireRedTTS3 worker] 加载 Base: {model_path}", flush=True)
                model = FireRedTTS3(str(model_path), **common_kwargs)
                gen_audio, sample_rate = model.generate(
                    language=language,
                    prompt_text=prompt_text,
                    prompt_audio=prompt_audio,
                    prompt_audio_sr=int(prompt_audio_sr),
                    text=text,
                    stop_threshold=float(request.get("stop_threshold", 0.5)),
                    n_timesteps=int(request.get("n_timesteps", 10)),
                    inference_cfg=float(request.get("inference_cfg", 2.0)),
                    seed=int(request.get("seed", 1234)),
                    do_tn=bool(request.get("do_tn", True)),
                )
            else:
                instruction = str(request.get("instruction") or "").strip()
                text = str(request.get("text") or "").strip()
                if not instruction or not text:
                    raise ValueError("FireRedTTS3 timbre 需要非空 instruction 和 text。")
                print(f"[FireRedTTS3 worker] 加载 Instruct: {model_path}", flush=True)
                model = FireRedTTS3Instruct(str(model_path), **common_kwargs)
                gen_audio, sample_rate, voice_plan = model.generate_voice_design(
                    instruction=instruction,
                    text=text,
                    language=language,
                    n_timesteps=int(request.get("n_timesteps", 10)),
                    inference_cfg=float(request.get("inference_cfg", 1.2)),
                    seed=int(request.get("seed", 2)),
                    do_tn=bool(request.get("do_tn", True)),
                )
                print(f"[FireRedTTS3 worker] voice plan: {voice_plan}", flush=True)

        waveform = normalize_audio(gen_audio)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_wav), waveform, int(sample_rate))
    finally:
        if model is not None:
            del model
        gc.collect()
        # 官方模型固定使用 CUDA；清理放在 finally，保证异常也不会长期保留显存。
        try:
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            print(f"[FireRedTTS3 worker] CUDA 清理跳过: {exc}", file=sys.stderr)


def main() -> int:
    """执行一次 worker 并将异常打印给 HTTP 父进程。"""
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
