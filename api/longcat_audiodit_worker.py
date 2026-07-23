#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


LONGCAT_REPO_ENV = "LONGCAT_REPO_PATH"
DEFAULT_LONGCAT_REPO_CANDIDATES = (
    Path(__file__).resolve().parent / "vendor/LongCat-AudioDiT",
    Path("/tmp/LongCat-AudioDiT"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot LongCat AudioDiT worker")
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


def unique_resolved_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(Path(resolved))
    return unique_paths


def iter_longcat_repo_candidates(explicit_repo_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_repo_path:
        candidates.append(Path(explicit_repo_path))

    env_repo_path = os.environ.get(LONGCAT_REPO_ENV)
    if env_repo_path:
        candidates.append(Path(env_repo_path))

    for path_text in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if path_text:
            candidates.append(Path(path_text))

    for path_text in sys.path:
        if path_text:
            candidates.append(Path(path_text))

    candidates.extend(DEFAULT_LONGCAT_REPO_CANDIDATES)
    return unique_resolved_paths(candidates)


def resolve_longcat_repo_path(explicit_repo_path: str | None) -> Path | None:
    for candidate in iter_longcat_repo_candidates(explicit_repo_path):
        resolved = candidate.expanduser().resolve()
        if (resolved / "audiodit").is_dir():
            return resolved
    return None


def maybe_add_repo_path(repo_path: Path | None) -> None:
    if repo_path is None:
        return
    path_text = str(repo_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def normalize_text(text: str) -> str:
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"[\"“”‘’]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
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


def approx_duration_from_text(text: str, max_duration: float = 30.0) -> float:
    en_dur_per_char = 0.082
    zh_dur_per_char = 0.21
    compact = re.sub(r"\s+", "", text)
    num_zh = num_en = num_other = 0

    for char in compact:
        if "\u4e00" <= char <= "\u9fff":
            num_zh += 1
        elif char.isalpha():
            num_en += 1
        else:
            num_other += 1

    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, num_zh * zh_dur_per_char + num_en * en_dur_per_char)


def to_mono_float32(waveform: Any, np: Any):
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
    return audio.reshape(-1)


def trim_leading_silence(
    waveform: Any,
    sample_rate: int,
    enabled: bool,
    threshold_db: float,
    min_silence_ms: int,
    analysis_window_ms: int,
    pre_roll_ms: int,
    max_trim_ms: int,
    np: Any,
):
    audio = to_mono_float32(waveform, np)
    if not enabled or audio.size == 0:
        return audio, 0

    analysis_window = max(1, int(sample_rate * max(analysis_window_ms, 1) / 1000))
    threshold = float(10 ** (threshold_db / 20.0))
    power = np.square(audio, dtype=np.float32)
    kernel = np.ones(analysis_window, dtype=np.float32) / analysis_window
    rms = np.sqrt(np.convolve(power, kernel, mode="same")).astype(np.float32, copy=False)
    active_indices = np.flatnonzero(rms >= threshold)
    if active_indices.size == 0:
        return audio, 0

    trim_index = int(active_indices[0])
    min_silence_samples = int(sample_rate * max(min_silence_ms, 0) / 1000)
    if trim_index < min_silence_samples:
        return audio, 0

    max_trim_samples = int(sample_rate * max(max_trim_ms, 0) / 1000)
    if max_trim_samples > 0:
        trim_index = min(trim_index, max_trim_samples)

    pre_roll_samples = int(sample_rate * max(pre_roll_ms, 0) / 1000)
    trim_start = max(0, trim_index - pre_roll_samples)
    if trim_start <= 0:
        return audio, 0

    trimmed_audio = audio[trim_start:]
    if trimmed_audio.size == 0:
        return audio, 0
    return trimmed_audio, trim_start


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, np: Any):
    if not waveforms:
        raise RuntimeError("LongCat-AudioDiT 未返回音频片段。")

    segments = [to_mono_float32(waveform, np) for waveform in waveforms]
    pause_samples = int(sample_rate * max(pause_ms, 0) / 1000)
    if pause_samples <= 0 or len(segments) == 1:
        return np.concatenate(segments)

    pause = np.zeros(pause_samples, dtype=np.float32)
    joined = []
    for index, segment in enumerate(segments):
        joined.append(segment)
        if index < len(segments) - 1:
            joined.append(pause)
    return np.concatenate(joined)


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


def prepare_environment(request: dict[str, Any]) -> Path | None:
    runtime_cache_dir = str(
        request.get("runtime_cache_dir") or Path(__file__).resolve().parent / ".cache/runtime"
    )
    hf_mirror_dir = str(request.get("hf_mirror_dir") or Path.home() / "hf-mirror")
    local_files_only = bool(request.get("local_files_only", True))
    explicit_repo_path = str(request["repo_path"]).strip() if request.get("repo_path") else None

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("HF_HOME", hf_mirror_dir)
    os.environ.setdefault("HF_MODULES_CACHE", os.path.join(runtime_cache_dir, "hf_modules"))
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(runtime_cache_dir, "numba"))
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(runtime_cache_dir, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(runtime_cache_dir, "xdg"))

    for key in ("HF_MODULES_CACHE", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        os.makedirs(os.environ[key], exist_ok=True)

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    repo_path = resolve_longcat_repo_path(explicit_repo_path)
    if repo_path is not None:
        os.environ[LONGCAT_REPO_ENV] = str(repo_path)
        current_pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(repo_path) if not current_pythonpath else str(repo_path) + os.pathsep + current_pythonpath
        maybe_add_repo_path(repo_path)
    return repo_path


def import_runtime():
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        import torch
        import torch.nn.functional as F

        import audiodit  # noqa: F401
        from audiodit import AudioDiTModel
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LongCat-AudioDiT 运行时不可导入。请确认 longcat_audiodit conda 环境已安装依赖，"
            "并通过 LONGCAT_REPO_PATH 或 PYTHONPATH 指向官方 LongCat-AudioDiT 源码目录。"
            f" 缺失导入：{exc.name or exc}"
        ) from exc
    return AudioDiTModel, AutoTokenizer, F, librosa, np, sf, torch


def load_tokenizer(AutoTokenizer: Any, tokenizer_source: str, local_files_only: bool):
    kwargs = {"local_files_only": local_files_only, "fix_mistral_regex": True}
    try:
        return AutoTokenizer.from_pretrained(tokenizer_source, **kwargs)
    except TypeError:
        kwargs.pop("fix_mistral_regex")
        return AutoTokenizer.from_pretrained(tokenizer_source, **kwargs)


def require_cuda(torch: Any) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("LongCat-AudioDiT 合成需要 CUDA GPU。")
    return "cuda"


def set_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def apply_vae_dtype(model: Any, torch: Any, dtype: str) -> None:
    if dtype == "float16" and hasattr(model.vae, "to_half"):
        model.vae.to_half()
        return
    target_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    model.vae.to(target_dtype)


def load_prompt_audio(path: Path, sample_rate: int, librosa: Any, torch: Any):
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)


def prompt_latent_frames(model: Any, prompt_audio: Any, full_hop: int, device: str, F: Any, torch: Any) -> int:
    off = 3
    prompt = prompt_audio.squeeze(0)
    if prompt.shape[-1] % full_hop != 0:
        prompt = F.pad(prompt, (0, full_hop - prompt.shape[-1] % full_hop))
    prompt = F.pad(prompt, (0, full_hop * off))

    vae_dtype = next(model.vae.parameters()).dtype
    with torch.inference_mode():
        latents = model.vae.encode(prompt.unsqueeze(0).to(device=device, dtype=vae_dtype))
    if off:
        latents = latents[..., :-off]
    return int(latents.shape[-1])


def estimate_generation_frames(
    gen_text: str,
    prompt_text: str | None,
    prompt_frames: int,
    sample_rate: int,
    full_hop: int,
    max_duration: float,
    duration_scale: float,
    np: Any,
) -> int:
    """估算单个文本块所需的生成帧，不占用参考音频的时长预算。"""
    prompt_time = prompt_frames * full_hop / sample_rate
    gen_duration = approx_duration_from_text(gen_text, max_duration=max_duration)

    if prompt_text:
        approx_prompt_duration = approx_duration_from_text(prompt_text, max_duration=max_duration)
        if approx_prompt_duration > 0:
            ratio = float(np.clip(prompt_time / approx_prompt_duration, 1.0, 1.5))
            gen_duration *= ratio

    gen_duration *= max(duration_scale, 0.1)
    max_frames = max(1, int(max_duration * sample_rate // full_hop))
    max_generation_frames = max(1, max_frames - 1)
    return min(max(1, int(gen_duration * sample_rate // full_hop)), max_generation_frames)


def prompt_frame_budget(
    chunks: list[str],
    prompt_text: str | None,
    prompt_frames: int,
    sample_rate: int,
    full_hop: int,
    max_duration: float,
    duration_scale: float,
    np: Any,
) -> int:
    """为最长文本块预留生成空间后，返回参考音频最多可占用的帧数。"""
    max_frames = max(1, int(max_duration * sample_rate // full_hop))
    generation_frames = max(
        estimate_generation_frames(
            gen_text=chunk,
            prompt_text=prompt_text,
            prompt_frames=prompt_frames,
            sample_rate=sample_rate,
            full_hop=full_hop,
            max_duration=max_duration,
            duration_scale=duration_scale,
            np=np,
        )
        for chunk in chunks
    )
    return max(1, max_frames - generation_frames)


def truncate_prompt_text(
    prompt_text: str | None,
    kept_frames: int,
    original_frames: int,
) -> str | None:
    """按参考音频保留比例截取转写前缀，避免音频与文本明显错位。"""
    if not prompt_text or kept_frames >= original_frames or original_frames <= 0:
        return prompt_text

    kept_chars = max(1, int(len(prompt_text) * kept_frames / original_frames))
    return prompt_text[:kept_chars].rstrip() or prompt_text[:1]


def estimate_duration_frames(
    gen_text: str,
    prompt_text: str | None,
    prompt_frames: int,
    sample_rate: int,
    full_hop: int,
    max_duration: float,
    duration_scale: float,
    np: Any,
) -> int:
    prompt_time = prompt_frames * full_hop / sample_rate
    available_duration = max(max_duration - prompt_time, full_hop / sample_rate)
    gen_duration = approx_duration_from_text(gen_text, max_duration=available_duration)

    if prompt_text:
        approx_prompt_duration = approx_duration_from_text(prompt_text, max_duration=max_duration)
        if approx_prompt_duration > 0:
            ratio = float(np.clip(prompt_time / approx_prompt_duration, 1.0, 1.5))
            gen_duration *= ratio

    gen_duration *= max(duration_scale, 0.1)
    gen_frames = max(1, int(gen_duration * sample_rate // full_hop))
    max_frames = max(1, int(max_duration * sample_rate // full_hop))
    return min(prompt_frames + gen_frames, max_frames)


def resolve_tokenizer_source(request: dict[str, Any], model: Any) -> str:
    raw = request.get("tokenizer_path")
    if raw is None or str(raw).strip() == "":
        return str(model.config.text_encoder_model)

    candidate = Path(str(raw)).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(raw)


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    repo_path = prepare_environment(request)
    AudioDiTModel, AutoTokenizer, F, librosa, np, sf, torch = import_runtime()

    model_path = require_path(str(request.get("model_path") or ""), "模型路径")
    ref_audio_path = require_path(str(request.get("ref_audio_path") or ""), "参考音频")
    text = normalize_text(str(request.get("text") or ""))
    prompt_text = normalize_text(str(request["prompt_text"])) if request.get("prompt_text") else None

    max_chars_per_chunk = int(request.get("max_chars_per_chunk") or 90)
    pause_ms = int(request.get("pause_ms") or 250)
    nfe = int(request.get("nfe") or 16)
    guidance_strength = float(request.get("guidance_strength") or 4.0)
    guidance_method = str(request.get("guidance_method") or "apg")
    seed = int(request.get("seed") or 1024)
    duration_scale = float(request.get("duration_scale") or 1.0)
    vae_dtype = str(request.get("vae_dtype") or "float16")
    local_files_only = bool(request.get("local_files_only", True))
    trim_leading_silence_enabled = bool(request.get("trim_leading_silence", True))
    trim_leading_silence_threshold_db = float(
        request["trim_leading_silence_threshold_db"]
    ) if request.get("trim_leading_silence_threshold_db") is not None else -42.0
    trim_leading_silence_min_ms = int(
        request["trim_leading_silence_min_ms"]
    ) if request.get("trim_leading_silence_min_ms") is not None else 120
    trim_leading_silence_analysis_window_ms = int(
        request["trim_leading_silence_analysis_window_ms"]
    ) if request.get("trim_leading_silence_analysis_window_ms") is not None else 30
    trim_leading_silence_pre_roll_ms = int(
        request["trim_leading_silence_pre_roll_ms"]
    ) if request.get("trim_leading_silence_pre_roll_ms") is not None else 40
    trim_leading_silence_max_ms = int(
        request["trim_leading_silence_max_ms"]
    ) if request.get("trim_leading_silence_max_ms") is not None else 8000

    device = require_cuda(torch)
    torch.backends.cudnn.benchmark = False
    set_seed(torch, seed)

    chunks = split_text(text, max_chars_per_chunk)
    model = None
    started = time.perf_counter()
    try:
        print(f"[LongCat worker] 模型目录: {model_path}")
        print(f"[LongCat worker] tokenizer: {request.get('tokenizer_path') or 'model.config.text_encoder_model'}")
        print(f"[LongCat worker] repo_path: {repo_path or '未解析'}")
        print(f"[LongCat worker] 参考音频: {ref_audio_path}")
        print(f"[LongCat worker] 文本长度: {len(text)} 字, chunks={len(chunks)}")
        print(f"[LongCat worker] prompt_text: {'provided' if prompt_text else 'not provided'}")
        print(
            f"[LongCat worker] trim_leading_silence={trim_leading_silence_enabled}, "
            f"threshold_db={trim_leading_silence_threshold_db}, "
            f"min_ms={trim_leading_silence_min_ms}, "
            f"pre_roll_ms={trim_leading_silence_pre_roll_ms}, "
            f"max_ms={trim_leading_silence_max_ms}"
        )

        model = AudioDiTModel.from_pretrained(str(model_path), local_files_only=local_files_only).to(device)
        apply_vae_dtype(model, torch, vae_dtype)
        model.eval()

        tokenizer_source = resolve_tokenizer_source(request, model)
        tokenizer = load_tokenizer(AutoTokenizer, tokenizer_source, local_files_only)

        sample_rate = int(model.config.sampling_rate)
        full_hop = int(model.config.latent_hop)
        max_duration = float(model.config.max_wav_duration)
        prompt_audio = load_prompt_audio(ref_audio_path, sample_rate, librosa, torch)
        prompt_frames = prompt_latent_frames(model, prompt_audio, full_hop, device, F, torch)
        original_prompt_frames = prompt_frames
        max_prompt_frames = prompt_frame_budget(
            chunks=chunks,
            prompt_text=prompt_text,
            prompt_frames=prompt_frames,
            sample_rate=sample_rate,
            full_hop=full_hop,
            max_duration=max_duration,
            duration_scale=duration_scale,
            np=np,
        )
        if prompt_frames > max_prompt_frames:
            original_prompt_text = prompt_text
            prompt_audio = prompt_audio[..., : max_prompt_frames * full_hop]
            prompt_text = truncate_prompt_text(prompt_text, max_prompt_frames, prompt_frames)
            prompt_frames = prompt_latent_frames(model, prompt_audio, full_hop, device, F, torch)
            print(
                "[LongCat worker] 参考音频超过模型总时长预算，"
                f"已截取 {original_prompt_frames} -> {prompt_frames} latent frames"
                f"（{original_prompt_frames * full_hop / sample_rate:.2f}s -> "
                f"{prompt_frames * full_hop / sample_rate:.2f}s），"
                f"prompt_text={len(original_prompt_text or '')} -> {len(prompt_text or '')} chars"
            )
        prompt_time = prompt_frames * full_hop / sample_rate
        print(f"[LongCat worker] sample_rate={sample_rate}, prompt_duration={prompt_time:.2f}s")

        waveforms = []
        with torch.inference_mode():
            for index, chunk in enumerate(chunks, start=1):
                set_seed(torch, seed + index - 1)
                full_text = f"{prompt_text} {chunk}" if prompt_text else chunk
                inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
                duration = estimate_duration_frames(
                    gen_text=chunk,
                    prompt_text=prompt_text,
                    prompt_frames=prompt_frames,
                    sample_rate=sample_rate,
                    full_hop=full_hop,
                    max_duration=max_duration,
                    duration_scale=duration_scale,
                    np=np,
                )
                print(
                    f"[LongCat worker] 合成 chunk {index}/{len(chunks)} "
                    f"({len(chunk)} chars, duration={duration} latent frames)"
                )
                output = model(
                    input_ids=inputs.input_ids.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                    prompt_audio=prompt_audio,
                    duration=duration,
                    steps=nfe,
                    cfg_strength=guidance_strength,
                    guidance_method=guidance_method,
                )
                chunk_waveform, trimmed_samples = trim_leading_silence(
                    output.waveform.squeeze().detach().cpu().numpy(),
                    sample_rate=sample_rate,
                    enabled=trim_leading_silence_enabled,
                    threshold_db=trim_leading_silence_threshold_db,
                    min_silence_ms=trim_leading_silence_min_ms,
                    analysis_window_ms=trim_leading_silence_analysis_window_ms,
                    pre_roll_ms=trim_leading_silence_pre_roll_ms,
                    max_trim_ms=trim_leading_silence_max_ms,
                    np=np,
                )
                if trimmed_samples > 0:
                    print(
                        f"[LongCat worker] chunk {index}/{len(chunks)} "
                        f"裁掉前导空白 {trimmed_samples / sample_rate:.2f}s"
                    )
                waveforms.append(chunk_waveform)

        waveform = join_waveforms(waveforms, sample_rate, pause_ms, np)
        waveform, trimmed_samples = trim_leading_silence(
            waveform,
            sample_rate=sample_rate,
            enabled=trim_leading_silence_enabled,
            threshold_db=trim_leading_silence_threshold_db,
            min_silence_ms=trim_leading_silence_min_ms,
            analysis_window_ms=trim_leading_silence_analysis_window_ms,
            pre_roll_ms=trim_leading_silence_pre_roll_ms,
            max_trim_ms=trim_leading_silence_max_ms,
            np=np,
        )
        if trimmed_samples > 0:
            print(f"[LongCat worker] 最终音频裁掉前导空白 {trimmed_samples / sample_rate:.2f}s")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, sample_rate)
        elapsed = time.perf_counter() - started
        print(f"[LongCat worker] 完成: sample_rate={sample_rate}, elapsed={elapsed:.2f}s, output={output_wav}")
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
