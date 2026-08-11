"""One-shot LongCat-AudioDiT-3.5B voice-cloning worker.

The HTTP wrapper deliberately stays free of the heavy AudioDiT runtime.  This
process loads the official ``audiodit`` package in the dedicated Conda
environment, writes one WAV, and explicitly releases CUDA allocations before
exiting.
"""

from __future__ import annotations

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


DEFAULT_REPO_CANDIDATES = (
    Path.home() / "tts-depency/LongCat-AudioDiT",
    Path.home() / "LongCat-AudioDiT",
    Path("/tmp/LongCat-AudioDiT"),
)


def load_request(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("LongCat worker request must be a JSON object.")
    return value


def maybe_add_repo_path(repo_path: str | Path | None) -> None:
    """Make the official repository containing ``audiodit`` importable."""
    candidates = [Path(repo_path).expanduser()] if repo_path else list(DEFAULT_REPO_CANDIDATES)
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "audiodit").is_dir() and str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
            return


def require_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def require_model_path(path: str | Path) -> Path:
    model_path = require_path(path, "LongCat model path")
    if not model_path.is_dir():
        raise NotADirectoryError(f"LongCat model path is not a directory: {model_path}")
    missing = [
        name
        for name in ("config.json", "model.safetensors")
        if not (model_path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "LongCat model path is missing required file(s): "
            f"{', '.join(missing)}; path={model_path}"
        )
    return model_path


def normalize_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = re.sub(r'["“”‘’]', " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def split_text(text: str, max_chars: int) -> list[str]:
    if not text:
        raise ValueError("synthesis text is empty")
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
            chunks.extend(
                part[index : index + max_chars]
                for index in range(0, len(part), max_chars)
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


def approx_duration_from_text(text: str, max_duration: float = 30.0) -> float:
    en_dur_per_char = 0.082
    zh_dur_per_char = 0.21
    compact_text = re.sub(r"\s+", "", text)
    num_zh = 0
    num_en = 0
    num_other = 0
    for char in compact_text:
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


def import_runtime():
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        import torch
        import torch.nn.functional as F

        import audiodit  # noqa: F401  # registers AudioDiT with Transformers
        from audiodit import AudioDiTModel
        from transformers import AutoTokenizer
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "LongCat-AudioDiT runtime is not importable. Install the official "
            "requirements in the LongCat-AudioDiT-3.5B-bf16 Conda environment, "
            "make the official repository available through LONGCAT_AUDIODIT_REPO_PATH "
            f"or PYTHONPATH, and retry. Missing import: {missing}"
        ) from exc
    return AudioDiTModel, AutoTokenizer, F, librosa, np, sf, torch


def load_tokenizer(auto_tokenizer: Any, source: str, local_files_only: bool):
    kwargs = {"local_files_only": local_files_only, "fix_mistral_regex": True}
    try:
        return auto_tokenizer.from_pretrained(source, **kwargs)
    except TypeError:
        kwargs.pop("fix_mistral_regex", None)
        return auto_tokenizer.from_pretrained(source, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"无法加载 LongCat 使用的 UMT5 tokenizer: {source}。原始错误：{exc}"
        ) from exc


def resolve_tokenizer_source(
    tokenizer_path: str | None,
    model: Any,
    default_tokenizer_path: str | None,
) -> str:
    if tokenizer_path:
        return str(require_path(tokenizer_path, "LongCat tokenizer path"))
    if default_tokenizer_path and Path(default_tokenizer_path).expanduser().exists():
        return str(Path(default_tokenizer_path).expanduser().resolve())
    return str(model.config.text_encoder_model)


def require_cuda(torch: Any) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for LongCat-AudioDiT-3.5B voice cloning.")
    return "cuda"


def set_seed(torch: Any, seed: int) -> None:
    if seed < 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_vae_dtype(model: Any, torch: Any, dtype: str) -> None:
    if dtype == "float16" and hasattr(model.vae, "to_half"):
        model.vae.to_half()
        return
    target_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    model.vae.to(target_dtype)


def load_prompt_audio(path: Path, sample_rate: int, librosa: Any, torch: Any):
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if getattr(audio, "size", len(audio)) == 0:
        raise ValueError(f"reference audio is empty: {path}")
    return torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)


def prompt_latent_frames(
    model: Any,
    prompt_audio: Any,
    full_hop: int,
    device: str,
    functional: Any,
    torch: Any,
) -> int:
    off = 3
    prompt = prompt_audio.squeeze(0)
    if prompt.shape[-1] % full_hop != 0:
        prompt = functional.pad(prompt, (0, full_hop - prompt.shape[-1] % full_hop))
    prompt = functional.pad(prompt, (0, full_hop * off))
    with torch.inference_mode():
        latents = model.vae.encode(prompt.unsqueeze(0).to(device=device))
    if off:
        latents = latents[..., :-off]
    return int(latents.shape[-1])


def estimate_duration_frames(
    gen_text: str,
    prompt_text: str,
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
    approx_prompt_duration = approx_duration_from_text(prompt_text, max_duration=max_duration)
    if approx_prompt_duration > 0:
        ratio = float(np.clip(prompt_time / approx_prompt_duration, 1.0, 1.5))
        gen_duration *= ratio
    gen_duration *= duration_scale
    gen_frames = max(1, int(gen_duration * sample_rate // full_hop))
    max_frames = max(1, int(max_duration * sample_rate // full_hop))
    return min(prompt_frames + gen_frames, max_frames)


def to_mono_float32(waveform: Any, np: Any):
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
    return audio.reshape(-1)


def join_waveforms(waveforms: list[Any], sample_rate: int, pause_ms: int, np: Any):
    if not waveforms:
        raise RuntimeError("LongCat-AudioDiT returned no audio segments.")
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
    """Remove a model-generated silent prefix before a segment is joined."""
    trimmed_waveform, trimmed_samples = trim_leading_silence(
        waveform,
        sample_rate=sample_rate,
        np=np,
    )
    if trimmed_samples > 0:
        print(f"[LongCat] {label} 裁掉前导静音 {trimmed_samples / sample_rate:.2f}s")
    return trimmed_waveform


def release_cuda_memory(torch: Any) -> None:
    """Release allocator caches even when the worker is reused by a test harness."""
    gc.collect()
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        if not cuda.is_available():
            return
    except Exception as exc:
        print(f"[LongCat] CUDA availability check failed during cleanup: {exc}")
        return
    for method_name in ("synchronize", "empty_cache", "ipc_collect"):
        method = getattr(cuda, method_name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception as exc:
            # Cleanup must never mask the original inference exception.
            print(f"[LongCat] CUDA cleanup {method_name} failed: {exc}")


def synthesize(request: dict[str, Any], output_wav: Path) -> None:
    model = tokenizer = prompt_audio = inputs = output = None
    waveforms: list[Any] = []
    maybe_add_repo_path(request.get("repo_path"))
    AudioDiTModel, AutoTokenizer, functional, librosa, np, sf, torch = import_runtime()
    try:
        model_path = require_model_path(request["model_path"])
        ref_audio = require_path(request["ref_audio_path"], "reference audio")
        prompt_text = normalize_text(str(request.get("prompt_text") or ""))
        if not prompt_text:
            raise ValueError("LongCat voice cloning requires the accurate prompt_text transcript.")
        text = normalize_text(str(request.get("text") or ""))
        if not text:
            raise ValueError("synthesis text is empty")

        max_chars_per_chunk = int(request.get("max_chars_per_chunk", 180))
        pause_ms = int(request.get("pause_ms", 250))
        nfe = int(request.get("nfe", 16))
        guidance_strength = float(request.get("guidance_strength", 4.0))
        guidance_method = str(request.get("guidance_method", "apg"))
        seed = int(request.get("seed", 20260614))
        duration_scale = float(request.get("duration_scale", 1.0))
        vae_dtype = str(request.get("vae_dtype", "float16"))
        local_files_only = bool(request.get("local_files_only", True))

        if max_chars_per_chunk < 0:
            raise ValueError("max_chars_per_chunk must be >= 0")
        if pause_ms < 0:
            raise ValueError("pause_ms must be >= 0")
        if nfe < 2:
            raise ValueError("nfe must be >= 2")
        if guidance_strength < 0:
            raise ValueError("guidance_strength must be >= 0")
        if guidance_method not in {"cfg", "apg"}:
            raise ValueError("guidance_method must be cfg or apg")
        if duration_scale <= 0:
            raise ValueError("duration_scale must be > 0")
        if vae_dtype not in {"float16", "float32"}:
            raise ValueError("vae_dtype must be float16 or float32")

        device = require_cuda(torch)
        torch.backends.cudnn.benchmark = False
        set_seed(torch, seed)
        chunks = split_text(text, max_chars_per_chunk)

        started = time.perf_counter()
        model = AudioDiTModel.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
        ).to(device)
        apply_vae_dtype(model, torch, vae_dtype)
        model.eval()

        tokenizer_source = resolve_tokenizer_source(
            request.get("tokenizer_path"),
            model,
            request.get("default_tokenizer_path"),
        )
        tokenizer = load_tokenizer(AutoTokenizer, tokenizer_source, local_files_only)

        sample_rate = int(model.config.sampling_rate)
        full_hop = int(model.config.latent_hop)
        max_duration = float(model.config.max_wav_duration)
        if sample_rate != 24000:
            raise ValueError(
                "LongCat-AudioDiT-3.5B requires the official 24 kHz model configuration; "
                f"received sampling_rate={sample_rate}"
            )
        prompt_audio = load_prompt_audio(ref_audio, sample_rate, librosa, torch)
        prompt_frames = prompt_latent_frames(
            model,
            prompt_audio,
            full_hop,
            device,
            functional,
            torch,
        )
        prompt_time = prompt_frames * full_hop / sample_rate
        if prompt_time >= max_duration:
            raise ValueError(
                f"reference audio is too long for model max_wav_duration={max_duration}s: "
                f"{prompt_time:.2f}s"
            )

        print(
            f"[LongCat] chunks={len(chunks)}, sample_rate={sample_rate}, "
            f"prompt_duration={prompt_time:.2f}s, nfe={nfe}, "
            f"guidance_method={guidance_method}, vae_dtype={vae_dtype}"
        )
        with torch.inference_mode():
            for index, chunk in enumerate(chunks, start=1):
                if seed >= 0:
                    set_seed(torch, seed + index - 1)
                full_text = f"{prompt_text} {chunk}"
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
                    f"[LongCat] synthesizing chunk {index}/{len(chunks)} "
                    f"({len(chunk)} chars, duration={duration} latent frames)"
                )
                output = model(
                    input_ids=inputs.input_ids.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                    prompt_audio=prompt_audio.to(device),
                    duration=duration,
                    steps=nfe,
                    cfg_strength=guidance_strength,
                    guidance_method=guidance_method,
                )
                waveform = output.waveform.squeeze().detach().cpu().numpy()
                waveforms.append(
                    trim_generated_waveform(
                        waveform,
                        sample_rate,
                        np,
                        label=f"chunk {index}/{len(chunks)}",
                    )
                )
                del output
                output = None

        waveform = join_waveforms(waveforms, sample_rate, pause_ms, np)
        waveform, trimmed_samples = trim_leading_silence(
            waveform,
            sample_rate=sample_rate,
            np=np,
        )
        if trimmed_samples > 0:
            print(f"[LongCat] 最终音频裁掉前导静音 {trimmed_samples / sample_rate:.2f}s")
        output_wav = output_wav.expanduser().resolve()
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), waveform, sample_rate)
        print(f"[LongCat] elapsed={time.perf_counter() - started:.2f}s output={output_wav}")
    finally:
        # Explicit cleanup is useful for direct worker invocation and makes the
        # lifecycle guarantee independent of Python's process-exit timing.
        del output, inputs, prompt_audio, tokenizer, model, waveforms
        release_cuda_memory(torch)


def main() -> int:
    parser = argparse.ArgumentParser(description="LongCat-AudioDiT one-shot worker")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    args = parser.parse_args()
    try:
        synthesize(load_request(args.input_json), args.output_wav)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
