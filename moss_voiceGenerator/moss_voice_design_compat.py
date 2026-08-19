"""已发布 MOSS VoiceGenerator processor 的兼容辅助函数。"""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from typing import Any

MOSS_V1_CODEC_MODEL_TYPE = "moss-audio-tokenizer"


def _read_codec_config(codec_path: Path) -> dict[str, Any]:
    config_path = codec_path / "config.json"
    if not codec_path.is_dir():
        raise RuntimeError(f"MOSS 音频 tokenizer 目录不存在：{codec_path}")
    if not config_path.is_file():
        raise RuntimeError(
            "MOSS 音频 tokenizer 目录不完整："
            f"{codec_path} 缺少 config.json。请下载完整的 "
            "OpenMOSS-Team/MOSS-Audio-Tokenizer v1 权重，"
            "不要只创建空目录或使用 v2 目录。"
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 MOSS codec 配置：{config_path}（{exc}）") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"MOSS codec 配置必须是 JSON 对象：{config_path}")
    return payload


def _validate_codec_weights(codec_path: Path) -> None:
    index_paths = sorted(
        [*codec_path.glob("*.safetensors.index.json"), *codec_path.glob("*.bin.index.json")]
    )
    for index_path in index_paths:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map", {})
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法读取 MOSS codec 权重索引：{index_path}（{exc}）") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"MOSS codec 权重索引为空：{index_path}")
        missing = sorted(
            {
                str(codec_path / str(filename))
                for filename in weight_map.values()
                if not (codec_path / str(filename)).is_file()
                or (codec_path / str(filename)).stat().st_size == 0
            }
        )
        if missing:
            raise RuntimeError(
                "MOSS 音频 tokenizer 权重未下载完整，缺少分片："
                + ", ".join(missing[:5])
                + (" …" if len(missing) > 5 else "")
            )
        return

    weight_paths = sorted(
        [
            *codec_path.glob("*.safetensors"),
            *codec_path.glob("*.bin"),
            *codec_path.glob("*.pt"),
            *codec_path.glob("*.pth"),
        ]
    )
    if not any(path.is_file() and path.stat().st_size > 0 for path in weight_paths):
        raise RuntimeError(
            "MOSS 音频 tokenizer 目录不完整：未找到模型权重（*.safetensors 或 *.bin）。"
            "请完成 OpenMOSS-Team/MOSS-Audio-Tokenizer v1 的下载。"
        )


def validate_moss_codec_path(codec_path: str | Path) -> None:
    """在 Transformers 加载远程代码前校验本地 v1 codec。"""
    path = Path(codec_path).expanduser().resolve()
    config = _read_codec_config(path)
    if config.get("model_type") != MOSS_V1_CODEC_MODEL_TYPE:
        raise RuntimeError(
            "MOSS-VoiceGenerator 需要 MOSS-Audio-Tokenizer v1；"
            f"{path / 'config.json'} 的 model_type={config.get('model_type')!r}。"
            "请不要把 MOSS-Audio-Tokenizer-v2（48 kHz、双声道）配置给原始 1.7B 模型。"
        )
    sampling_rate = config.get("sampling_rate", config.get("sample_rate"))
    channels = config.get("number_channels", config.get("audio_channels", 1))
    try:
        sampling_rate = int(sampling_rate)
        channels = int(channels)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"MOSS codec 配置缺少有效的 sampling_rate/number_channels：{path / 'config.json'}"
        ) from exc
    if sampling_rate != 24000 or channels != 1:
        raise RuntimeError(
            "MOSS-VoiceGenerator 原始 1.7B 模型需要 MOSS-Audio-Tokenizer v1"
            "（24 kHz、单声道）；当前本地 codec 配置为 "
            f"{sampling_rate} Hz、{channels} 声道。"
        )
    _validate_codec_weights(path)


def is_moss_codec_path_ready(codec_path: str | Path) -> bool:
    try:
        validate_moss_codec_path(codec_path)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def split_sizes_from_break_positions(total_size: int, break_positions: Any) -> list[int]:
    """将 torch.where 得到的断点位置转换为 torch.split 所需的长度。"""
    if total_size < 0:
        raise ValueError("total_size 不能为负数。")
    if hasattr(break_positions, "tolist"):
        break_positions = break_positions.tolist()
    positions = [int(position) for position in break_positions]
    if any(position <= 0 or position >= total_size for position in positions):
        raise ValueError(f"break_positions 必须位于 (0, {total_size}) 内，实际为 {positions}。")
    if positions != sorted(set(positions)):
        raise ValueError(f"break_positions 必须严格递增，实际为 {positions}。")
    boundaries = [0, *positions, total_size]
    return [right - left for left, right in zip(boundaries, boundaries[1:], strict=False)]


def validate_moss_codec_compatibility(processor: Any) -> None:
    """拒绝与 VoiceGenerator 不兼容的 codec checkpoint。"""
    model_config = processor.model_config
    codec = processor.audio_tokenizer
    codec_config = getattr(codec, "config", None)
    expected_rate = int(model_config.sampling_rate)
    codec_rate = getattr(codec, "sampling_rate", None)
    if codec_rate is None and codec_config is not None:
        codec_rate = getattr(codec_config, "sampling_rate", None)
    codec_channels = getattr(codec, "number_channels", None)
    if codec_channels is None and codec_config is not None:
        codec_channels = getattr(codec_config, "number_channels", 1)
    codec_channels = 1 if codec_channels is None else codec_channels
    if codec_rate is None:
        raise RuntimeError("无法从 MOSS codec 配置中读取 sampling_rate。")
    if int(codec_rate) != expected_rate or int(codec_channels) != 1:
        raise RuntimeError(
            "MOSS-VoiceGenerator 需要 MOSS-Audio-Tokenizer v1（24 kHz、单声道）；"
            f"当前 codec 为 {int(codec_rate)} Hz、{int(codec_channels)} 声道。"
            "请勿使用 MOSS-Audio-Tokenizer-v2（48 kHz、双声道）。"
        )


def _parse_audio_codes_with_fixed_segments(self, start_length, audio_codes):
    import torch

    audio_codes = self.apply_de_delay_pattern(audio_codes)
    is_pad = (audio_codes == self.model_config.audio_pad_code).all(dim=1)
    non_pad = ~is_pad
    if not non_pad.any():
        return []
    idx = torch.nonzero(non_pad).squeeze(1)
    breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
    if breaks.numel() == 0:
        segments_idx = [idx]
    else:
        split_sizes = split_sizes_from_break_positions(int(idx.numel()), breaks)
        segments_idx = torch.split(idx, split_sizes)
    audio_codes_list = [audio_codes[segment] for segment in segments_idx]
    decoded_audio_list = self.decode_audio_codes(audio_codes_list)
    if start_length > 0 and audio_codes_list and decoded_audio_list:
        first_codes_length = audio_codes_list[0].shape[0]
        if first_codes_length > 0:
            trim_ratio = max(0.0, min(float(start_length) / float(first_codes_length), 1.0))
            first_audio = decoded_audio_list[0]
            if trim_ratio >= 1.0:
                decoded_audio_list = decoded_audio_list[1:]
            elif trim_ratio > 0.0:
                trim_samples = int(first_audio.shape[-1] * trim_ratio)
                decoded_audio_list[0] = first_audio[..., trim_samples:]
    return decoded_audio_list


def install_moss_decode_compatibility(processor: Any) -> None:
    """只修补当前 processor 实例，不修改缓存的上游源码。"""
    if getattr(processor, "_unitale_fixed_audio_parser", False):
        return
    processor._parse_audio_codes = MethodType(
        _parse_audio_codes_with_fixed_segments,
        processor,
    )
    processor._unitale_fixed_audio_parser = True
