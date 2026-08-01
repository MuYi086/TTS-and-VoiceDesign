"""VoxCPM2 无参考音频音色设计的独立请求与 worker 调度。

该模块只服务主 API 的 ``/v1/voxcpm2/design`` 路由。克隆协议仍由
``voxcpm2_api.py`` 维护，避免两种 VoxCPM2 调用方式互相干扰。
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from voxcpm2_api import (
    VOXCPM2_HELPER_SCRIPT,
    VOXCPM2_MODEL_DIR,
    manager,
    normalize_optional_text,
    normalize_synthesis_text,
)


API_DIR = os.path.dirname(os.path.abspath(__file__))
VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT = os.path.join(
    API_DIR,
    "voxcpm2_voice_design_worker.py",
)


class VoxCpm2VoiceDesignRequest(BaseModel):
    """与 Qwen / MiMo 对齐的核心音色设计请求，附带 VoxCPM2 可选生成参数。"""

    voice_description: str
    text: str = "这是生成的参考音频预览。"
    save_as: Optional[str] = "designed_voice.wav"
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None
    normalize: Optional[bool] = None
    denoise: Optional[bool] = None
    retry_badcase: Optional[bool] = None
    load_denoiser: Optional[bool] = None
    optimize: Optional[bool] = None
    device: Optional[str] = None
    seed: Optional[int] = None


def build_voice_design_worker_payload(request: VoxCpm2VoiceDesignRequest) -> dict:
    """把 WebUI 请求转换为官方 ``(音色描述)正文`` worker 所需的负载。"""
    voice_description = normalize_optional_text(request.voice_description)
    if not voice_description:
        raise HTTPException(status_code=422, detail="VoxCPM2 音色设计需要 voice_description。")

    return {
        "operation": "voice_design",
        "text": normalize_synthesis_text(request.text),
        "voice_description": voice_description,
        **manager.build_generation_settings(request),
    }


def run_voxcpm2_voice_design(request: VoxCpm2VoiceDesignRequest) -> bytes:
    """串行执行独立音色设计 worker，并在请求结束后释放其模型显存。"""
    payload = build_voice_design_worker_payload(request)
    with manager.lock:
        return manager.run_worker(
            payload,
            worker_script=VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT,
        )


def voice_design_is_ready() -> bool:
    """返回主 API 可在不加载模型的前提下检查到的音色设计运行条件。"""
    return (
        os.path.isdir(VOXCPM2_MODEL_DIR)
        and os.path.isfile(VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT)
        and os.path.isfile(VOXCPM2_HELPER_SCRIPT)
    )
