"""Shared request contract for the standalone dots.tts-soar service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class CloneSynthesisRequest(BaseModel):
    """Base schema for ``/v2/dotsTTS/clone`` reference-audio cloning requests.

    The WebUI sends compatibility fields shared by several local TTS models,
    so unknown fields remain ignored.  A style prompt is rejected because
    this endpoint is cloning an uploaded voice, not designing a new voice.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def reject_style_prompt(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "style_prompt" in value:
            raise ValueError("style_prompt 不适用于 /v2/dotsTTS/clone；该接口仅用于参考音频克隆。")
        return value
