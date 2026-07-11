"""Shared request contract for voice-cloning synthesis endpoints."""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class CloneSynthesisRequest(BaseModel):
    """Base schema for ``/v2/synthesize`` voice-cloning requests.

    The WebUI may send model-specific compatibility fields, so unknown fields
    remain ignored.  ``style_prompt`` is deliberately an exception: synthesis
    is only for cloning an uploaded reference voice, not voice design, and a
    style prompt can be mistaken for text that should be spoken.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def reject_style_prompt(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "style_prompt" in value:
            raise ValueError(
                "style_prompt 不适用于 /v2/synthesize；该接口仅用于参考音频克隆。"
            )
        return value
