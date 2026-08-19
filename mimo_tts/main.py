#!/usr/bin/env python3
"""Standalone MiMo TTS VoiceDesign HTTP service.

MiMo is a cloud provider, so this service contains only request orchestration,
retry/chunk handling, WAV concatenation, and local timbre caching.  It does
not import Torch or depend on the repository control-plane API.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from audio_output import persist_audio_bytes

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage")))
TIMBRE_STORAGE_DIR = expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(Path(STORAGE_DIR) / "timbre")))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", str(Path(STORAGE_DIR) / ".cache/runtime"))
)
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2.5-tts-voicedesign")
MIMO_AUTH_HEADER = os.getenv("MIMO_AUTH_HEADER", "api-key")
MIMO_TIMEOUT = float(os.getenv("MIMO_TIMEOUT", "300"))
MIMO_MAX_CHARS_PER_CHUNK = int(os.getenv("MIMO_MAX_CHARS_PER_CHUNK", "300"))
MIMO_PAUSE_MS = int(os.getenv("MIMO_PAUSE_MS", "250"))
MIMO_OPTIMIZE_TEXT_PREVIEW = os.getenv("MIMO_OPTIMIZE_TEXT_PREVIEW", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MIMO_MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("MIMO_MIN_REQUEST_INTERVAL_SECONDS", "0"))
MIMO_MAX_RETRIES = int(os.getenv("MIMO_MAX_RETRIES", "3"))
MIMO_RETRY_BASE_SECONDS = float(os.getenv("MIMO_RETRY_BASE_SECONDS", "5"))
MIMO_RETRY_MAX_SECONDS = float(os.getenv("MIMO_RETRY_MAX_SECONDS", "60"))
API_HOST = os.getenv("MIMO_TTS_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("MIMO_TTS_PORT", os.getenv("PORT", "8303")))

os.makedirs(TIMBRE_STORAGE_DIR, exist_ok=True)
os.makedirs(RUNTIME_CACHE_DIR, exist_ok=True)

app = FastAPI(title="Unitale MiMo TTS VoiceDesign API")


class ForceCORS(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Credentials": "false",
                },
            )
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response


app.add_middleware(ForceCORS)


class MimoDesignRequest(BaseModel):
    voice_description: str
    text: str = "这是生成的参考音频预览。"
    save_as: str | None = "designed_voice.wav"
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    auth_header: str | None = None
    timeout: float | None = None
    max_chars_per_chunk: int | None = None
    pause_ms: int | None = None
    optimize_text_preview: bool | None = None
    min_request_interval_seconds: float | None = None
    max_retries: int | None = None
    retry_base_seconds: float | None = None
    retry_max_seconds: float | None = None


class MiMoHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str, retry_after: float | None = None):
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        super().__init__(f"MiMo HTTP {status_code}: {body}")


class MiMoTransportError(RuntimeError):
    """MiMo API cannot be reached from this backend process."""


MIMO_REQUEST_LOCK = threading.Lock()


def split_long_voice_design_text(text: str, max_chars: int) -> list[str]:
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


def split_voice_design_text(text: str, max_chars: int) -> list[str]:
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
            chunks.extend(split_long_voice_design_text(sentence, max_chars))
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


def resolve_mimo_api_key(api_key: str | None) -> str:
    resolved = api_key or os.getenv("MIMO_API_KEY")
    if not resolved:
        raise RuntimeError("MiMo API key 缺失。请设置 MIMO_API_KEY，或在请求中传入 api_key。")
    return resolved


def mimo_chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def mimo_request_headers(api_key: str, auth_header: str) -> dict[str, str]:
    normalized = auth_header.strip().lower()
    if normalized not in {"api-key", "bearer", "both"}:
        raise ValueError(f"不支持的 MiMo auth_header: {auth_header}")

    headers = {"Content-Type": "application/json"}
    if normalized in {"api-key", "both"}:
        headers["api-key"] = api_key
    if normalized in {"bearer", "both"}:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def mimo_build_messages(voice_instruction: str, chunk: str) -> list[dict[str, str]]:
    voice_instruction = voice_instruction.strip()
    if not voice_instruction:
        return [{"role": "assistant", "content": chunk}]
    return [
        {"role": "user", "content": voice_instruction},
        {"role": "assistant", "content": chunk},
    ]


def mimo_parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return None


def mimo_post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        retry_after = mimo_parse_retry_after(
            exc.headers.get("Retry-After") if exc.headers else None
        )
        raise MiMoHTTPError(exc.code, error_body, retry_after) from exc
    except urllib.error.URLError as exc:
        raise MiMoTransportError(f"MiMo 网络请求失败: {exc.reason}") from exc
    except TimeoutError as exc:
        raise MiMoTransportError(f"MiMo 网络请求超时: {exc}") from exc

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise MiMoTransportError(f"MiMo 返回了非 JSON 响应: {response_body[:500]}") from exc


def mimo_is_retryable_http_error(exc: MiMoHTTPError) -> bool:
    return exc.status_code == 429 or 500 <= exc.status_code <= 599


def mimo_retry_delay_seconds(
    exc: MiMoHTTPError | MiMoTransportError,
    attempt: int,
    base: float,
    maximum: float,
) -> float:
    if isinstance(exc, MiMoHTTPError) and exc.retry_after is not None:
        return min(maximum, exc.retry_after)
    return min(maximum, max(0.0, base) * (2 ** max(0, attempt - 1)))


def mimo_post_json_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    min_request_interval_seconds: float,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    chunk_label: str,
) -> dict[str, Any]:
    with MIMO_REQUEST_LOCK:
        for attempt in range(1, max_retries + 2):
            try:
                response = mimo_post_json(url, payload, headers, timeout)
            except (MiMoHTTPError, MiMoTransportError) as exc:
                retryable = isinstance(exc, MiMoTransportError) or mimo_is_retryable_http_error(exc)
                if not retryable or attempt > max_retries:
                    raise
                delay = mimo_retry_delay_seconds(
                    exc, attempt, retry_base_seconds, retry_max_seconds
                )
                error_label = (
                    f"MiMo HTTP {exc.status_code}"
                    if isinstance(exc, MiMoHTTPError)
                    else "MiMo 网络错误"
                )
                print(
                    f"{error_label}，{delay:.1f}s 后重试 {chunk_label}，"
                    f"第 {attempt}/{max_retries} 次"
                )
                time.sleep(delay)
                continue

            if min_request_interval_seconds > 0:
                time.sleep(min_request_interval_seconds)
            return response

    raise RuntimeError("MiMo request did not return a response")


def mimo_extract_audio_bytes(response: dict[str, Any]) -> bytes:
    try:
        encoded = response["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"MiMo response 缺少 choices[0].message.audio.data: {response}") from exc
    return base64.b64decode(encoded)


def read_wav_params(audio_bytes: bytes) -> wave._wave_params:
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        params = reader.getparams()
    if params.comptype != "NONE":
        raise RuntimeError(f"不支持压缩 wav 拼接: {params.comptype}")
    return params


def join_wav_bytes(chunks: list[bytes], pause_ms: int) -> bytes:
    if not chunks:
        raise RuntimeError("MiMo 未返回音频片段。")

    first_params = read_wav_params(chunks[0])
    sample_rate = int(first_params.framerate)
    frame_size = first_params.nchannels * first_params.sampwidth
    pause_frames = max(0, int(sample_rate * pause_ms / 1000))
    pause = b"\x00" * pause_frames * frame_size

    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams(first_params)
        for index, chunk in enumerate(chunks):
            with wave.open(io.BytesIO(chunk), "rb") as reader:
                params = reader.getparams()
                if params[:3] != first_params[:3] or params[4:] != first_params[4:]:
                    raise RuntimeError(
                        "MiMo wav 分块参数不一致，无法拼接："
                        f"chunk 1={first_params}, chunk {index + 1}={params}"
                    )
                writer.writeframes(reader.readframes(reader.getnframes()))
            if index < len(chunks) - 1 and pause:
                writer.writeframes(pause)

    return output.getvalue()


def run_mimo_voice_design(request_data: dict[str, Any]) -> bytes:
    text = str(request_data.get("text") or "").strip()
    if not text:
        raise RuntimeError("text 不能为空。")

    voice_instruction = str(request_data.get("voice_description") or "").strip()
    if not voice_instruction:
        raise RuntimeError("voice_description 不能为空。")

    model = str(request_data.get("model") or MIMO_MODEL).strip()
    if model != "mimo-v2.5-tts-voicedesign":
        raise RuntimeError(f"MiMo 音色设计仅支持 mimo-v2.5-tts-voicedesign，当前为: {model}")

    api_key = resolve_mimo_api_key(request_data.get("api_key"))
    base_url = str(request_data.get("base_url") or MIMO_BASE_URL)
    auth_header = str(request_data.get("auth_header") or MIMO_AUTH_HEADER)
    timeout = float(
        request_data.get("timeout") if request_data.get("timeout") is not None else MIMO_TIMEOUT
    )
    max_chars_per_chunk = int(
        request_data.get("max_chars_per_chunk")
        if request_data.get("max_chars_per_chunk") is not None
        else MIMO_MAX_CHARS_PER_CHUNK
    )
    pause_ms = int(
        request_data.get("pause_ms") if request_data.get("pause_ms") is not None else MIMO_PAUSE_MS
    )
    optimize_text_preview = (
        bool(request_data["optimize_text_preview"])
        if request_data.get("optimize_text_preview") is not None
        else MIMO_OPTIMIZE_TEXT_PREVIEW
    )
    min_request_interval_seconds = float(
        request_data.get("min_request_interval_seconds")
        if request_data.get("min_request_interval_seconds") is not None
        else MIMO_MIN_REQUEST_INTERVAL_SECONDS
    )
    max_retries = max(
        0,
        int(
            request_data.get("max_retries")
            if request_data.get("max_retries") is not None
            else MIMO_MAX_RETRIES
        ),
    )
    retry_base_seconds = float(
        request_data.get("retry_base_seconds")
        if request_data.get("retry_base_seconds") is not None
        else MIMO_RETRY_BASE_SECONDS
    )
    retry_max_seconds = float(
        request_data.get("retry_max_seconds")
        if request_data.get("retry_max_seconds") is not None
        else MIMO_RETRY_MAX_SECONDS
    )

    chunks = split_voice_design_text(text, max_chars_per_chunk)
    audio_payload: dict[str, Any] = {"format": "wav"}
    if optimize_text_preview:
        audio_payload["optimize_text_preview"] = True

    print(f"[MiMo] model={model}, base_url={base_url}, chunks={len(chunks)}")
    url = mimo_chat_completions_url(base_url)
    headers = mimo_request_headers(api_key, auth_header)
    audio_chunks: list[bytes] = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"[MiMo] 合成 chunk {index}/{len(chunks)}: {len(chunk)} 字")
        payload = {
            "model": model,
            "messages": mimo_build_messages(voice_instruction, chunk),
            "audio": audio_payload,
        }
        response = mimo_post_json_with_retry(
            url=url,
            payload=payload,
            headers=headers,
            timeout=timeout,
            min_request_interval_seconds=min_request_interval_seconds,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            chunk_label=f"MiMo chunk {index}/{len(chunks)}",
        )
        audio_chunks.append(mimo_extract_audio_bytes(response))

    return join_wav_bytes(audio_chunks, pause_ms)


@app.get("/v1/health")
async def health():
    return {
        "code": 200,
        "paths": {
            "storage_dir": STORAGE_DIR,
            "timbre_storage_dir": TIMBRE_STORAGE_DIR,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
        },
        "available": {
            "mimo_api_key": bool(os.getenv("MIMO_API_KEY")),
        },
        "runtime": {
            "provider": "mimo",
            "model": MIMO_MODEL,
            "base_url": MIMO_BASE_URL,
            "auth_header": MIMO_AUTH_HEADER,
            "timeout": MIMO_TIMEOUT,
            "max_chars_per_chunk": MIMO_MAX_CHARS_PER_CHUNK,
            "pause_ms": MIMO_PAUSE_MS,
            "optimize_text_preview": MIMO_OPTIMIZE_TEXT_PREVIEW,
            "max_retries": MIMO_MAX_RETRIES,
        },
    }


@app.get("/v1/voice-design/providers")
async def voice_design_providers():
    return {
        "code": 200,
        "providers": [
            {
                "id": "mimo",
                "name": "MiMo TTS VoiceDesign",
                "route": "/v1/mimo/timbre",
                "type": "cloud_api",
                "ready": bool(os.getenv("MIMO_API_KEY")),
            }
        ],
    }


@app.post("/v1/mimo/timbre")
async def mimo_design(request: MimoDesignRequest):
    try:
        audio_bytes = run_mimo_voice_design(request.model_dump())
    except MiMoHTTPError as exc:
        if exc.status_code == 429:
            status_code = 429
        elif exc.status_code >= 500:
            status_code = 503
        else:
            status_code = 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except MiMoTransportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{exc}。后端无法连接 MiMo API，请检查此机器到 "
                "https://api.xiaomimimo.com 的 DNS、HTTPS 出网或 HTTPS_PROXY 配置。"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    saved_output_path = persist_audio_bytes(
        audio_bytes,
        "mimo_voicedesign",
        TIMBRE_STORAGE_DIR,
    )
    print(f"[MiMo] 已保存音色音频: {saved_output_path}")
    return Response(content=audio_bytes, media_type="audio/wav")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale MiMo TTS VoiceDesign (uv)")
    print("==================================================")
    print(f"[配置] MiMo base URL: {MIMO_BASE_URL}")
    print(f"[配置] MiMo 模型: {MIMO_MODEL}")
    print(f"[配置] MiMo API key: {'已配置' if os.getenv('MIMO_API_KEY') else '未配置'}")
    print(f"[配置] 音色缓存目录: {TIMBRE_STORAGE_DIR}")
    print(f"[配置] host={API_HOST}, port={API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
