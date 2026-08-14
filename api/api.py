import fcntl
import base64
import hashlib
import io
import json
import os
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
import wave
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import uvicorn
from typing import Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from gpu_runtime import cuda_status
from voxcpm2_voice_design import (
    VOXCPM2_MODEL_DIR,
    VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT,
    VoxCpm2VoiceDesignRequest,
    run_voxcpm2_voice_design,
    voice_design_is_ready,
)
from step_audio_editx import (
    STEP_AUDIO_EDITX_CODE_PATH,
    STEP_AUDIO_EDITX_CONDA_ENV,
    STEP_AUDIO_EDITX_RUNTIME,
    STEP_AUDIO_EDITX_UV_BASE_URL,
    STEP_AUDIO_EDITX_MODEL_DIR,
    STEP_AUDIO_EDITX_REQUEST_TIMEOUT,
    STEP_AUDIO_TOKENIZER_PATH,
    StepAudioEditXEditRequest,
    manager as step_audio_editx_manager,
    step_audio_editx_is_ready,
)

# ==========================================
# 0. 系统配置
# ==========================================
API_DIR = os.path.dirname(os.path.abspath(__file__))

def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", os.path.join(API_DIR, "prompts")))
TTS_OUTPUT_DIR = expand_path(
    os.getenv("TTS_OUTPUT_DIR", os.path.join(API_DIR, "tempAudio"))
)
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", os.path.join(API_DIR, ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2.5-tts-voicedesign")
MIMO_AUTH_HEADER = os.getenv("MIMO_AUTH_HEADER", "api-key")
MIMO_TIMEOUT = float(os.getenv("MIMO_TIMEOUT", "300"))
MIMO_MAX_CHARS_PER_CHUNK = int(os.getenv("MIMO_MAX_CHARS_PER_CHUNK", "300"))
MIMO_PAUSE_MS = int(os.getenv("MIMO_PAUSE_MS", "250"))
MIMO_OPTIMIZE_TEXT_PREVIEW = env_bool("MIMO_OPTIMIZE_TEXT_PREVIEW", False)
MIMO_MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("MIMO_MIN_REQUEST_INTERVAL_SECONDS", "0"))
MIMO_MAX_RETRIES = int(os.getenv("MIMO_MAX_RETRIES", "3"))
MIMO_RETRY_BASE_SECONDS = float(os.getenv("MIMO_RETRY_BASE_SECONDS", "5"))
MIMO_RETRY_MAX_SECONDS = float(os.getenv("MIMO_RETRY_MAX_SECONDS", "60"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8300"))

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(RUNTIME_CACHE_DIR, "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(RUNTIME_CACHE_DIR, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RUNTIME_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RUNTIME_CACHE_DIR, "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
gpu_lock_dir = os.path.dirname(GPU_LOCK_FILE)
if gpu_lock_dir:
    os.makedirs(gpu_lock_dir, exist_ok=True)

app = FastAPI(title="Super Unitale Smart API")

class ForceCORS(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return Response(status_code=200, headers={
                "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Credentials": "false",
            })
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

app.add_middleware(ForceCORS)

def hash_filename(filename: str) -> str:
    ext = os.path.splitext(filename)[1] or ".wav"
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{h}{ext}"


@contextmanager
def gpu_runtime_lock(label: str):
    with open(GPU_LOCK_FILE, "a+", encoding="utf-8") as lock_file:
        print(f"[GPU 锁] 等待进入: {label}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(f"[GPU 锁] 已进入: {label}")
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(f"[GPU 锁] 已退出: {label}")


def wait_after_cuda_release(label: str = "") -> None:
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


# ==========================================
# 1. 接口定义
# ==========================================
class MimoDesignRequest(BaseModel):
    voice_description: str
    text: str = "这是生成的参考音频预览。"
    save_as: Optional[str] = "designed_voice.wav"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    auth_header: Optional[str] = None
    timeout: Optional[float] = None
    max_chars_per_chunk: Optional[int] = None
    pause_ms: Optional[int] = None
    optimize_text_preview: Optional[bool] = None
    min_request_interval_seconds: Optional[float] = None
    max_retries: Optional[int] = None
    retry_base_seconds: Optional[float] = None
    retry_max_seconds: Optional[float] = None


class MiMoHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str, retry_after: Optional[float] = None):
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


def resolve_mimo_api_key(api_key: Optional[str]) -> str:
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


def mimo_parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return None


def mimo_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        retry_after = mimo_parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
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
                retryable = (
                    isinstance(exc, MiMoTransportError)
                    or mimo_is_retryable_http_error(exc)
                )
                if not retryable or attempt > max_retries:
                    raise
                delay = mimo_retry_delay_seconds(exc, attempt, retry_base_seconds, retry_max_seconds)
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
    timeout = float(request_data.get("timeout") if request_data.get("timeout") is not None else MIMO_TIMEOUT)
    max_chars_per_chunk = int(
        request_data.get("max_chars_per_chunk")
        if request_data.get("max_chars_per_chunk") is not None
        else MIMO_MAX_CHARS_PER_CHUNK
    )
    pause_ms = int(request_data.get("pause_ms") if request_data.get("pause_ms") is not None else MIMO_PAUSE_MS)
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
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "voxcpm2_model_dir": VOXCPM2_MODEL_DIR,
            "voxcpm2_voice_design_worker_script": VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT,
            "step_audio_editx_model_dir": str(STEP_AUDIO_EDITX_MODEL_DIR),
            "step_audio_tokenizer_path": str(STEP_AUDIO_TOKENIZER_PATH),
            "step_audio_editx_code_path": str(STEP_AUDIO_EDITX_CODE_PATH),
            "prompts_dir": PROMPTS_DIR,
            "tts_output_dir": TTS_OUTPUT_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "mimo_base_url": MIMO_BASE_URL,
        },
        "available": {
            "voxcpm2_model_dir": os.path.isdir(VOXCPM2_MODEL_DIR),
            "voxcpm2_voice_design": voice_design_is_ready(),
            "step_audio_editx": step_audio_editx_is_ready(),
            "mimo_api_key": bool(os.getenv("MIMO_API_KEY")),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "last_errors": {
            "step_audio_editx": step_audio_editx_manager.last_error,
        },
        "offline": {
            "local_files_only": LOCAL_FILES_ONLY,
            "hf_hub_offline": os.getenv("HF_HUB_OFFLINE"),
            "transformers_offline": os.getenv("TRANSFORMERS_OFFLINE"),
        },
        "runtime": {
            "voice_design_providers": ["mimo", "voxcpm2"],
            "step_audio_editx_runtime": STEP_AUDIO_EDITX_RUNTIME,
            "step_audio_editx_worker_env": STEP_AUDIO_EDITX_CONDA_ENV,
            "step_audio_editx_uv_base_url": STEP_AUDIO_EDITX_UV_BASE_URL,
            "step_audio_editx_request_timeout": STEP_AUDIO_EDITX_REQUEST_TIMEOUT,
            "api_cuda_context": "disabled; health uses nvidia-smi",
            "gpu_scheduling": "all local services share one exclusive file lock",
            "mimo_model": MIMO_MODEL,
            "mimo_auth_header": MIMO_AUTH_HEADER,
            "mimo_timeout": MIMO_TIMEOUT,
            "mimo_max_chars_per_chunk": MIMO_MAX_CHARS_PER_CHUNK,
            "mimo_pause_ms": MIMO_PAUSE_MS,
            "mimo_optimize_text_preview": MIMO_OPTIMIZE_TEXT_PREVIEW,
            "mimo_min_request_interval_seconds": MIMO_MIN_REQUEST_INTERVAL_SECONDS,
            "mimo_max_retries": MIMO_MAX_RETRIES,
        },
    }


@app.get("/v1/voice-design/providers")
async def voice_design_providers():
    return {
        "code": 200,
        "providers": [
            {
                "id": "voxcpm2",
                "name": "VoxCPM2 VoiceDesign",
                "route": "/v1/voxcpm2/design",
                "type": "local_model",
                "ready": voice_design_is_ready(),
            },
            {
                "id": "mimo",
                "name": "MiMo TTS VoiceDesign",
                "route": "/v1/mimo/design",
                "type": "cloud_api",
                "ready": bool(os.getenv("MIMO_API_KEY")),
            },
        ],
    }


@app.post("/v1/upload_audio")
async def upload_audio(audio: UploadFile = File(...), full_path: str = Form(...)):
    content = await audio.read()
    save_path = os.path.join(PROMPTS_DIR, hash_filename(full_path))
    with open(save_path, "wb") as f: f.write(content)
    return {"code": 200, "msg": "上传成功", "filename": full_path}


@app.post("/v1/step-audio-editx/edit")
async def step_audio_editx_edit(request: StepAudioEditXEditRequest):
    """兼容 WebUI 的 Step-Audio-EditX 编辑入口。"""
    prompt_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.prompt_audio))
    try:
        if STEP_AUDIO_EDITX_RUNTIME == "uv":
            # 迁移期间保留此 8300 入口；GPU 锁由独立 8316 服务持有，避免跨进程重复加锁死锁。
            with step_audio_editx_manager.lock:
                audio_bytes = step_audio_editx_manager.run_uv_service(
                    request, Path(prompt_audio_path)
                )
        else:
            # 旧 Conda worker 继续作为显式回退，待迁移确认后再删除。
            with gpu_runtime_lock("step-audio-editx/edit"):
                with step_audio_editx_manager.lock:
                    payload = step_audio_editx_manager.build_worker_payload(
                        request,
                        Path(prompt_audio_path),
                        local_files_only=LOCAL_FILES_ONLY,
                    )
                    audio_bytes = step_audio_editx_manager.run_worker(payload)
                wait_after_cuda_release("after Step-Audio-EditX worker")
        return Response(content=audio_bytes, media_type="audio/wav")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=int(getattr(exc, "status_code", 500)), detail=str(exc)
        ) from exc

@app.get("/v1/check/audio")
async def check_audio_exists(file_name: str):
    exists = os.path.isfile(os.path.join(PROMPTS_DIR, hash_filename(file_name)))
    return {"code": 200 if exists else 404, "exists": exists}

@app.post("/v1/voxcpm2/design")
async def voxcpm2_design(request: VoxCpm2VoiceDesignRequest):
    """独立的 VoxCPM2 音色设计接口，不与 Qwen 音色设计路由混用。"""
    with gpu_runtime_lock("voxcpm2/design"):
        try:
            audio_bytes = run_voxcpm2_voice_design(request)
            return Response(content=audio_bytes, media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            wait_after_cuda_release("after VoxCPM2 voice design")


@app.post("/v1/mimo/design")
async def mimo_design(request: MimoDesignRequest):
    # MiMo is a cloud API. It neither uses CUDA nor needs the local-model lock;
    # the dedicated request lock above serializes MiMo calls and retries instead.
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
    return Response(content=audio_bytes, media_type="audio/wav")

if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端服务 EditX + MiMo/VoxCPM2")
    print("==================================================")
    print(f"[配置] VoxCPM2 VoiceDesign 模型目录: {VOXCPM2_MODEL_DIR}")
    print(f"[配置] Step-Audio-EditX worker env: {STEP_AUDIO_EDITX_CONDA_ENV}")
    print(f"[配置] Step-Audio-EditX 模型目录: {STEP_AUDIO_EDITX_MODEL_DIR}")
    print(f"[配置] Step-Audio-Tokenizer 目录: {STEP_AUDIO_TOKENIZER_PATH}")
    print(f"[配置] Step-Audio-EditX 源码目录: {STEP_AUDIO_EDITX_CODE_PATH}")
    print(f"[配置] MiMo base URL: {MIMO_BASE_URL}")
    print(f"[配置] MiMo 模型: {MIMO_MODEL}")
    print(f"[配置] MiMo API key: {'已配置' if os.getenv('MIMO_API_KEY') else '未配置'}")
    print(f"[配置] prompts 目录: {PROMPTS_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
