"""HTTP wrapper for one-shot dots.tts-soar voice cloning."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Field
from starlette.middleware.base import BaseHTTPMiddleware

from runtime import UvWorkerConfig, cuda_status, persist_audio_bytes, run_uv_worker
from synthesis_request import CloneSynthesisRequest

LOGGER = logging.getLogger(__name__)

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SERVICE_DIR)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def normalize_synthesis_text(text: str) -> str:
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise HTTPException(status_code=400, detail="text 不能为空。")
    return normalized


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", os.path.join(PROJECT_DIR, "storage")))
CLONE_STORAGE_DIR = expand_path(os.getenv("CLONE_STORAGE_DIR", os.path.join(STORAGE_DIR, "clone")))
TIMBRE_STORAGE_DIR = expand_path(
    os.getenv("TIMBRE_STORAGE_DIR", os.path.join(STORAGE_DIR, "timbre"))
)
TIMBRE_REFERENCE_DIR = os.path.join(TIMBRE_STORAGE_DIR, ".references")
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", CLONE_STORAGE_DIR))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", os.path.join(STORAGE_DIR, ".cache/runtime"))
)
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8324"))

DOTS_TTS_SOAR_MODEL_DIR = expand_path(
    os.getenv("DOTS_TTS_SOAR_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "rednote-hilab/dots.tts-soar"))
)
# ============================================================================
# dots.tts-soar 克隆调试默认值
#
# 调试参考音频克隆效果时，优先直接修改下面带 _DEFAULT 后缀的值，然后
# 重启服务即可生效。环境变量仍可覆盖默认值，方便部署时统一配置。
# ============================================================================
# 推理精度：bfloat16 通常更省显存；遇到硬件不兼容时可改为 float16 或 float32。
DOTS_TTS_SOAR_PRECISION_DEFAULT = "bfloat16"
# 语言标识：官方示例使用 chinese；多语言场景可按模型支持范围调整。
DOTS_TTS_SOAR_LANGUAGE_DEFAULT = "chinese"
# ODE 求解器：euler 速度较快；具体可选值取决于当前 dots.tts-soar 版本。
DOTS_TTS_SOAR_ODE_METHOD_DEFAULT = "euler"
# 推理步数：步数越高通常越稳定，但会增加生成耗时。
DOTS_TTS_SOAR_NUM_STEPS_DEFAULT = 10
# 文本引导强度：提高可强化内容约束，过高可能影响自然度。
DOTS_TTS_SOAR_GUIDANCE_SCALE_DEFAULT = 1.2
# 说话人条件强度：提高可强化参考音色，过高可能带来音色或韵律失真。
DOTS_TTS_SOAR_SPEAKER_SCALE_DEFAULT = 1.5
# 单段最大生成长度：用于限制模型一次生成的 token/帧长度，具体含义由官方 runtime 决定。
DOTS_TTS_SOAR_MAX_GENERATE_LENGTH_DEFAULT = 500
# 文本分片字符数：0 表示不分片；分片可以降低显存压力，但会插入停顿。
DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK_DEFAULT = 120
# 分片之间的停顿时长，单位为毫秒；仅在发生分片时生效。
DOTS_TTS_SOAR_PAUSE_MS_DEFAULT = 250
# 随机种子：固定整数便于复现；如需随机行为请按当前官方 runtime 支持范围调整。
DOTS_TTS_SOAR_SEED_DEFAULT = 42
# 是否让官方 runtime 规范化输入文本；保持 False 可最大程度保留原始文本。
DOTS_TTS_SOAR_NORMALIZE_TEXT_DEFAULT = False
# 是否开启推理性能分析；仅调试性能时开启，避免额外开销。
DOTS_TTS_SOAR_PROFILE_INFERENCE_DEFAULT = False
# 单次请求超时时间，单位为秒；包含 worker 启动和完整合成。
DOTS_TTS_SOAR_REQUEST_TIMEOUT_DEFAULT = 900.0

DOTS_TTS_SOAR_PRECISION = os.getenv("DOTS_TTS_SOAR_PRECISION", DOTS_TTS_SOAR_PRECISION_DEFAULT)
DOTS_TTS_SOAR_LANGUAGE = normalize_optional_text(
    os.getenv("DOTS_TTS_SOAR_LANGUAGE", DOTS_TTS_SOAR_LANGUAGE_DEFAULT)
)
DOTS_TTS_SOAR_ODE_METHOD = os.getenv("DOTS_TTS_SOAR_ODE_METHOD", DOTS_TTS_SOAR_ODE_METHOD_DEFAULT)
DOTS_TTS_SOAR_NUM_STEPS = int(
    os.getenv("DOTS_TTS_SOAR_NUM_STEPS", str(DOTS_TTS_SOAR_NUM_STEPS_DEFAULT))
)
DOTS_TTS_SOAR_GUIDANCE_SCALE = float(
    os.getenv("DOTS_TTS_SOAR_GUIDANCE_SCALE", str(DOTS_TTS_SOAR_GUIDANCE_SCALE_DEFAULT))
)
DOTS_TTS_SOAR_SPEAKER_SCALE = float(
    os.getenv("DOTS_TTS_SOAR_SPEAKER_SCALE", str(DOTS_TTS_SOAR_SPEAKER_SCALE_DEFAULT))
)
DOTS_TTS_SOAR_MAX_GENERATE_LENGTH = int(
    os.getenv(
        "DOTS_TTS_SOAR_MAX_GENERATE_LENGTH",
        str(DOTS_TTS_SOAR_MAX_GENERATE_LENGTH_DEFAULT),
    )
)
DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK = int(
    os.getenv(
        "DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK",
        str(DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK_DEFAULT),
    )
)
DOTS_TTS_SOAR_PAUSE_MS = int(
    os.getenv("DOTS_TTS_SOAR_PAUSE_MS", str(DOTS_TTS_SOAR_PAUSE_MS_DEFAULT))
)
DOTS_TTS_SOAR_SEED = int(os.getenv("DOTS_TTS_SOAR_SEED", str(DOTS_TTS_SOAR_SEED_DEFAULT)))
DOTS_TTS_SOAR_NORMALIZE_TEXT = env_bool(
    "DOTS_TTS_SOAR_NORMALIZE_TEXT", DOTS_TTS_SOAR_NORMALIZE_TEXT_DEFAULT
)
DOTS_TTS_SOAR_PROFILE_INFERENCE = env_bool(
    "DOTS_TTS_SOAR_PROFILE_INFERENCE", DOTS_TTS_SOAR_PROFILE_INFERENCE_DEFAULT
)
DOTS_TTS_SOAR_REQUEST_TIMEOUT = float(
    os.getenv("DOTS_TTS_SOAR_REQUEST_TIMEOUT", str(DOTS_TTS_SOAR_REQUEST_TIMEOUT_DEFAULT))
)
DOTS_TTS_SOAR_WORKER_SCRIPT = os.path.join(SERVICE_DIR, "worker.py")
DOTS_TTS_SOAR_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "dots_tts_soar_worker")
DOTS_TTS_SOAR_OUTPUT_DIR = expand_path(
    os.getenv(
        "DOTS_TTS_SOAR_OUTPUT_DIR",
        CLONE_STORAGE_DIR,
    )
)

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(RUNTIME_CACHE_DIR, "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(RUNTIME_CACHE_DIR, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RUNTIME_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RUNTIME_CACHE_DIR, "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(TIMBRE_STORAGE_DIR, exist_ok=True)
os.makedirs(TIMBRE_REFERENCE_DIR, exist_ok=True)
os.makedirs(DOTS_TTS_SOAR_WORKER_TMP_DIR, exist_ok=True)
os.makedirs(DOTS_TTS_SOAR_OUTPUT_DIR, exist_ok=True)
for cache_key in ("HF_MODULES_CACHE", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
    os.makedirs(os.environ[cache_key], exist_ok=True)
lock_dir = os.path.dirname(GPU_LOCK_FILE)
if lock_dir:
    os.makedirs(lock_dir, exist_ok=True)

app = FastAPI(title="Unitale dots.tts-soar Voice Clone API")


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


def hash_filename(filename: str) -> str:
    ext = os.path.splitext(filename)[1] or ".wav"
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{digest}{ext}"


def clone_prompt_audio_path(filename: str) -> str:
    return os.path.join(PROMPTS_DIR, hash_filename(filename))


def timbre_reference_map_path(filename: str) -> str:
    return os.path.join(TIMBRE_REFERENCE_DIR, f"{hash_filename(filename)}.path")


def prompt_audio_path(filename: str) -> str:
    """解析克隆上传，或解析只保存在音色目录中的设计音频。"""
    clone_path = clone_prompt_audio_path(filename)
    if os.path.isfile(clone_path):
        return clone_path

    reference_path = timbre_reference_map_path(filename)
    if os.path.isfile(reference_path):
        with open(reference_path, encoding="utf-8") as reference_file:
            timbre_path = reference_file.read().strip()
        if timbre_path and os.path.isfile(timbre_path):
            return timbre_path
    return clone_path


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_matching_timbre_audio(content: bytes) -> str | None:
    """识别已生成的音色，避免把同一份 WAV 再复制到克隆目录。"""
    content_digest = hashlib.sha256(content).hexdigest()
    with os.scandir(TIMBRE_STORAGE_DIR) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.lower().endswith(".wav"):
                continue
            if file_sha256(entry.path) == content_digest:
                return entry.path
    return None


def prompt_text_sidecar_path(filename: str) -> str:
    clone_sidecar_path = os.path.join(PROMPTS_DIR, f"{hash_filename(filename)}.prompt.txt")
    if os.path.isfile(clone_prompt_audio_path(filename)):
        return clone_sidecar_path
    if os.path.isfile(timbre_reference_map_path(filename)):
        return os.path.join(TIMBRE_REFERENCE_DIR, f"{hash_filename(filename)}.prompt.txt")
    return clone_sidecar_path


def load_prompt_text_sidecar(filename: str) -> str | None:
    path = prompt_text_sidecar_path(filename)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as file:
        return normalize_optional_text(file.read())


def save_prompt_text_sidecar(filename: str, prompt_text: str | None) -> None:
    path = prompt_text_sidecar_path(filename)
    normalized = normalize_optional_text(prompt_text)
    if normalized is None:
        if os.path.exists(path):
            os.remove(path)
        return
    with open(path, "w", encoding="utf-8") as file:
        file.write(normalized)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_local_request(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="仅允许本机访问内部接口")


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


def wait_after_cuda_release(label: str) -> None:
    if CUDA_RELEASE_DELAY > 0:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
        import time

        time.sleep(CUDA_RELEASE_DELAY)


class DotsTtsSoarSynthesizeRequest(CloneSynthesisRequest):
    """dots.tts-soar official clone parameters."""

    text: str
    audio_path: str
    prompt_text: str | None = None
    language: str | None = None
    template_name: Literal["tts", "instruction_tts", "text_to_audio", "tts_interleave"] | None = (
        None
    )
    precision: str | None = None
    seed: int | None = None
    ode_method: str | None = None
    num_steps: int | None = Field(default=None, ge=1)
    guidance_scale: float | None = Field(default=None, ge=0)
    speaker_scale: float | None = Field(default=None, ge=0)
    max_generate_length: int | None = Field(default=None, ge=1)
    max_chars_per_chunk: int | None = Field(default=None, ge=0)
    pause_ms: int | None = Field(default=None, ge=0)
    normalize_text: bool | None = None
    profile_inference: bool | None = None


DOTS_TTS_SOAR_WORKER = UvWorkerConfig(
    python_executable=sys.executable,
    worker_script=DOTS_TTS_SOAR_WORKER_SCRIPT,
    model_dir=DOTS_TTS_SOAR_MODEL_DIR,
    temp_dir=DOTS_TTS_SOAR_WORKER_TMP_DIR,
    timeout=DOTS_TTS_SOAR_REQUEST_TIMEOUT,
    label="dots.tts-soar",
    file_prefix="dots_tts_soar",
)


class DotsTtsSoarWorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: DotsTtsSoarSynthesizeRequest) -> dict:
        ref_audio_path = prompt_audio_path(request.audio_path)
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        prompt_text = normalize_optional_text(request.prompt_text)
        if prompt_text is None:
            prompt_text = load_prompt_text_sidecar(request.audio_path)

        return {
            "operation": "clone",
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "model_path": DOTS_TTS_SOAR_MODEL_DIR,
            "language": (
                normalize_optional_text(request.language)
                if request.language is not None
                else DOTS_TTS_SOAR_LANGUAGE
            ),
            "template_name": request.template_name,
            "precision": request.precision or DOTS_TTS_SOAR_PRECISION,
            "seed": request.seed if request.seed is not None else DOTS_TTS_SOAR_SEED,
            "ode_method": request.ode_method or DOTS_TTS_SOAR_ODE_METHOD,
            "num_steps": request.num_steps
            if request.num_steps is not None
            else DOTS_TTS_SOAR_NUM_STEPS,
            "guidance_scale": (
                request.guidance_scale
                if request.guidance_scale is not None
                else DOTS_TTS_SOAR_GUIDANCE_SCALE
            ),
            "speaker_scale": (
                request.speaker_scale
                if request.speaker_scale is not None
                else DOTS_TTS_SOAR_SPEAKER_SCALE
            ),
            "max_generate_length": (
                request.max_generate_length
                if request.max_generate_length is not None
                else DOTS_TTS_SOAR_MAX_GENERATE_LENGTH
            ),
            "max_chars_per_chunk": (
                request.max_chars_per_chunk
                if request.max_chars_per_chunk is not None
                else DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK
            ),
            "pause_ms": request.pause_ms
            if request.pause_ms is not None
            else DOTS_TTS_SOAR_PAUSE_MS,
            "normalize_text": (
                request.normalize_text
                if request.normalize_text is not None
                else DOTS_TTS_SOAR_NORMALIZE_TEXT
            ),
            "profile_inference": (
                request.profile_inference
                if request.profile_inference is not None
                else DOTS_TTS_SOAR_PROFILE_INFERENCE
            ),
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def run_worker(self, payload: dict) -> bytes:
        try:
            audio = run_uv_worker(payload, DOTS_TTS_SOAR_WORKER)
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise


manager = DotsTtsSoarWorkerManager()


@app.get("/v1/health")
async def health():
    cuda = cuda_status()
    required_files = {
        name: os.path.isfile(os.path.join(DOTS_TTS_SOAR_MODEL_DIR, name))
        for name in (
            "config.json",
            "model.safetensors",
            "speaker_encoder.safetensors",
            "vocoder.safetensors",
        )
    }
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "model_dir": DOTS_TTS_SOAR_MODEL_DIR,
            "prompts_dir": PROMPTS_DIR,
            "tts_output_dir": DOTS_TTS_SOAR_OUTPUT_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": DOTS_TTS_SOAR_WORKER_SCRIPT,
            "worker_tmp_dir": DOTS_TTS_SOAR_WORKER_TMP_DIR,
        },
        "available": {
            "python": sys.executable,
            "worker_runtime": "uv",
            "worker_script": os.path.isfile(DOTS_TTS_SOAR_WORKER_SCRIPT),
            "model_dir": os.path.isdir(DOTS_TTS_SOAR_MODEL_DIR),
            "model_required_files": all(required_files.values()),
            "model_required_files_detail": required_files,
            "torch": module_available("torch"),
            "cuda": cuda["available"],
            "flash_attn": module_available("flash_attn"),
        },
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model": "rednote-hilab/dots.tts-soar",
            "model_lifecycle": "one request -> one worker -> explicit CUDA cleanup -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": DOTS_TTS_SOAR_REQUEST_TIMEOUT,
            "sample_rate": 48000,
            "precision": DOTS_TTS_SOAR_PRECISION,
            "language": DOTS_TTS_SOAR_LANGUAGE,
            "ode_method": DOTS_TTS_SOAR_ODE_METHOD,
            "seed": DOTS_TTS_SOAR_SEED,
            "num_steps": DOTS_TTS_SOAR_NUM_STEPS,
            "guidance_scale": DOTS_TTS_SOAR_GUIDANCE_SCALE,
            "speaker_scale": DOTS_TTS_SOAR_SPEAKER_SCALE,
            "max_generate_length": DOTS_TTS_SOAR_MAX_GENERATE_LENGTH,
            "max_chars_per_chunk": DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK,
            "pause_ms": DOTS_TTS_SOAR_PAUSE_MS,
            "normalize_text": DOTS_TTS_SOAR_NORMALIZE_TEXT,
            "profile_inference": DOTS_TTS_SOAR_PROFILE_INFERENCE,
            "flash_attention_policy": (
                "not required; dots.tts uses native PyTorch flex_attention and does not import flash_attn"
            ),
            "clone_contract": "prompt_audio + optional exact prompt_text; 48 kHz mono output",
        },
        "last_errors": {"dots_tts_soar": manager.last_error},
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    with gpu_runtime_lock("dots_tts_soar/unload"):
        with manager.lock:
            pass
    return JSONResponse({"code": 200, "msg": "dots.tts-soar worker 已退出，无常驻模型"})


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...),
    prompt_text: str | None = Form(None),
):
    content = await audio.read()
    clone_path = clone_prompt_audio_path(full_path)
    timbre_path = find_matching_timbre_audio(content)
    if timbre_path:
        # 设计音色的原始 WAV 只保存在 timbre；这里仅保存解析引用供克隆服务使用。
        if os.path.isfile(clone_path):
            os.remove(clone_path)
        clone_sidecar_path = os.path.join(PROMPTS_DIR, f"{hash_filename(full_path)}.prompt.txt")
        if os.path.isfile(clone_sidecar_path):
            os.remove(clone_sidecar_path)
        with open(timbre_reference_map_path(full_path), "w", encoding="utf-8") as reference_file:
            reference_file.write(timbre_path)
    else:
        reference_map_path = timbre_reference_map_path(full_path)
        if os.path.isfile(reference_map_path):
            os.remove(reference_map_path)
        with open(clone_path, "wb") as file:
            file.write(content)
    normalized_prompt_text = normalize_optional_text(prompt_text)
    save_prompt_text_sidecar(full_path, normalized_prompt_text)
    return {
        "code": 200,
        "msg": "上传成功",
        "filename": full_path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "has_prompt_text": bool(normalized_prompt_text),
    }


@app.get("/v1/check/audio")
async def check_audio_exists(file_name: str):
    audio_path = prompt_audio_path(file_name)
    exists = os.path.isfile(audio_path)
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "size_bytes": os.path.getsize(audio_path) if exists else None,
        "sha256": sha256_file(audio_path) if exists else None,
        "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
    }


@app.post("/v2/dotsTTS/clone")
async def synthesize_v2(request: DotsTtsSoarSynthesizeRequest):
    with gpu_runtime_lock("dots_tts_soar/synthesize"):
        with manager.lock:
            try:
                payload = manager.build_worker_payload(request)
                audio_bytes = manager.run_worker(payload)
                saved_output_path = persist_audio_bytes(
                    audio_bytes,
                    "dots_tts_soar",
                    DOTS_TTS_SOAR_OUTPUT_DIR,
                )
                print(f"[dots.tts-soar] 已保存生成音频: {saved_output_path}")
                return Response(content=audio_bytes, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as exc:
                LOGGER.exception("dots.tts-soar request failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                wait_after_cuda_release("after dots.tts-soar worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 dots.tts-soar Voice Clone")
    print("==================================================")
    print("[配置] worker runtime: uv")
    print(f"[配置] worker python: {sys.executable}")
    print(f"[配置] model: {DOTS_TTS_SOAR_MODEL_DIR}")
    print(f"[配置] port: {API_PORT}")
    print(
        f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={DOTS_TTS_SOAR_REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
