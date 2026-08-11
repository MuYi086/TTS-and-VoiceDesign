"""HTTP wrapper for one-shot dots.tts-soar voice cloning."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import threading
import traceback
from contextlib import contextmanager
from typing import Literal, Optional

import fcntl
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Field
from starlette.middleware.base import BaseHTTPMiddleware

from audio_output import persist_audio_bytes
from gpu_runtime import cuda_status
from local_worker import LocalWorkerConfig, resolve_conda_executable, run_local_worker
from synthesis_request import CloneSynthesisRequest


API_DIR = os.path.dirname(os.path.abspath(__file__))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def normalize_optional_text(value: Optional[str]) -> Optional[str]:
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


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", os.path.join(API_DIR, "prompts")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", os.path.join(API_DIR, ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8308"))

DOTS_TTS_SOAR_CONDA_ENV = os.getenv("DOTS_TTS_SOAR_CONDA_ENV", "dots_tts_soar")
DOTS_TTS_SOAR_MODEL_DIR = expand_path(
    os.getenv("DOTS_TTS_SOAR_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "rednote-hilab/dots.tts-soar"))
)
DOTS_TTS_SOAR_PRECISION = os.getenv("DOTS_TTS_SOAR_PRECISION", "bfloat16")
DOTS_TTS_SOAR_LANGUAGE = normalize_optional_text(os.getenv("DOTS_TTS_SOAR_LANGUAGE", "chinese"))
DOTS_TTS_SOAR_ODE_METHOD = os.getenv("DOTS_TTS_SOAR_ODE_METHOD", "euler")
DOTS_TTS_SOAR_NUM_STEPS = int(os.getenv("DOTS_TTS_SOAR_NUM_STEPS", "10"))
DOTS_TTS_SOAR_GUIDANCE_SCALE = float(os.getenv("DOTS_TTS_SOAR_GUIDANCE_SCALE", "1.2"))
DOTS_TTS_SOAR_SPEAKER_SCALE = float(os.getenv("DOTS_TTS_SOAR_SPEAKER_SCALE", "1.5"))
DOTS_TTS_SOAR_MAX_GENERATE_LENGTH = int(os.getenv("DOTS_TTS_SOAR_MAX_GENERATE_LENGTH", "500"))
DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK = int(os.getenv("DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK", "120"))
DOTS_TTS_SOAR_PAUSE_MS = int(os.getenv("DOTS_TTS_SOAR_PAUSE_MS", "250"))
DOTS_TTS_SOAR_SEED = int(os.getenv("DOTS_TTS_SOAR_SEED", "42"))
DOTS_TTS_SOAR_NORMALIZE_TEXT = env_bool("DOTS_TTS_SOAR_NORMALIZE_TEXT", False)
DOTS_TTS_SOAR_PROFILE_INFERENCE = env_bool("DOTS_TTS_SOAR_PROFILE_INFERENCE", False)
DOTS_TTS_SOAR_REQUEST_TIMEOUT = float(os.getenv("DOTS_TTS_SOAR_REQUEST_TIMEOUT", "900"))
DOTS_TTS_SOAR_WORKER_SCRIPT = os.path.join(API_DIR, "dots_tts_soar_worker.py")
DOTS_TTS_SOAR_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "dots_tts_soar_worker")
DOTS_TTS_SOAR_OUTPUT_DIR = expand_path(
    os.getenv(
        "DOTS_TTS_SOAR_OUTPUT_DIR",
        os.getenv("TTS_OUTPUT_DIR", os.path.join(API_DIR, "tempAudio")),
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


def prompt_text_sidecar_path(filename: str) -> str:
    return os.path.join(PROMPTS_DIR, f"{hash_filename(filename)}.prompt.txt")


def load_prompt_text_sidecar(filename: str) -> Optional[str]:
    path = prompt_text_sidecar_path(filename)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return normalize_optional_text(file.read())


def save_prompt_text_sidecar(filename: str, prompt_text: Optional[str]) -> None:
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
    prompt_text: Optional[str] = None
    language: Optional[str] = None
    template_name: Optional[Literal["tts", "instruction_tts", "text_to_audio", "tts_interleave"]] = None
    precision: Optional[str] = None
    seed: Optional[int] = None
    ode_method: Optional[str] = None
    num_steps: Optional[int] = Field(default=None, ge=1)
    guidance_scale: Optional[float] = Field(default=None, ge=0)
    speaker_scale: Optional[float] = Field(default=None, ge=0)
    max_generate_length: Optional[int] = Field(default=None, ge=1)
    max_chars_per_chunk: Optional[int] = Field(default=None, ge=0)
    pause_ms: Optional[int] = Field(default=None, ge=0)
    normalize_text: Optional[bool] = None
    profile_inference: Optional[bool] = None


DOTS_TTS_SOAR_WORKER = LocalWorkerConfig(
    conda_env=DOTS_TTS_SOAR_CONDA_ENV,
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
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: DotsTtsSoarSynthesizeRequest) -> dict:
        ref_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
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
            "num_steps": request.num_steps if request.num_steps is not None else DOTS_TTS_SOAR_NUM_STEPS,
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
            "pause_ms": request.pause_ms if request.pause_ms is not None else DOTS_TTS_SOAR_PAUSE_MS,
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
            audio = run_local_worker(payload, DOTS_TTS_SOAR_WORKER)
            saved_output_path = persist_audio_bytes(
                audio,
                "dots_tts_soar",
                DOTS_TTS_SOAR_OUTPUT_DIR,
            )
            print(f"[dots.tts-soar] 已保存生成音频: {saved_output_path}")
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
        for name in ("config.json", "model.safetensors", "speaker_encoder.safetensors", "vocoder.safetensors")
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
            "conda": bool(resolve_conda_executable()),
            "worker_script": os.path.isfile(DOTS_TTS_SOAR_WORKER_SCRIPT),
            "model_dir": os.path.isdir(DOTS_TTS_SOAR_MODEL_DIR),
            "model_required_files": all(required_files.values()),
            "model_required_files_detail": required_files,
            "torch": module_available("torch"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "worker_env": DOTS_TTS_SOAR_CONDA_ENV,
            "model": "rednote-hilab/dots.tts-soar",
            "model_lifecycle": "one request -> one worker -> explicit CUDA cleanup -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": DOTS_TTS_SOAR_REQUEST_TIMEOUT,
            "sample_rate": 48000,
            "precision": DOTS_TTS_SOAR_PRECISION,
            "language": DOTS_TTS_SOAR_LANGUAGE,
            "num_steps": DOTS_TTS_SOAR_NUM_STEPS,
            "guidance_scale": DOTS_TTS_SOAR_GUIDANCE_SCALE,
            "speaker_scale": DOTS_TTS_SOAR_SPEAKER_SCALE,
            "max_generate_length": DOTS_TTS_SOAR_MAX_GENERATE_LENGTH,
            "max_chars_per_chunk": DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK,
            "pause_ms": DOTS_TTS_SOAR_PAUSE_MS,
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
    prompt_text: Optional[str] = Form(None),
):
    content = await audio.read()
    save_path = os.path.join(PROMPTS_DIR, hash_filename(full_path))
    with open(save_path, "wb") as file:
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
    audio_path = os.path.join(PROMPTS_DIR, hash_filename(file_name))
    exists = os.path.isfile(audio_path)
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "size_bytes": os.path.getsize(audio_path) if exists else None,
        "sha256": sha256_file(audio_path) if exists else None,
        "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
    }


@app.post("/v2/synthesize")
async def synthesize_v2(request: DotsTtsSoarSynthesizeRequest):
    with gpu_runtime_lock("dots_tts_soar/synthesize"):
        with manager.lock:
            try:
                payload = manager.build_worker_payload(request)
                audio_bytes = manager.run_worker(payload)
                return Response(content=audio_bytes, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as exc:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                wait_after_cuda_release("after dots.tts-soar worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 dots.tts-soar Voice Clone")
    print("==================================================")
    print(f"[配置] worker env: {DOTS_TTS_SOAR_CONDA_ENV}")
    print(f"[配置] model: {DOTS_TTS_SOAR_MODEL_DIR}")
    print(f"[配置] port: {API_PORT}")
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={DOTS_TTS_SOAR_REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
