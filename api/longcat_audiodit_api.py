"""HTTP wrapper for one-shot LongCat-AudioDiT-3.5B voice cloning."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
import re
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Field
from starlette.middleware.base import BaseHTTPMiddleware

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


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", os.path.join(API_DIR, "prompts")))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", os.path.join(API_DIR, ".cache/runtime"))
)
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8307"))

LONGCAT_AUDIODIT_CONDA_ENV = os.getenv(
    "LONGCAT_AUDIODIT_CONDA_ENV", "LongCat-AudioDiT-3.5B-bf16"
)
LONGCAT_AUDIODIT_MODEL_DIR = expand_path(
    os.getenv(
        "LONGCAT_AUDIODIT_MODEL_DIR",
        os.getenv(
            "LONGCAT_AUDIODIT_35B_BF16_MODEL_PATH",
            os.path.join(HF_MIRROR_DIR, "drbaph/LongCat-AudioDiT-3.5B-bf16"),
        ),
    )
)
LONGCAT_AUDIODIT_REPO_PATH = expand_path(
    os.getenv("LONGCAT_AUDIODIT_REPO_PATH", "~/tts-depency/LongCat-AudioDiT")
)
LONGCAT_AUDIODIT_TOKENIZER_PATH = expand_path(
    os.getenv("LONGCAT_AUDIODIT_TOKENIZER_PATH", "~/hf-mirror/google/umt5-base")
)
LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK = int(
    os.getenv("LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK", "180")
)
LONGCAT_AUDIODIT_PAUSE_MS = int(os.getenv("LONGCAT_AUDIODIT_PAUSE_MS", "250"))
LONGCAT_AUDIODIT_NFE = int(os.getenv("LONGCAT_AUDIODIT_NFE", "16"))
LONGCAT_AUDIODIT_GUIDANCE_STRENGTH = float(
    os.getenv("LONGCAT_AUDIODIT_GUIDANCE_STRENGTH", "4.0")
)
LONGCAT_AUDIODIT_GUIDANCE_METHOD = os.getenv("LONGCAT_AUDIODIT_GUIDANCE_METHOD", "apg")
LONGCAT_AUDIODIT_SEED = int(os.getenv("LONGCAT_AUDIODIT_SEED", "20260614"))
LONGCAT_AUDIODIT_DURATION_SCALE = float(
    os.getenv("LONGCAT_AUDIODIT_DURATION_SCALE", "1.0")
)
LONGCAT_AUDIODIT_VAE_DTYPE = os.getenv("LONGCAT_AUDIODIT_VAE_DTYPE", "float16")
LONGCAT_AUDIODIT_REQUEST_TIMEOUT = float(
    os.getenv("LONGCAT_AUDIODIT_REQUEST_TIMEOUT", "900")
)
LONGCAT_AUDIODIT_WORKER_SCRIPT = os.path.join(API_DIR, "longcat_audiodit_worker.py")
LONGCAT_AUDIODIT_WORKER_TMP_DIR = os.path.join(
    RUNTIME_CACHE_DIR, "longcat_audiodit_worker"
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
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(LONGCAT_AUDIODIT_WORKER_TMP_DIR, exist_ok=True)
lock_dir = os.path.dirname(GPU_LOCK_FILE)
if lock_dir:
    os.makedirs(lock_dir, exist_ok=True)

app = FastAPI(title="Unitale LongCat-AudioDiT-3.5B Voice Clone API")


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
        text = file.read().strip()
    return text or None


def save_prompt_text_sidecar(filename: str, prompt_text: Optional[str]) -> None:
    path = prompt_text_sidecar_path(filename)
    normalized = prompt_text.strip() if prompt_text and prompt_text.strip() else None
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


def normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def normalize_synthesis_text(text: str) -> str:
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise HTTPException(status_code=400, detail="text 不能为空。")
    return normalized


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


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


def wait_after_cuda_release(label: str = "") -> None:
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


class LongCatAudioDitSynthesizeRequest(CloneSynthesisRequest):
    """Official LongCat zero-shot voice-cloning request."""

    text: str
    audio_path: str
    prompt_text: Optional[str] = None
    max_chars_per_chunk: Optional[int] = Field(default=None, ge=0)
    pause_ms: Optional[int] = Field(default=None, ge=0)
    nfe: Optional[int] = Field(default=None, ge=2)
    guidance_strength: Optional[float] = Field(default=None, ge=0)
    guidance_method: Optional[Literal["cfg", "apg"]] = None
    seed: Optional[int] = None
    duration_scale: Optional[float] = Field(default=None, gt=0)
    vae_dtype: Optional[Literal["float16", "float32"]] = None


LONGCAT_WORKER = LocalWorkerConfig(
    conda_env=LONGCAT_AUDIODIT_CONDA_ENV,
    worker_script=LONGCAT_AUDIODIT_WORKER_SCRIPT,
    model_dir=LONGCAT_AUDIODIT_MODEL_DIR,
    temp_dir=LONGCAT_AUDIODIT_WORKER_TMP_DIR,
    timeout=LONGCAT_AUDIODIT_REQUEST_TIMEOUT,
    label="LongCat-AudioDiT-3.5B",
    file_prefix="longcat_audiodit",
)


class LongCatAudioDitWorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: LongCatAudioDitSynthesizeRequest) -> dict:
        ref_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        prompt_text = normalize_optional_text(request.prompt_text)
        if prompt_text is None:
            prompt_text = load_prompt_text_sidecar(request.audio_path)
        if prompt_text is None:
            raise HTTPException(
                status_code=400,
                detail="LongCat-AudioDiT 克隆需要参考音频的准确 prompt_text。",
            )

        return {
            "operation": "clone",
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "model_path": LONGCAT_AUDIODIT_MODEL_DIR,
            "repo_path": LONGCAT_AUDIODIT_REPO_PATH,
            "tokenizer_path": LONGCAT_AUDIODIT_TOKENIZER_PATH,
            "default_tokenizer_path": LONGCAT_AUDIODIT_TOKENIZER_PATH,
            "max_chars_per_chunk": (
                request.max_chars_per_chunk
                if request.max_chars_per_chunk is not None
                else LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK
            ),
            "pause_ms": request.pause_ms if request.pause_ms is not None else LONGCAT_AUDIODIT_PAUSE_MS,
            "nfe": request.nfe if request.nfe is not None else LONGCAT_AUDIODIT_NFE,
            "guidance_strength": (
                request.guidance_strength
                if request.guidance_strength is not None
                else LONGCAT_AUDIODIT_GUIDANCE_STRENGTH
            ),
            "guidance_method": (
                request.guidance_method
                if request.guidance_method is not None
                else LONGCAT_AUDIODIT_GUIDANCE_METHOD
            ),
            "seed": request.seed if request.seed is not None else LONGCAT_AUDIODIT_SEED,
            "duration_scale": (
                request.duration_scale
                if request.duration_scale is not None
                else LONGCAT_AUDIODIT_DURATION_SCALE
            ),
            "vae_dtype": request.vae_dtype or LONGCAT_AUDIODIT_VAE_DTYPE,
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def run_worker(self, payload: dict) -> bytes:
        try:
            audio = run_local_worker(payload, LONGCAT_WORKER)
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise


manager = LongCatAudioDitWorkerManager()


@app.get("/v1/health")
async def health():
    cuda = cuda_status()
    model_required = (
        os.path.isfile(os.path.join(LONGCAT_AUDIODIT_MODEL_DIR, "config.json"))
        and os.path.isfile(os.path.join(LONGCAT_AUDIODIT_MODEL_DIR, "model.safetensors"))
    )
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "model_dir": LONGCAT_AUDIODIT_MODEL_DIR,
            "repo_path": LONGCAT_AUDIODIT_REPO_PATH,
            "tokenizer_path": LONGCAT_AUDIODIT_TOKENIZER_PATH,
            "prompts_dir": PROMPTS_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": LONGCAT_AUDIODIT_WORKER_SCRIPT,
            "worker_tmp_dir": LONGCAT_AUDIODIT_WORKER_TMP_DIR,
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "worker_script": os.path.isfile(LONGCAT_AUDIODIT_WORKER_SCRIPT),
            "model_dir": os.path.isdir(LONGCAT_AUDIODIT_MODEL_DIR),
            "model_required_files": model_required,
            "repo_path": os.path.isdir(LONGCAT_AUDIODIT_REPO_PATH),
            "tokenizer_path": os.path.isdir(LONGCAT_AUDIODIT_TOKENIZER_PATH),
            "torch": module_available("torch"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "worker_env": LONGCAT_AUDIODIT_CONDA_ENV,
            "model": "LongCat-AudioDiT-3.5B-bf16",
            "model_lifecycle": "one request -> one worker -> explicit CUDA cleanup -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": LONGCAT_AUDIODIT_REQUEST_TIMEOUT,
            "sample_rate": 24000,
            "max_chars_per_chunk": LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK,
            "pause_ms": LONGCAT_AUDIODIT_PAUSE_MS,
            "nfe": LONGCAT_AUDIODIT_NFE,
            "guidance_strength": LONGCAT_AUDIODIT_GUIDANCE_STRENGTH,
            "guidance_method": LONGCAT_AUDIODIT_GUIDANCE_METHOD,
            "seed": LONGCAT_AUDIODIT_SEED,
            "duration_scale": LONGCAT_AUDIODIT_DURATION_SCALE,
            "vae_dtype": LONGCAT_AUDIODIT_VAE_DTYPE,
            "clone_contract": "accurate prompt_text + 24 kHz mono prompt_audio; model max_wav_duration applies",
        },
        "last_errors": {"longcat_audiodit": manager.last_error},
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    with gpu_runtime_lock("longcat_audiodit/unload"):
        with manager.lock:
            pass
    return JSONResponse({"code": 200, "msg": "LongCat worker 已退出，无常驻模型"})


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
async def synthesize_v2(request: LongCatAudioDitSynthesizeRequest):
    with gpu_runtime_lock("longcat_audiodit/synthesize"):
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
                wait_after_cuda_release("after LongCat-AudioDiT worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 LongCat-AudioDiT-3.5B")
    print("==================================================")
    print(f"[配置] worker env: {LONGCAT_AUDIODIT_CONDA_ENV}")
    print(f"[配置] model: {LONGCAT_AUDIODIT_MODEL_DIR}")
    print(f"[配置] official repo: {LONGCAT_AUDIODIT_REPO_PATH}")
    print(f"[配置] tokenizer: {LONGCAT_AUDIODIT_TOKENIZER_PATH}")
    print(f"[配置] port: {API_PORT}")
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={LONGCAT_AUDIODIT_REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
