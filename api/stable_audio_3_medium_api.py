#!/usr/bin/env python3
"""HTTP wrapper for one-shot Stable Audio 3 Medium generation.

The API process deliberately does not import the Stable Audio runtime.  Each
request is delegated to a worker in the dedicated Medium Conda environment;
the worker clears its CUDA allocations and exits before the response is
returned, so the 1.4B model is never kept resident in the API process.
"""

from __future__ import annotations

import fcntl
import importlib.util
import os
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.middleware.base import BaseHTTPMiddleware

from audio_output import persist_audio_bytes
from gpu_runtime import cuda_status
from local_worker import LocalWorkerConfig, resolve_conda_executable, run_local_worker


API_DIR = Path(__file__).resolve().parent
REQUIRED_MODEL_FILES = (
    "model_config.json",
    "model.safetensors",
    "t5gemma-b-b-ul2/config.json",
    "t5gemma-b-b-ul2/model.safetensors",
    "t5gemma-b-b-ul2/tokenizer.json",
)
# Stable Audio 3 Medium's official maximum is 380 seconds.
MAX_SECONDS = 380.0


def expand_path(value: str) -> Path:
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def local_model_is_complete(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in REQUIRED_MODEL_FILES)


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(API_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
TTS_OUTPUT_DIR = expand_path(os.getenv("TTS_OUTPUT_DIR", str(API_DIR / "tempAudio")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8313"))

STABLE_AUDIO_3_MEDIUM_CONDA_ENV = os.getenv(
    "STABLE_AUDIO_3_MEDIUM_CONDA_ENV", "stable_audio_3_medium"
)
STABLE_AUDIO_3_MEDIUM_MODEL_DIR = expand_path(
    os.getenv(
        "STABLE_AUDIO_3_MEDIUM_MODEL_DIR",
        str(HF_MIRROR_DIR / "stabilityai/stable-audio-3-medium"),
    )
)
STABLE_AUDIO_3_REPO_PATH = expand_path(
    os.getenv("STABLE_AUDIO_3_REPO_PATH", "~/tts-depency/stable-audio-3")
)
# Medium is GPU-only in the official implementation.  The worker validates
# this value again instead of silently falling back to CPU.
STABLE_AUDIO_3_MEDIUM_DEVICE = os.getenv("STABLE_AUDIO_3_MEDIUM_DEVICE", "cuda")
STABLE_AUDIO_3_MEDIUM_DTYPE = os.getenv("STABLE_AUDIO_3_MEDIUM_DTYPE", "float16")
STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS = float(
    os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS", "7")
)
STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS = int(
    os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS", "8")
)
STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE = float(
    os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE", "1.0")
)
STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED = int(
    os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED", "-1")
)
STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT = float(
    os.getenv("STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT", "900")
)
STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR = expand_path(
    os.getenv("STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR", str(TTS_OUTPUT_DIR))
)
WORKER_SCRIPT = API_DIR / "stable_audio_3_medium_worker.py"
WORKER_TMP_DIR = RUNTIME_CACHE_DIR / "stable_audio_3_medium_worker"

for directory in (
    RUNTIME_CACHE_DIR,
    WORKER_TMP_DIR,
    STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR,
    GPU_LOCK_FILE.parent,
):
    directory.mkdir(parents=True, exist_ok=True)

if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


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


class StableAudio3MediumGenerateRequest(BaseModel):
    """Stable Audio 3 Medium text-to-audio parameters.

    ``seconds`` keeps the same request contract as MOSS-SoundEffect and the
    Small-SFX adapter; it is passed to the official runtime as ``duration``.
    """

    prompt: str = Field(min_length=1, max_length=2_000)
    # ``seconds`` is the existing MOSS/WebUI contract; ``duration`` is the
    # official Stable Audio spelling.  Both normalize to worker ``seconds``.
    seconds: Optional[float] = Field(default=None, gt=0, le=MAX_SECONDS)
    duration: Optional[float] = Field(default=None, gt=0, le=MAX_SECONDS)
    steps: int = Field(default=STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS, ge=1, le=100)
    cfg_scale: float = Field(default=STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE, ge=0, le=100)
    seed: int = Field(default=STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED)
    device: Optional[Literal["cuda"]] = None
    dtype: Optional[Literal["float16"]] = None

    @field_validator("prompt")
    @classmethod
    def trim_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt 不能为空")
        return normalized

    @model_validator(mode="after")
    def normalize_duration(self):
        if self.seconds is None and self.duration is None:
            self.seconds = STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS
        elif self.seconds is None:
            self.seconds = self.duration
        elif self.duration is not None and self.seconds != self.duration:
            raise ValueError("seconds 与 duration 必须相同")
        return self


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


def wait_after_cuda_release() -> None:
    if CUDA_RELEASE_DELAY > 0:
        print(
            "[CUDA] 等待 "
            f"{CUDA_RELEASE_DELAY:.1f}s，确保 Stable Audio 3 Medium worker 显存已释放"
        )
        time.sleep(CUDA_RELEASE_DELAY)


STABLE_AUDIO_3_MEDIUM_WORKER = LocalWorkerConfig(
    conda_env=STABLE_AUDIO_3_MEDIUM_CONDA_ENV,
    worker_script=str(WORKER_SCRIPT),
    model_dir=str(STABLE_AUDIO_3_MEDIUM_MODEL_DIR),
    temp_dir=str(WORKER_TMP_DIR),
    timeout=STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT,
    label="Stable Audio 3 Medium",
    file_prefix="stable_audio_3_medium",
)


class StableAudio3MediumWorkerManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: StableAudio3MediumGenerateRequest) -> dict:
        return {
            "prompt": request.prompt,
            "seconds": request.seconds,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "seed": request.seed,
            "device": request.device or STABLE_AUDIO_3_MEDIUM_DEVICE,
            "dtype": request.dtype or STABLE_AUDIO_3_MEDIUM_DTYPE,
            "model_path": str(STABLE_AUDIO_3_MEDIUM_MODEL_DIR),
            "upstream_path": str(STABLE_AUDIO_3_REPO_PATH),
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": str(RUNTIME_CACHE_DIR),
            "hf_mirror_dir": str(HF_MIRROR_DIR),
        }

    def run_worker(self, payload: dict) -> bytes:
        if not local_model_is_complete(STABLE_AUDIO_3_MEDIUM_MODEL_DIR):
            missing = [
                name
                for name in REQUIRED_MODEL_FILES
                if not (STABLE_AUDIO_3_MEDIUM_MODEL_DIR / name).is_file()
            ]
            raise RuntimeError(
                "Stable Audio 3 Medium 本地权重不完整或不存在: "
                f"{STABLE_AUDIO_3_MEDIUM_MODEL_DIR}; 缺少: {', '.join(missing)}"
            )
        if not STABLE_AUDIO_3_REPO_PATH.joinpath("stable_audio_3").is_dir():
            raise RuntimeError(
                "Stable Audio 3 官方源码目录不存在或不完整: "
                f"{STABLE_AUDIO_3_REPO_PATH}"
            )

        try:
            audio = run_local_worker(payload, STABLE_AUDIO_3_MEDIUM_WORKER)
            saved_output_path = persist_audio_bytes(
                audio,
                "stable_audio_3_medium",
                str(STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR),
            )
            print(f"[Stable Audio 3 Medium] 已保存生成音频: {saved_output_path}")
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise


app = FastAPI(title="Unitale Stable Audio 3 Medium API")
app.add_middleware(ForceCORS)
manager = StableAudio3MediumWorkerManager()


@app.get("/v1/health")
async def health() -> dict:
    cuda = cuda_status()
    required_files = {
        name: (STABLE_AUDIO_3_MEDIUM_MODEL_DIR / name).is_file()
        for name in REQUIRED_MODEL_FILES
    }
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": str(HF_MIRROR_DIR),
            "model_dir": str(STABLE_AUDIO_3_MEDIUM_MODEL_DIR),
            "upstream_path": str(STABLE_AUDIO_3_REPO_PATH),
            "tts_output_dir": str(STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR),
            "worker_script": str(WORKER_SCRIPT),
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "gpu_lock_file": str(GPU_LOCK_FILE),
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "model_dir": STABLE_AUDIO_3_MEDIUM_MODEL_DIR.is_dir(),
            "model_required_files": all(required_files.values()),
            "model_required_files_detail": required_files,
            "upstream_source": STABLE_AUDIO_3_REPO_PATH.joinpath("stable_audio_3").is_dir(),
            "worker_script": WORKER_SCRIPT.is_file(),
            "torch_in_api_env": module_available("torch"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "worker_env": STABLE_AUDIO_3_MEDIUM_CONDA_ENV,
            "model": "stabilityai/stable-audio-3-medium",
            "model_lifecycle": "one request -> one worker -> explicit CUDA cleanup -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT,
            "device": STABLE_AUDIO_3_MEDIUM_DEVICE,
            "dtype": STABLE_AUDIO_3_MEDIUM_DTYPE,
            "default_seconds": STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS,
            "default_steps": STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS,
            "default_cfg_scale": STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE,
            "default_seed": STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED,
            "sample_rate": 44100,
            "channels": 2,
            "max_seconds": MAX_SECONDS,
            "prompt_language": "English recommended; other languages may underperform",
            "hardware": "CUDA GPU with Ampere-or-newer compute capability and Flash Attention 2",
            "contract": "supports music and sound effects; not designed for speech or voice generation",
        },
        "last_errors": {"stable_audio_3_medium": manager.last_error},
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request) -> JSONResponse:
    assert_local_request(request)
    with gpu_runtime_lock("stable_audio_3_medium/unload"):
        with manager.lock:
            pass
    return JSONResponse(
        {
            "code": 200,
            "msg": "Stable Audio 3 Medium wrapper 无常驻模型；worker 已退出后显存已释放。",
        }
    )


@app.post("/v1/generate")
@app.post("/v2/synthesize")
async def generate(request: StableAudio3MediumGenerateRequest) -> Response:
    with gpu_runtime_lock("stable_audio_3_medium/generate"):
        with manager.lock:
            try:
                audio = manager.run_worker(manager.build_worker_payload(request))
                return Response(content=audio, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as exc:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                wait_after_cuda_release()


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale Stable Audio 3 Medium API")
    print("==================================================")
    print(f"[配置] worker env: {STABLE_AUDIO_3_MEDIUM_CONDA_ENV}")
    print(f"[配置] 模型目录: {STABLE_AUDIO_3_MEDIUM_MODEL_DIR}")
    print(f"[配置] 官方源码: {STABLE_AUDIO_3_REPO_PATH}")
    print(f"[配置] 输出目录: {STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        "[配置] "
        f"local_files_only={LOCAL_FILES_ONLY}, timeout={STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
