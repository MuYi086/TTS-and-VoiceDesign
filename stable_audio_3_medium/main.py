#!/usr/bin/env python3
"""Unitale Stable Audio 3 Medium HTTP service.

The API process validates requests and owns the GPU lock, while a fresh worker
process loads the heavyweight model for each request and exits before the
response is returned.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.middleware.base import BaseHTTPMiddleware

from runtime import (
    WorkerConfig,
    cuda_status,
    gpu_runtime_lock,
    module_available,
    persist_audio_bytes,
    run_local_worker,
)

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
REQUIRED_MODEL_FILES = (
    "model_config.json",
    "model.safetensors",
    "t5gemma-b-b-ul2/config.json",
    "t5gemma-b-b-ul2/model.safetensors",
    "t5gemma-b-b-ul2/tokenizer.json",
)
MAX_SECONDS = 380.0


def expand_path(value: str) -> Path:
    """Expand environment variables and ``~`` into an absolute path."""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def env_bool(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def local_model_is_complete(model_dir: Path) -> bool:
    """Check the minimum files required by the local worker."""
    return all((model_dir / name).is_file() for name in REQUIRED_MODEL_FILES)


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(PROJECT_ROOT / "storage")))
SOUNDEFFECT_STORAGE_DIR = expand_path(
    os.getenv("SOUNDEFFECT_STORAGE_DIR", str(STORAGE_DIR / "soundEffect"))
)
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(PROJECT_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8311"))

STABLE_AUDIO_3_MEDIUM_MODEL_DIR = expand_path(
    os.getenv(
        "STABLE_AUDIO_3_MEDIUM_MODEL_DIR",
        str(HF_MIRROR_DIR / "stabilityai/stable-audio-3-medium"),
    )
)
STABLE_AUDIO_3_REPO_PATH = expand_path(
    os.getenv("STABLE_AUDIO_3_REPO_PATH", "~/tts-depency/stable-audio-3")
)
STABLE_AUDIO_3_MEDIUM_DEVICE = os.getenv("STABLE_AUDIO_3_MEDIUM_DEVICE", "cuda")
STABLE_AUDIO_3_MEDIUM_DTYPE = os.getenv("STABLE_AUDIO_3_MEDIUM_DTYPE", "float16")
STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS = float(
    os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS", "7")
)
STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS = int(os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS", "8"))
STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE = float(
    os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE", "1.0")
)
STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED = int(os.getenv("STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED", "-1"))
STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT = float(
    os.getenv("STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT", "900")
)
STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR = expand_path(
    os.getenv("STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR", str(SOUNDEFFECT_STORAGE_DIR))
)
# The upstream runtime has a tested SDPA/flex-attention fallback.  Keep it as
# the uv default because the old cp310 FlashAttention wheel cannot be reused by
# Python 3.12; deployments can set this to 1 for the strict official path.
STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN = env_bool(
    "STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN", False
)
WORKER_SCRIPT = PROJECT_DIR / "worker.py"
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
    """Preserve the permissive CORS behavior used by the local WebUI."""

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
    """Stable Audio 3 Medium text-to-audio request."""

    prompt: str = Field(min_length=1, max_length=2_000)
    seconds: float | None = Field(default=None, gt=0, le=MAX_SECONDS)
    duration: float | None = Field(default=None, gt=0, le=MAX_SECONDS)
    steps: int = Field(default=STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS, ge=1, le=100)
    cfg_scale: float = Field(default=STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE, ge=0, le=100)
    seed: int = Field(default=STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED)
    device: Literal["cuda"] | None = None
    dtype: Literal["float16"] | None = None

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
    """Restrict internal control endpoints to the local machine."""
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="仅允许本机访问内部接口")


def wait_after_cuda_release() -> None:
    """Allow the terminated worker's CUDA context to disappear before unlock."""
    if CUDA_RELEASE_DELAY > 0:
        print(
            f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s，确保 Stable Audio 3 Medium worker 显存已释放"
        )
        time.sleep(CUDA_RELEASE_DELAY)


def flash_attention_status() -> dict[str, object]:
    """Report availability without importing a compiled CUDA extension in the API."""
    available = module_available("flash_attn")
    return {
        "available": available,
        "required_by_default": STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN,
        "mode": "flash-attn" if available else "upstream SDPA/flex fallback",
        "note": (
            "worker imports flash_attn when available; upstream stable-audio-3 "
            "falls back to flex_attention/SDPA when it is absent"
        ),
    }


STABLE_AUDIO_3_MEDIUM_WORKER = WorkerConfig(
    worker_script=WORKER_SCRIPT,
    temp_dir=WORKER_TMP_DIR,
    timeout=STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT,
    label="Stable Audio 3 Medium",
    file_prefix="stable_audio_3_medium",
)


class StableAudio3MediumWorkerManager:
    """Validate local assets, run one worker, and persist successful output."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: str | None = None

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
            "require_flash_attn": STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN,
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
        if not (STABLE_AUDIO_3_REPO_PATH / "stable_audio_3").is_dir():
            raise RuntimeError(
                f"Stable Audio 3 官方源码目录不存在或不完整: {STABLE_AUDIO_3_REPO_PATH}"
            )

        try:
            audio = run_local_worker(payload, STABLE_AUDIO_3_MEDIUM_WORKER)
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
    """Report service readiness without loading model weights."""
    cuda = cuda_status()
    required_files = {
        name: (STABLE_AUDIO_3_MEDIUM_MODEL_DIR / name).is_file() for name in REQUIRED_MODEL_FILES
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
            "project_dir": str(PROJECT_DIR),
            "worker_python": os.environ.get("STABLE_AUDIO_3_MEDIUM_PYTHON", os.sys.executable),
        },
        "available": {
            # Kept for health-schema compatibility; Conda is no longer required.
            "conda": bool(shutil.which("conda")),
            "uv": bool(shutil.which("uv")) or Path(os.sys.executable).is_file(),
            "model_dir": STABLE_AUDIO_3_MEDIUM_MODEL_DIR.is_dir(),
            "model_required_files": all(required_files.values()),
            "model_required_files_detail": required_files,
            "upstream_source": (STABLE_AUDIO_3_REPO_PATH / "stable_audio_3").is_dir(),
            "worker_script": WORKER_SCRIPT.is_file(),
            "torch_in_api_env": module_available("torch"),
            "flash_attn_in_api_env": module_available("flash_attn"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "worker_env": "stable_audio_3_medium/.venv",
            "worker_runtime": "same uv project",
            "model": "stabilityai/stable-audio-3-medium",
            "model_lifecycle": "one request -> one uv worker -> explicit CUDA cleanup -> process exit releases VRAM",
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
            "hardware": "CUDA GPU with Ampere-or-newer compute capability",
            "flash_attention": flash_attention_status(),
            "contract": "supports music and sound effects; not designed for speech or voice generation",
        },
        "last_errors": {"stable_audio_3_medium": manager.last_error},
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request) -> JSONResponse:
    """Keep the old control route even though no model remains resident."""
    assert_local_request(request)
    with gpu_runtime_lock(GPU_LOCK_FILE, "stable_audio_3_medium/unload"):
        with manager.lock:
            pass
    return JSONResponse(
        {
            "code": 200,
            "msg": "Stable Audio 3 Medium uv 服务无常驻模型；worker 退出后显存已释放。",
        }
    )


@app.post("/v1/stableAudio/soundEffect")
async def generate(request: StableAudio3MediumGenerateRequest) -> Response:
    """Generate a WAV while serializing access to the shared GPU."""
    with gpu_runtime_lock(GPU_LOCK_FILE, "stable_audio_3_medium/generate"):
        with manager.lock:
            try:
                audio = manager.run_worker(manager.build_worker_payload(request))
                saved_output_path = persist_audio_bytes(
                    audio,
                    "stable_audio_3_medium",
                    STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR,
                )
                print(f"[Stable Audio 3 Medium] 已保存生成音频: {saved_output_path}")
                return Response(content=audio, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as exc:
                LOGGER.exception("Stable Audio 3 Medium request failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                wait_after_cuda_release()


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale Stable Audio 3 Medium uv API")
    print("==================================================")
    print(f"[配置] uv project: {PROJECT_DIR}")
    print(f"[配置] 模型目录: {STABLE_AUDIO_3_MEDIUM_MODEL_DIR}")
    print(f"[配置] 官方源码: {STABLE_AUDIO_3_REPO_PATH}")
    print(f"[配置] 输出目录: {STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        "[配置] "
        f"local_files_only={LOCAL_FILES_ONLY}, timeout={STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT}, "
        f"require_flash_attn={STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
