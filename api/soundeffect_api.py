#!/usr/bin/env python3
"""HTTP wrapper for one-shot MOSS-SoundEffect v2.0 generation.

The wrapper never imports the SoundEffect model.  Every generation runs in a
fresh worker process in the model's dedicated Conda environment.  When that
process exits, CUDA destroys its allocations, so this service cannot keep the
model resident in VRAM between requests.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware


API_DIR = Path(__file__).resolve().parent


def expand_path(value: str) -> Path:
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_conda_executable() -> Optional[str]:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and expand_path(conda_exe).is_file():
        return str(expand_path(conda_exe))
    return shutil.which("conda")


def local_model_is_complete(model_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            model_dir / "model_index.json",
            model_dir / "transformer" / "diffusion_pytorch_model.safetensors",
            model_dir / "vae" / "vae_128d_48k.pth",
        )
    )


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(API_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", str(RUNTIME_CACHE_DIR / "gpu-runtime.lock")))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8311"))

MOSS_SOUNDEFFECT_CONDA_ENV = os.getenv("MOSS_SOUNDEFFECT_CONDA_ENV", "moss-soundEffect")
MOSS_SOUNDEFFECT_MODEL_DIR = expand_path(
    os.getenv(
        "MOSS_SOUNDEFFECT_MODEL_DIR",
        str(HF_MIRROR_DIR / "OpenMOSS-Team/MOSS-SoundEffect-v2.0"),
    )
)
MOSS_SOUNDEFFECT_DEVICE = os.getenv("MOSS_SOUNDEFFECT_DEVICE", "cuda")
MOSS_SOUNDEFFECT_DTYPE = os.getenv("MOSS_SOUNDEFFECT_DTYPE", "bfloat16")
MOSS_SOUNDEFFECT_DEFAULT_SECONDS = float(os.getenv("MOSS_SOUNDEFFECT_DEFAULT_SECONDS", "10"))
MOSS_SOUNDEFFECT_DEFAULT_STEPS = int(os.getenv("MOSS_SOUNDEFFECT_DEFAULT_STEPS", "100"))
MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE = float(os.getenv("MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE", "4.0"))
MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT = float(os.getenv("MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT", "5.0"))
MOSS_SOUNDEFFECT_DEFAULT_SEED = int(os.getenv("MOSS_SOUNDEFFECT_DEFAULT_SEED", "0"))
MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO = env_bool("MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO", True)
MOSS_SOUNDEFFECT_REQUEST_TIMEOUT = float(os.getenv("MOSS_SOUNDEFFECT_REQUEST_TIMEOUT", "600"))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))

WORKER_SCRIPT = API_DIR / "soundeffect_worker.py"
WORKER_TMP_DIR = RUNTIME_CACHE_DIR / "soundeffect_worker"
for directory in (RUNTIME_CACHE_DIR, WORKER_TMP_DIR, GPU_LOCK_FILE.parent):
    directory.mkdir(parents=True, exist_ok=True)


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


class SoundEffectGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    seconds: float = Field(default=MOSS_SOUNDEFFECT_DEFAULT_SECONDS, gt=0, le=30)
    num_inference_steps: int = Field(default=MOSS_SOUNDEFFECT_DEFAULT_STEPS, gt=0, le=500)
    cfg_scale: float = Field(default=MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE, ge=0, le=100)
    sigma_shift: float = Field(default=MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT, gt=0, le=100)
    seed: int = Field(default=MOSS_SOUNDEFFECT_DEFAULT_SEED)
    device: Optional[str] = Field(default=None, min_length=1)
    torch_dtype: Optional[str] = Field(default=None, min_length=1)

    @field_validator("prompt")
    @classmethod
    def trim_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt 不能为空")
        return normalized


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


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-8:]) if lines else "SoundEffect worker 未输出错误信息。"


def wait_after_cuda_release() -> None:
    if CUDA_RELEASE_DELAY > 0:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s，确保 SoundEffect worker 显存已释放")
        time.sleep(CUDA_RELEASE_DELAY)


def assert_local_request(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="仅允许本机访问内部接口")


class SoundEffectWorkerManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: SoundEffectGenerateRequest) -> dict:
        return {
            "prompt": request.prompt,
            "seconds": request.seconds,
            "num_inference_steps": request.num_inference_steps,
            "cfg_scale": request.cfg_scale,
            "sigma_shift": request.sigma_shift,
            "seed": request.seed,
            "device": request.device or MOSS_SOUNDEFFECT_DEVICE,
            "torch_dtype": request.torch_dtype or MOSS_SOUNDEFFECT_DTYPE,
            "model_path": str(MOSS_SOUNDEFFECT_MODEL_DIR),
            "local_files_only": LOCAL_FILES_ONLY,
            "disable_torchdynamo": MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO,
        }

    def run_worker(self, payload: dict) -> bytes:
        conda_exe = resolve_conda_executable()
        if not conda_exe:
            raise RuntimeError("未找到 conda 命令，无法调用 SoundEffect worker。")
        if not WORKER_SCRIPT.is_file():
            raise RuntimeError(f"SoundEffect worker 脚本不存在: {WORKER_SCRIPT}")
        if not local_model_is_complete(MOSS_SOUNDEFFECT_MODEL_DIR):
            raise RuntimeError(f"MOSS-SoundEffect v2.0 本地权重不完整或不存在: {MOSS_SOUNDEFFECT_MODEL_DIR}")

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR, prefix="soundeffect_req_", suffix=".json"
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR, prefix="soundeffect_out_", suffix=".wav"
        )
        os.close(request_fd)
        os.close(output_fd)
        try:
            with open(request_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)

            command = [
                conda_exe,
                "run",
                "--no-capture-output",
                "-n",
                MOSS_SOUNDEFFECT_CONDA_ENV,
                "python",
                str(WORKER_SCRIPT),
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[SoundEffect] 启动一次性 worker: env={MOSS_SOUNDEFFECT_CONDA_ENV}")
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=os.environ.copy(),
            )
            try:
                stdout, stderr = process.communicate(timeout=MOSS_SOUNDEFFECT_REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=10)
                raise RuntimeError(f"SoundEffect worker 超时（>{MOSS_SOUNDEFFECT_REQUEST_TIMEOUT:.0f}s）")

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[SoundEffect] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s")
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("SoundEffect worker 未生成音频文件。")
            with open(output_path, "rb") as file:
                audio = file.read()
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            for path in (request_path, output_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass


app = FastAPI(title="Unitale MOSS-SoundEffect v2 API")
app.add_middleware(ForceCORS)
manager = SoundEffectWorkerManager()


@app.get("/v1/health")
async def health() -> dict:
    return {
        "code": 200,
        "paths": {
            "model_dir": str(MOSS_SOUNDEFFECT_MODEL_DIR),
            "worker_script": str(WORKER_SCRIPT),
            "worker_tmp_dir": str(WORKER_TMP_DIR),
            "gpu_lock_file": str(GPU_LOCK_FILE),
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "model_dir": MOSS_SOUNDEFFECT_MODEL_DIR.is_dir(),
            "model_weights_complete": local_model_is_complete(MOSS_SOUNDEFFECT_MODEL_DIR),
            "worker_script": WORKER_SCRIPT.is_file(),
        },
        "runtime": {
            "worker_env": MOSS_SOUNDEFFECT_CONDA_ENV,
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": MOSS_SOUNDEFFECT_REQUEST_TIMEOUT,
            "device": MOSS_SOUNDEFFECT_DEVICE,
            "torch_dtype": MOSS_SOUNDEFFECT_DTYPE,
            "default_seconds": MOSS_SOUNDEFFECT_DEFAULT_SECONDS,
            "default_num_inference_steps": MOSS_SOUNDEFFECT_DEFAULT_STEPS,
            "default_cfg_scale": MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE,
            "default_sigma_shift": MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT,
            "disable_torchdynamo": MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
        },
        "last_errors": {"soundeffect": manager.last_error},
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request) -> JSONResponse:
    assert_local_request(request)
    return JSONResponse(
        {
            "code": 200,
            "msg": "SoundEffect wrapper 无常驻模型；没有 worker 运行时显存已处于释放状态。",
        }
    )


@app.post("/v1/generate")
@app.post("/v2/synthesize")
async def generate(request: SoundEffectGenerateRequest) -> Response:
    with gpu_runtime_lock("soundeffect/generate"):
        with manager.lock:
            try:
                audio = manager.run_worker(manager.build_worker_payload(request))
                return Response(content=audio, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                # The worker owns all CUDA allocations.  It has either exited
                # normally or been killed before the shared GPU lock is freed.
                wait_after_cuda_release()


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 MOSS-SoundEffect v2")
    print("==================================================")
    print(f"[配置] worker env: {MOSS_SOUNDEFFECT_CONDA_ENV}")
    print(f"[配置] 模型目录: {MOSS_SOUNDEFFECT_MODEL_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] worker 脚本: {WORKER_SCRIPT}")
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, timeout={MOSS_SOUNDEFFECT_REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
