import fcntl
import gc
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Optional

# Align VoxCPM2 with the standalone step_3 script instead of inheriting
# global CUDA runtime tweaks that break this model's GPU path.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.pop("CUDA_MODULE_LOADING", None)

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from synthesis_request import CloneSynthesisRequest

API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(API_DIR)


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


def normalize_device_name(value: Optional[str], default: str = "cuda") -> str:
    normalized = normalize_optional_text(value)
    return normalized.lower() if normalized is not None else default


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", os.path.join(API_DIR, "prompts")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", os.path.join(API_DIR, ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8306"))

VOXCPM2_CONDA_ENV = os.getenv("VOXCPM2_CONDA_ENV", "voxcpm2")
VOXCPM2_MODEL_DIR = expand_path(
    os.getenv("VOXCPM2_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "openbmb/VoxCPM2"))
)
VOXCPM2_HELPER_DEFAULT = expand_path(
    os.path.join("~", "github", "timbre-design", "modelScript", "tts_local_voxcpm2.py")
)
VOXCPM2_HELPER_LEGACY_PATH = expand_path(
    os.path.join("~", "github", "timbre-design", "scripts", "tts_local_voxcpm2.py")
)
VOXCPM2_HELPER_SCRIPT = expand_path(os.getenv("VOXCPM2_HELPER_SCRIPT", VOXCPM2_HELPER_DEFAULT))
# The helper was moved from scripts/ to modelScript/.  Repair the stale path
# emitted by earlier start.sh versions while preserving any other user override.
if VOXCPM2_HELPER_SCRIPT == VOXCPM2_HELPER_LEGACY_PATH and os.path.isfile(VOXCPM2_HELPER_DEFAULT):
    VOXCPM2_HELPER_SCRIPT = VOXCPM2_HELPER_DEFAULT
VOXCPM2_CFG_VALUE = float(os.getenv("VOXCPM2_CFG_VALUE", "2.0"))
VOXCPM2_INFERENCE_TIMESTEPS = int(os.getenv("VOXCPM2_INFERENCE_TIMESTEPS", "10"))
VOXCPM2_LOAD_DENOISER = env_bool("VOXCPM2_LOAD_DENOISER", False)
VOXCPM2_OPTIMIZE = env_bool("VOXCPM2_OPTIMIZE", False)
VOXCPM2_DEVICE = normalize_device_name(os.getenv("VOXCPM2_DEVICE"), "cuda")
VOXCPM2_SEED = int(os.getenv("VOXCPM2_SEED", "20260614"))
VOXCPM2_MAX_CHARS_PER_CHUNK = int(os.getenv("VOXCPM2_MAX_CHARS_PER_CHUNK", "0"))
VOXCPM2_PAUSE_MS = int(os.getenv("VOXCPM2_PAUSE_MS", "250"))
VOXCPM2_REQUEST_TIMEOUT = float(os.getenv("VOXCPM2_REQUEST_TIMEOUT", "600"))

VOXCPM2_WORKER_SCRIPT = os.path.join(API_DIR, "voxcpm2_worker.py")
VOXCPM2_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "voxcpm2_worker")

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
os.makedirs(VOXCPM2_WORKER_TMP_DIR, exist_ok=True)
gpu_lock_dir = os.path.dirname(GPU_LOCK_FILE)
if gpu_lock_dir:
    os.makedirs(gpu_lock_dir, exist_ok=True)

app = FastAPI(title="Unitale VoxCPM2 Voice Clone API")


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
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{h}{ext}"


def prompt_text_sidecar_path(filename: str) -> str:
    return os.path.join(PROMPTS_DIR, f"{hash_filename(filename)}.prompt.txt")


def load_prompt_text_sidecar(filename: str) -> Optional[str]:
    sidecar_path = prompt_text_sidecar_path(filename)
    if not os.path.isfile(sidecar_path):
        return None
    with open(sidecar_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text or None


def save_prompt_text_sidecar(filename: str, prompt_text: Optional[str]) -> None:
    sidecar_path = prompt_text_sidecar_path(filename)
    normalized = prompt_text.strip() if prompt_text and prompt_text.strip() else None
    if normalized is None:
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
        return
    with open(sidecar_path, "w", encoding="utf-8") as f:
        f.write(normalized)


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def resolve_conda_executable() -> Optional[str]:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and os.path.isfile(expand_path(conda_exe)):
        return expand_path(conda_exe)
    return shutil.which("conda")


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


def clear_cuda_cache(label: str = "") -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return

    prefix = f"[CUDA] {label}: " if label else "[CUDA] "
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"{prefix}synchronize 跳过: {exc}")
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"{prefix}cache cleanup 跳过: {exc}")


def wait_after_cuda_release(label: str = "") -> None:
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


def cuda_status() -> dict:
    status = {"available": False}
    try:
        status["available"] = torch.cuda.is_available()
        if not status["available"]:
            return status

        status["device_count"] = torch.cuda.device_count()
        status["device_name"] = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        status["memory"] = {
            "free_mib": round(free_bytes / 1024 / 1024, 1),
            "total_mib": round(total_bytes / 1024 / 1024, 1),
            "allocated_mib": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
            "reserved_mib": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
        }
    except Exception as exc:
        status["error"] = str(exc)
    return status


def normalize_synthesis_text(text: str) -> str:
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise ValueError("text 不能为空。")
    return normalized


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "VoxCPM2 worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class VoxCpm2SynthesizeRequest(CloneSynthesisRequest):

    text: str
    audio_path: str
    prompt_text: Optional[str] = None
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None
    load_denoiser: Optional[bool] = None
    optimize: Optional[bool] = None
    device: Optional[str] = None
    seed: Optional[int] = None
    max_chars_per_chunk: Optional[int] = None
    pause_ms: Optional[int] = None


class VoxCpm2WorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: VoxCpm2SynthesizeRequest) -> dict:
        ref_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        prompt_text = request.prompt_text.strip() if request.prompt_text and request.prompt_text.strip() else None
        if prompt_text is None:
            prompt_text = load_prompt_text_sidecar(request.audio_path)
        device = normalize_device_name(request.device, VOXCPM2_DEVICE)
        if not device.startswith("cuda"):
            raise HTTPException(status_code=400, detail=f"VoxCPM2 仅支持 GPU 设备，当前 device={device}")

        return {
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "model_path": VOXCPM2_MODEL_DIR,
            "voxcpm2_helper_script": VOXCPM2_HELPER_SCRIPT,
            "cfg_value": request.cfg_value if request.cfg_value is not None else VOXCPM2_CFG_VALUE,
            "inference_timesteps": (
                request.inference_timesteps
                if request.inference_timesteps is not None
                else VOXCPM2_INFERENCE_TIMESTEPS
            ),
            "load_denoiser": (
                request.load_denoiser if request.load_denoiser is not None else VOXCPM2_LOAD_DENOISER
            ),
            "optimize": request.optimize if request.optimize is not None else VOXCPM2_OPTIMIZE,
            "device": device,
            "seed": request.seed if request.seed is not None else VOXCPM2_SEED,
            "max_chars_per_chunk": (
                request.max_chars_per_chunk
                if request.max_chars_per_chunk is not None
                else VOXCPM2_MAX_CHARS_PER_CHUNK
            ),
            "pause_ms": request.pause_ms if request.pause_ms is not None else VOXCPM2_PAUSE_MS,
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def _run_worker_once(self, payload: dict) -> bytes:
        conda_exe = resolve_conda_executable()
        if not conda_exe:
            raise RuntimeError("未找到 conda 命令，无法调用 VoxCPM2 worker。")
        if not os.path.isfile(VOXCPM2_WORKER_SCRIPT):
            raise RuntimeError(f"VoxCPM2 worker 脚本不存在: {VOXCPM2_WORKER_SCRIPT}")
        if not os.path.isdir(VOXCPM2_MODEL_DIR):
            raise RuntimeError(f"VoxCPM2 模型目录不存在: {VOXCPM2_MODEL_DIR}")
        if not os.path.isfile(VOXCPM2_HELPER_SCRIPT):
            raise RuntimeError(f"VoxCPM2 辅助脚本不存在: {VOXCPM2_HELPER_SCRIPT}")

        request_fd, request_path = tempfile.mkstemp(
            dir=VOXCPM2_WORKER_TMP_DIR,
            prefix="voxcpm2_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=VOXCPM2_WORKER_TMP_DIR,
            prefix="voxcpm2_out_",
            suffix=".wav",
        )
        os.close(request_fd)
        os.close(output_fd)

        try:
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            command = [
                conda_exe,
                "run",
                "--no-capture-output",
                "-n",
                VOXCPM2_CONDA_ENV,
                "python",
                VOXCPM2_WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[VoxCPM2] 启动 worker: env={VOXCPM2_CONDA_ENV}")
            started = time.perf_counter()
            worker_env = os.environ.copy()
            worker_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            worker_env.pop("CUDA_MODULE_LOADING", None)
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=worker_env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=VOXCPM2_REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                stdout, stderr = proc.communicate(timeout=10)
                raise RuntimeError(f"VoxCPM2 worker 超时（>{VOXCPM2_REQUEST_TIMEOUT:.0f}s）")

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[VoxCPM2] worker 退出码={proc.returncode}，耗时 {elapsed:.2f}s")

            if proc.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("VoxCPM2 worker 未生成音频文件。")

            with open(output_path, "rb") as f:
                return f.read()
        finally:
            for path in (request_path, output_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

    def run_worker(self, payload: dict) -> bytes:
        try:
            audio_bytes = self._run_worker_once(payload)
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise


manager = VoxCpm2WorkerManager()


@app.get("/v1/health")
async def health():
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "voxcpm2_model_dir": VOXCPM2_MODEL_DIR,
            "voxcpm2_helper_script": VOXCPM2_HELPER_SCRIPT,
            "prompts_dir": PROMPTS_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": VOXCPM2_WORKER_SCRIPT,
            "worker_tmp_dir": VOXCPM2_WORKER_TMP_DIR,
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "worker_script": os.path.isfile(VOXCPM2_WORKER_SCRIPT),
            "voxcpm2_model_dir": os.path.isdir(VOXCPM2_MODEL_DIR),
            "voxcpm2_helper_script": os.path.isfile(VOXCPM2_HELPER_SCRIPT),
            "torch": module_available("torch"),
            "cuda": cuda_status()["available"],
        },
        "cuda": cuda_status(),
        "runtime": {
            "worker_env": VOXCPM2_CONDA_ENV,
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": VOXCPM2_REQUEST_TIMEOUT,
            "cfg_value": VOXCPM2_CFG_VALUE,
            "inference_timesteps": VOXCPM2_INFERENCE_TIMESTEPS,
            "load_denoiser": VOXCPM2_LOAD_DENOISER,
            "optimize": VOXCPM2_OPTIMIZE,
            "device": VOXCPM2_DEVICE,
            "seed": VOXCPM2_SEED,
            "max_chars_per_chunk": VOXCPM2_MAX_CHARS_PER_CHUNK,
            "pause_ms": VOXCPM2_PAUSE_MS,
            "prompt_text_fallback": "upload sidecar -> reference-only cloning mode",
        },
        "last_errors": {
            "voxcpm2_tts": manager.last_error,
        },
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    clear_cuda_cache("voxcpm2 api internal unload")
    wait_after_cuda_release("voxcpm2 api internal unload")
    return JSONResponse({"code": 200, "msg": "voxcpm2 wrapper 无常驻模型，已完成显存清理等待"})


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...),
    prompt_text: Optional[str] = Form(None),
):
    content = await audio.read()
    save_path = os.path.join(PROMPTS_DIR, hash_filename(full_path))
    with open(save_path, "wb") as f:
        f.write(content)

    normalized_prompt_text = prompt_text.strip() if prompt_text and prompt_text.strip() else None
    save_prompt_text_sidecar(full_path, normalized_prompt_text)

    return {
        "code": 200,
        "msg": "上传成功",
        "filename": full_path,
        "has_prompt_text": bool(normalized_prompt_text),
    }


@app.get("/v1/check/audio")
async def check_audio_exists(file_name: str):
    exists = os.path.isfile(os.path.join(PROMPTS_DIR, hash_filename(file_name)))
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
    }


@app.post("/v2/synthesize")
async def synthesize_v2(request: VoxCpm2SynthesizeRequest):
    with gpu_runtime_lock("voxcpm2/synthesize"):
        with manager.lock:
            try:
                payload = manager.build_worker_payload(request)
                audio_bytes = manager.run_worker(payload)
                return Response(content=audio_bytes, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as exc:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(exc))
            finally:
                clear_cuda_cache("after voxcpm2 worker")
                wait_after_cuda_release("after voxcpm2 worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 VoxCPM2 Voice Clone")
    print("==================================================")
    print(f"[配置] VoxCPM2 worker env: {VOXCPM2_CONDA_ENV}")
    print(f"[配置] VoxCPM2 模型目录: {VOXCPM2_MODEL_DIR}")
    print(f"[配置] VoxCPM2 helper: {VOXCPM2_HELPER_SCRIPT}")
    print(f"[配置] prompts 目录: {PROMPTS_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] worker 脚本: {VOXCPM2_WORKER_SCRIPT}")
    print(
        f"[配置] cfg_value={VOXCPM2_CFG_VALUE}, inference_timesteps={VOXCPM2_INFERENCE_TIMESTEPS}, "
        f"seed={VOXCPM2_SEED}"
    )
    print(
        f"[配置] load_denoiser={VOXCPM2_LOAD_DENOISER}, optimize={VOXCPM2_OPTIMIZE}, "
        f"max_chars_per_chunk={VOXCPM2_MAX_CHARS_PER_CHUNK}, pause_ms={VOXCPM2_PAUSE_MS}"
    )
    print(
        f"[配置] device={VOXCPM2_DEVICE}"
    )
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={VOXCPM2_REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
