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
from typing import Optional, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from synthesis_request import CloneSynthesisRequest

# ==========================================
# 0. 系统配置
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", os.path.join(PROJECT_DIR, "prompts")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", os.path.join(PROJECT_DIR, ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8301"))

DOTS_CONDA_ENV = os.getenv("DOTS_CONDA_ENV", "dots_tts")
DOTS_MODEL_DIR = expand_path(
    os.getenv("DOTS_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "rednote-hilab/dots.tts-base"))
)
DOTS_LANGUAGE = os.getenv("DOTS_LANGUAGE", "chinese")
DOTS_TEMPLATE_NAME = os.getenv("DOTS_TEMPLATE_NAME") or None
DOTS_PRECISION = os.getenv("DOTS_PRECISION", "bfloat16")
DOTS_SEED = int(os.getenv("DOTS_SEED", "42"))
DOTS_ODE_METHOD = os.getenv("DOTS_ODE_METHOD", "euler")
DOTS_NUM_STEPS = int(os.getenv("DOTS_NUM_STEPS", "10"))
DOTS_GUIDANCE_SCALE = float(os.getenv("DOTS_GUIDANCE_SCALE", "1.2"))
DOTS_SPEAKER_SCALE = float(os.getenv("DOTS_SPEAKER_SCALE", "1.5"))
DOTS_MAX_GENERATE_LENGTH = int(os.getenv("DOTS_MAX_GENERATE_LENGTH", "500"))
DOTS_MAX_CHARS_PER_CHUNK = int(os.getenv("DOTS_MAX_CHARS_PER_CHUNK", "120"))
DOTS_PAUSE_MS = int(os.getenv("DOTS_PAUSE_MS", "250"))
DOTS_NORMALIZE_TEXT = env_bool("DOTS_NORMALIZE_TEXT", False)
DOTS_PROFILE_INFERENCE = env_bool("DOTS_PROFILE_INFERENCE", False)
DOTS_REQUEST_TIMEOUT = float(os.getenv("DOTS_REQUEST_TIMEOUT", "300"))

DOTS_WORKER_SCRIPT = os.path.join(PROJECT_DIR, "dots_tts_worker.py")
DOTS_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "dots_worker")

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
os.makedirs(DOTS_WORKER_TMP_DIR, exist_ok=True)
gpu_lock_dir = os.path.dirname(GPU_LOCK_FILE)
if gpu_lock_dir:
    os.makedirs(gpu_lock_dir, exist_ok=True)

app = FastAPI(title="Unitale Dots TTS API")


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
    if not normalized:
        raise ValueError("text 不能为空。")
    return normalized


def normalize_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() == "none":
        return None
    return normalized


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "dots.tts worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class DotsSynthesizeRequest(CloneSynthesisRequest):

    text: str
    audio_path: str
    prompt_text: Optional[str] = None
    language: Optional[str] = None
    template_name: Optional[str] = None
    ode_method: Optional[str] = None
    num_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    speaker_scale: Optional[float] = None
    max_chars_per_chunk: Optional[int] = None
    pause_ms: Optional[int] = None
    normalize_text: Optional[bool] = None
    profile_inference: Optional[bool] = None
    emo_text: Optional[str] = None
    emo_vector: Optional[List[float]] = None


class DotsWorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: DotsSynthesizeRequest) -> dict:
        ref_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        prompt_text = request.prompt_text.strip() if request.prompt_text and request.prompt_text.strip() else None
        if prompt_text is None:
            prompt_text = load_prompt_text_sidecar(request.audio_path)

        return {
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "model_path": DOTS_MODEL_DIR,
            "language": normalize_language(request.language) if request.language is not None else normalize_language(DOTS_LANGUAGE),
            "template_name": request.template_name if request.template_name else DOTS_TEMPLATE_NAME,
            "precision": DOTS_PRECISION,
            "seed": DOTS_SEED,
            "ode_method": request.ode_method or DOTS_ODE_METHOD,
            "num_steps": request.num_steps if request.num_steps is not None else DOTS_NUM_STEPS,
            "guidance_scale": request.guidance_scale if request.guidance_scale is not None else DOTS_GUIDANCE_SCALE,
            "speaker_scale": request.speaker_scale if request.speaker_scale is not None else DOTS_SPEAKER_SCALE,
            "max_generate_length": DOTS_MAX_GENERATE_LENGTH,
            "max_chars_per_chunk": request.max_chars_per_chunk if request.max_chars_per_chunk is not None else DOTS_MAX_CHARS_PER_CHUNK,
            "pause_ms": request.pause_ms if request.pause_ms is not None else DOTS_PAUSE_MS,
            "normalize_text": request.normalize_text if request.normalize_text is not None else DOTS_NORMALIZE_TEXT,
            "profile_inference": request.profile_inference if request.profile_inference is not None else DOTS_PROFILE_INFERENCE,
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def run_worker(self, payload: dict) -> bytes:
        conda_exe = resolve_conda_executable()
        if not conda_exe:
            raise RuntimeError("未找到 conda 命令，无法调用 dots_tts worker。")
        if not os.path.isfile(DOTS_WORKER_SCRIPT):
            raise RuntimeError(f"dots_tts worker 脚本不存在: {DOTS_WORKER_SCRIPT}")
        if not os.path.isdir(DOTS_MODEL_DIR):
            raise RuntimeError(f"dots.tts 模型目录不存在: {DOTS_MODEL_DIR}")

        request_fd, request_path = tempfile.mkstemp(dir=DOTS_WORKER_TMP_DIR, prefix="dots_req_", suffix=".json")
        output_fd, output_path = tempfile.mkstemp(dir=DOTS_WORKER_TMP_DIR, prefix="dots_out_", suffix=".wav")
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
                DOTS_CONDA_ENV,
                "python",
                DOTS_WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[dots.tts] 启动 worker: env={DOTS_CONDA_ENV}")
            started = time.perf_counter()
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=os.environ.copy(),
            )
            try:
                stdout, stderr = proc.communicate(timeout=DOTS_REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                stdout, stderr = proc.communicate(timeout=10)
                raise RuntimeError(f"dots.tts worker 超时（>{DOTS_REQUEST_TIMEOUT:.0f}s）")

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[dots.tts] worker 退出码={proc.returncode}，耗时 {elapsed:.2f}s")

            if proc.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("dots.tts worker 未生成音频文件。")

            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            for path in (request_path, output_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass


manager = DotsWorkerManager()


@app.get("/v1/health")
async def health():
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "dots_model_dir": DOTS_MODEL_DIR,
            "prompts_dir": PROMPTS_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": DOTS_WORKER_SCRIPT,
            "worker_tmp_dir": DOTS_WORKER_TMP_DIR,
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "worker_script": os.path.isfile(DOTS_WORKER_SCRIPT),
            "dots_model_dir": os.path.isdir(DOTS_MODEL_DIR),
            "torch": module_available("torch"),
            "cuda": cuda_status()["available"],
        },
        "cuda": cuda_status(),
        "runtime": {
            "worker_env": DOTS_CONDA_ENV,
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": DOTS_REQUEST_TIMEOUT,
            "language": DOTS_LANGUAGE,
            "template_name": DOTS_TEMPLATE_NAME,
            "precision": DOTS_PRECISION,
            "num_steps": DOTS_NUM_STEPS,
            "guidance_scale": DOTS_GUIDANCE_SCALE,
            "speaker_scale": DOTS_SPEAKER_SCALE,
            "max_generate_length": DOTS_MAX_GENERATE_LENGTH,
            "max_chars_per_chunk": DOTS_MAX_CHARS_PER_CHUNK,
            "pause_ms": DOTS_PAUSE_MS,
            "normalize_text": DOTS_NORMALIZE_TEXT,
            "profile_inference": DOTS_PROFILE_INFERENCE,
        },
        "last_errors": {
            "dots_tts": manager.last_error,
        },
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    clear_cuda_cache("dots api internal unload")
    wait_after_cuda_release("dots api internal unload")
    return JSONResponse({"code": 200, "msg": "dots_tts wrapper 无常驻模型，已完成显存清理等待"})


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

    sidecar_path = prompt_text_sidecar_path(full_path)
    if prompt_text and prompt_text.strip():
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(prompt_text.strip())
    elif os.path.exists(sidecar_path):
        os.remove(sidecar_path)

    return {
        "code": 200,
        "msg": "上传成功",
        "filename": full_path,
        "has_prompt_text": bool(prompt_text and prompt_text.strip()),
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
async def synthesize_v2(request: DotsSynthesizeRequest):
    with gpu_runtime_lock("dots_tts/synthesize"):
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
                clear_cuda_cache("after dots_tts worker")
                wait_after_cuda_release("after dots_tts worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 dots.tts Voice Clone")
    print("==================================================")
    print(f"[配置] dots worker env: {DOTS_CONDA_ENV}")
    print(f"[配置] dots 模型目录: {DOTS_MODEL_DIR}")
    print(f"[配置] prompts 目录: {PROMPTS_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] worker 脚本: {DOTS_WORKER_SCRIPT}")
    print(
        f"[配置] language={DOTS_LANGUAGE}, precision={DOTS_PRECISION}, "
        f"num_steps={DOTS_NUM_STEPS}, guidance_scale={DOTS_GUIDANCE_SCALE}, "
        f"speaker_scale={DOTS_SPEAKER_SCALE}"
    )
    print(
        f"[配置] max_generate_length={DOTS_MAX_GENERATE_LENGTH}, "
        f"max_chars_per_chunk={DOTS_MAX_CHARS_PER_CHUNK}, pause_ms={DOTS_PAUSE_MS}"
    )
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={DOTS_REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
