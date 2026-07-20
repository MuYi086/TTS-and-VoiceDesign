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
from typing import Optional, Any

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
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(API_DIR)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_optional_text(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        value = default
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def env_optional_float(name: str, default: Optional[float] = None) -> Optional[float]:
    value = env_optional_text(name)
    if value is None:
        return default
    return float(value)


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", os.path.join(API_DIR, "prompts")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", os.path.join(API_DIR, ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8304"))

OMNIVOICE_CONDA_ENV = os.getenv("OMNIVOICE_CONDA_ENV", "omnivoice")
OMNIVOICE_MODEL_DIR = expand_path(
    os.getenv("OMNIVOICE_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "k2-fsa/OmniVoice"))
)
OMNIVOICE_DEVICE_MAP = os.getenv("OMNIVOICE_DEVICE_MAP", "cuda:0")
OMNIVOICE_DTYPE = os.getenv("OMNIVOICE_DTYPE", "float16")
OMNIVOICE_ATTN_IMPLEMENTATION = os.getenv("OMNIVOICE_ATTN_IMPLEMENTATION", "sdpa")
OMNIVOICE_SDPA_BACKEND = os.getenv("OMNIVOICE_SDPA_BACKEND", "math")
OMNIVOICE_LANGUAGE = env_optional_text("OMNIVOICE_LANGUAGE", "Chinese")
OMNIVOICE_SEED = int(os.getenv("OMNIVOICE_SEED", "42"))
OMNIVOICE_NUM_STEP = int(os.getenv("OMNIVOICE_NUM_STEP", "32"))
OMNIVOICE_GUIDANCE_SCALE = float(os.getenv("OMNIVOICE_GUIDANCE_SCALE", "2.0"))
OMNIVOICE_SPEED = env_optional_float("OMNIVOICE_SPEED", 1.0)
OMNIVOICE_DURATION = env_optional_float("OMNIVOICE_DURATION")
OMNIVOICE_T_SHIFT = float(os.getenv("OMNIVOICE_T_SHIFT", "0.1"))
OMNIVOICE_DENOISE = env_bool("OMNIVOICE_DENOISE", True)
OMNIVOICE_PREPROCESS_PROMPT = env_bool("OMNIVOICE_PREPROCESS_PROMPT", True)
OMNIVOICE_POSTPROCESS_OUTPUT = env_bool("OMNIVOICE_POSTPROCESS_OUTPUT", True)
OMNIVOICE_LAYER_PENALTY_FACTOR = float(os.getenv("OMNIVOICE_LAYER_PENALTY_FACTOR", "5.0"))
OMNIVOICE_POSITION_TEMPERATURE = float(os.getenv("OMNIVOICE_POSITION_TEMPERATURE", "5.0"))
OMNIVOICE_CLASS_TEMPERATURE = float(os.getenv("OMNIVOICE_CLASS_TEMPERATURE", "0.0"))
OMNIVOICE_AUDIO_CHUNK_DURATION = float(os.getenv("OMNIVOICE_AUDIO_CHUNK_DURATION", "15.0"))
OMNIVOICE_AUDIO_CHUNK_THRESHOLD = float(os.getenv("OMNIVOICE_AUDIO_CHUNK_THRESHOLD", "30.0"))
OMNIVOICE_PAD_DURATION = float(os.getenv("OMNIVOICE_PAD_DURATION", "0.1"))
OMNIVOICE_FADE_DURATION = float(os.getenv("OMNIVOICE_FADE_DURATION", "0.1"))
OMNIVOICE_MAX_CHARS_PER_CHUNK = int(os.getenv("OMNIVOICE_MAX_CHARS_PER_CHUNK", "60"))
OMNIVOICE_PAUSE_MS = int(os.getenv("OMNIVOICE_PAUSE_MS", "250"))
OMNIVOICE_REQUEST_TIMEOUT = float(os.getenv("OMNIVOICE_REQUEST_TIMEOUT", "600"))
OMNIVOICE_CUDA_RETRY_COUNT = max(0, int(os.getenv("OMNIVOICE_CUDA_RETRY_COUNT", "1")))
OMNIVOICE_CUDA_RETRY_MAX_CHARS = max(
    1,
    int(os.getenv("OMNIVOICE_CUDA_RETRY_MAX_CHARS", "48")),
)

OMNIVOICE_WORKER_SCRIPT = os.path.join(API_DIR, "omnivoice_tts_worker.py")
OMNIVOICE_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "omnivoice_worker")

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
os.makedirs(OMNIVOICE_WORKER_TMP_DIR, exist_ok=True)
gpu_lock_dir = os.path.dirname(GPU_LOCK_FILE)
if gpu_lock_dir:
    os.makedirs(gpu_lock_dir, exist_ok=True)

app = FastAPI(title="Unitale OmniVoice API")


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
        return "OmniVoice worker 未输出错误信息。"
    return " | ".join(lines[-8:])


def is_retryable_cuda_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cuda driver error",
            "device not ready",
            "cuda error:",
            "cublas_status_",
            "cudnn_status_",
        )
    )


class OmniVoiceSynthesizeRequest(CloneSynthesisRequest):

    text: str
    audio_path: str
    prompt_text: Optional[str] = None
    language: Optional[str] = None
    device_map: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    num_step: Optional[int] = None
    guidance_scale: Optional[float] = None
    speed: Optional[float] = None
    duration: Optional[float] = None
    t_shift: Optional[float] = None
    denoise: Optional[bool] = None
    preprocess_prompt: Optional[bool] = None
    postprocess_output: Optional[bool] = None
    layer_penalty_factor: Optional[float] = None
    position_temperature: Optional[float] = None
    class_temperature: Optional[float] = None
    audio_chunk_duration: Optional[float] = None
    audio_chunk_threshold: Optional[float] = None
    pad_duration: Optional[float] = None
    fade_duration: Optional[float] = None
    max_chars_per_chunk: Optional[int] = None
    pause_ms: Optional[int] = None


class OmniVoiceWorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: OmniVoiceSynthesizeRequest) -> dict:
        ref_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        prompt_text = request.prompt_text.strip() if request.prompt_text and request.prompt_text.strip() else None
        if prompt_text is None:
            prompt_text = load_prompt_text_sidecar(request.audio_path)

        return {
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "ref_text": prompt_text,
            "model_path": OMNIVOICE_MODEL_DIR,
            "device_map": request.device_map or OMNIVOICE_DEVICE_MAP,
            "dtype": request.dtype or OMNIVOICE_DTYPE,
            "attn_implementation": (
                request.attn_implementation or OMNIVOICE_ATTN_IMPLEMENTATION
            ),
            "sdpa_backend": OMNIVOICE_SDPA_BACKEND,
            "language": normalize_optional_text(request.language) if request.language is not None else OMNIVOICE_LANGUAGE,
            "seed": OMNIVOICE_SEED,
            "num_step": request.num_step if request.num_step is not None else OMNIVOICE_NUM_STEP,
            "guidance_scale": request.guidance_scale if request.guidance_scale is not None else OMNIVOICE_GUIDANCE_SCALE,
            "speed": request.speed if request.speed is not None else OMNIVOICE_SPEED,
            "duration": request.duration if request.duration is not None else OMNIVOICE_DURATION,
            "t_shift": request.t_shift if request.t_shift is not None else OMNIVOICE_T_SHIFT,
            "denoise": request.denoise if request.denoise is not None else OMNIVOICE_DENOISE,
            "preprocess_prompt": (
                request.preprocess_prompt
                if request.preprocess_prompt is not None
                else OMNIVOICE_PREPROCESS_PROMPT
            ),
            "postprocess_output": (
                request.postprocess_output
                if request.postprocess_output is not None
                else OMNIVOICE_POSTPROCESS_OUTPUT
            ),
            "layer_penalty_factor": (
                request.layer_penalty_factor
                if request.layer_penalty_factor is not None
                else OMNIVOICE_LAYER_PENALTY_FACTOR
            ),
            "position_temperature": (
                request.position_temperature
                if request.position_temperature is not None
                else OMNIVOICE_POSITION_TEMPERATURE
            ),
            "class_temperature": (
                request.class_temperature
                if request.class_temperature is not None
                else OMNIVOICE_CLASS_TEMPERATURE
            ),
            "audio_chunk_duration": (
                request.audio_chunk_duration
                if request.audio_chunk_duration is not None
                else OMNIVOICE_AUDIO_CHUNK_DURATION
            ),
            "audio_chunk_threshold": (
                request.audio_chunk_threshold
                if request.audio_chunk_threshold is not None
                else OMNIVOICE_AUDIO_CHUNK_THRESHOLD
            ),
            "pad_duration": request.pad_duration if request.pad_duration is not None else OMNIVOICE_PAD_DURATION,
            "fade_duration": request.fade_duration if request.fade_duration is not None else OMNIVOICE_FADE_DURATION,
            "max_chars_per_chunk": (
                request.max_chars_per_chunk
                if request.max_chars_per_chunk is not None
                else OMNIVOICE_MAX_CHARS_PER_CHUNK
            ),
            "pause_ms": request.pause_ms if request.pause_ms is not None else OMNIVOICE_PAUSE_MS,
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def run_worker(self, payload: dict) -> bytes:
        attempt_payload = dict(payload)
        try:
            for attempt in range(OMNIVOICE_CUDA_RETRY_COUNT + 1):
                try:
                    audio_bytes = self._run_worker_once(attempt_payload)
                    self.last_error = None
                    return audio_bytes
                except RuntimeError as exc:
                    if attempt >= OMNIVOICE_CUDA_RETRY_COUNT or not is_retryable_cuda_error(exc):
                        raise

                    retry_number = attempt + 1
                    print(
                        f"[OmniVoice] 检测到可恢复的 CUDA 异常，准备重试 "
                        f"{retry_number}/{OMNIVOICE_CUDA_RETRY_COUNT}: {exc}"
                    )
                    attempt_payload = dict(attempt_payload)
                    attempt_payload["attn_implementation"] = "eager"
                    attempt_payload["sdpa_backend"] = "math"
                    attempt_payload["max_chars_per_chunk"] = min(
                        int(
                            attempt_payload.get("max_chars_per_chunk")
                            or OMNIVOICE_MAX_CHARS_PER_CHUNK
                        ),
                        OMNIVOICE_CUDA_RETRY_MAX_CHARS,
                    )
                    wait_after_cuda_release("before OmniVoice CUDA retry")
        except Exception as exc:
            self.last_error = str(exc)
            raise

        raise RuntimeError("OmniVoice worker 重试流程异常结束。")

    def _run_worker_once(self, payload: dict) -> bytes:
        conda_exe = resolve_conda_executable()
        if not conda_exe:
            raise RuntimeError("未找到 conda 命令，无法调用 OmniVoice worker。")
        if not os.path.isfile(OMNIVOICE_WORKER_SCRIPT):
            raise RuntimeError(f"OmniVoice worker 脚本不存在: {OMNIVOICE_WORKER_SCRIPT}")
        if not os.path.isdir(OMNIVOICE_MODEL_DIR):
            raise RuntimeError(f"OmniVoice 模型目录不存在: {OMNIVOICE_MODEL_DIR}")

        request_fd, request_path = tempfile.mkstemp(
            dir=OMNIVOICE_WORKER_TMP_DIR,
            prefix="omnivoice_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=OMNIVOICE_WORKER_TMP_DIR,
            prefix="omnivoice_out_",
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
                OMNIVOICE_CONDA_ENV,
                "python",
                OMNIVOICE_WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[OmniVoice] 启动 worker: env={OMNIVOICE_CONDA_ENV}")
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
                stdout, stderr = proc.communicate(timeout=OMNIVOICE_REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                stdout, stderr = proc.communicate(timeout=10)
                raise RuntimeError(f"OmniVoice worker 超时（>{OMNIVOICE_REQUEST_TIMEOUT:.0f}s）")

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[OmniVoice] worker 退出码={proc.returncode}，耗时 {elapsed:.2f}s")

            if proc.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("OmniVoice worker 未生成音频文件。")

            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes
        finally:
            for path in (request_path, output_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass


manager = OmniVoiceWorkerManager()


@app.get("/v1/health")
async def health():
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "omnivoice_model_dir": OMNIVOICE_MODEL_DIR,
            "prompts_dir": PROMPTS_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": OMNIVOICE_WORKER_SCRIPT,
            "worker_tmp_dir": OMNIVOICE_WORKER_TMP_DIR,
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "worker_script": os.path.isfile(OMNIVOICE_WORKER_SCRIPT),
            "omnivoice_model_dir": os.path.isdir(OMNIVOICE_MODEL_DIR),
            "torch": module_available("torch"),
            "cuda": cuda_status()["available"],
        },
        "cuda": cuda_status(),
        "runtime": {
            "worker_env": OMNIVOICE_CONDA_ENV,
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": OMNIVOICE_REQUEST_TIMEOUT,
            "device_map": OMNIVOICE_DEVICE_MAP,
            "dtype": OMNIVOICE_DTYPE,
            "attn_implementation": OMNIVOICE_ATTN_IMPLEMENTATION,
            "sdpa_backend": OMNIVOICE_SDPA_BACKEND,
            "language": OMNIVOICE_LANGUAGE,
            "seed": OMNIVOICE_SEED,
            "num_step": OMNIVOICE_NUM_STEP,
            "guidance_scale": OMNIVOICE_GUIDANCE_SCALE,
            "speed": OMNIVOICE_SPEED,
            "duration": OMNIVOICE_DURATION,
            "t_shift": OMNIVOICE_T_SHIFT,
            "denoise": OMNIVOICE_DENOISE,
            "preprocess_prompt": OMNIVOICE_PREPROCESS_PROMPT,
            "postprocess_output": OMNIVOICE_POSTPROCESS_OUTPUT,
            "layer_penalty_factor": OMNIVOICE_LAYER_PENALTY_FACTOR,
            "position_temperature": OMNIVOICE_POSITION_TEMPERATURE,
            "class_temperature": OMNIVOICE_CLASS_TEMPERATURE,
            "audio_chunk_duration": OMNIVOICE_AUDIO_CHUNK_DURATION,
            "audio_chunk_threshold": OMNIVOICE_AUDIO_CHUNK_THRESHOLD,
            "pad_duration": OMNIVOICE_PAD_DURATION,
            "fade_duration": OMNIVOICE_FADE_DURATION,
            "max_chars_per_chunk": OMNIVOICE_MAX_CHARS_PER_CHUNK,
            "pause_ms": OMNIVOICE_PAUSE_MS,
            "cuda_retry_count": OMNIVOICE_CUDA_RETRY_COUNT,
            "cuda_retry_max_chars": OMNIVOICE_CUDA_RETRY_MAX_CHARS,
            "prompt_text_fallback": "upload sidecar -> OmniVoice internal ASR",
        },
        "last_errors": {
            "omnivoice_tts": manager.last_error,
        },
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    clear_cuda_cache("omnivoice api internal unload")
    wait_after_cuda_release("omnivoice api internal unload")
    return JSONResponse({"code": 200, "msg": "omnivoice wrapper 无常驻模型，已完成显存清理等待"})


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
async def synthesize_v2(request: OmniVoiceSynthesizeRequest):
    with gpu_runtime_lock("omnivoice/synthesize"):
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
                clear_cuda_cache("after omnivoice worker")
                wait_after_cuda_release("after omnivoice worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 OmniVoice Voice Clone")
    print("==================================================")
    print(f"[配置] OmniVoice worker env: {OMNIVOICE_CONDA_ENV}")
    print(f"[配置] OmniVoice 模型目录: {OMNIVOICE_MODEL_DIR}")
    print(f"[配置] prompts 目录: {PROMPTS_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] worker 脚本: {OMNIVOICE_WORKER_SCRIPT}")
    print(
        f"[配置] device_map={OMNIVOICE_DEVICE_MAP}, dtype={OMNIVOICE_DTYPE}, "
        f"attn_implementation={OMNIVOICE_ATTN_IMPLEMENTATION}, "
        f"sdpa_backend={OMNIVOICE_SDPA_BACKEND}, "
        f"language={OMNIVOICE_LANGUAGE or 'auto'}, num_step={OMNIVOICE_NUM_STEP}, "
        f"guidance_scale={OMNIVOICE_GUIDANCE_SCALE}"
    )
    print(
        f"[配置] speed={OMNIVOICE_SPEED}, duration={OMNIVOICE_DURATION}, "
        f"max_chars_per_chunk={OMNIVOICE_MAX_CHARS_PER_CHUNK}, pause_ms={OMNIVOICE_PAUSE_MS}"
    )
    print(
        f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={OMNIVOICE_REQUEST_TIMEOUT}, "
        f"cuda_retry={OMNIVOICE_CUDA_RETRY_COUNT}, "
        f"retry_max_chars={OMNIVOICE_CUDA_RETRY_MAX_CHARS}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
