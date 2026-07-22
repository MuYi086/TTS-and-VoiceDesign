import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from synthesis_request import CloneSynthesisRequest
from gpu_runtime import cuda_status, terminate_process_group

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


def env_optional_float(name: str) -> Optional[float]:
    value = env_optional_text(name)
    return float(value) if value is not None else None


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
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
API_PORT = int(os.getenv("PORT", "8305"))

QWEN3_TTS_CONDA_ENV = os.getenv("QWEN3_TTS_CONDA_ENV", "qwen3-tts")
QWEN3_TTS_MODEL_DIR = expand_path(
    os.getenv("QWEN3_TTS_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
)
QWEN3_TTS_DEVICE_MAP = os.getenv("QWEN3_TTS_DEVICE_MAP", "cuda:0")
QWEN3_TTS_DTYPE = os.getenv("QWEN3_TTS_DTYPE", "auto")
QWEN3_TTS_LANGUAGE = env_optional_text("QWEN3_TTS_LANGUAGE", "Chinese")
QWEN3_TTS_MAX_NEW_TOKENS = int(os.getenv("QWEN3_TTS_MAX_NEW_TOKENS", "2048"))
QWEN3_TTS_TOP_P = env_optional_float("QWEN3_TTS_TOP_P")
QWEN3_TTS_TEMPERATURE = env_optional_float("QWEN3_TTS_TEMPERATURE")
QWEN3_TTS_ATTN_IMPLEMENTATION = os.getenv("QWEN3_TTS_ATTN_IMPLEMENTATION", "auto")
QWEN3_TTS_X_VECTOR_ONLY = env_bool("QWEN3_TTS_X_VECTOR_ONLY", False)
QWEN3_TTS_MAX_CHARS_PER_CHUNK = int(os.getenv("QWEN3_TTS_MAX_CHARS_PER_CHUNK", "120"))
QWEN3_TTS_PAUSE_MS = int(os.getenv("QWEN3_TTS_PAUSE_MS", "250"))
QWEN3_TTS_TRIM_LEADING_SILENCE = env_bool("QWEN3_TTS_TRIM_LEADING_SILENCE", True)
QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB = float(
    os.getenv("QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB", "-42")
)
QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS = int(os.getenv("QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS", "120"))
QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS = int(
    os.getenv("QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS", "30")
)
QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS = int(
    os.getenv("QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS", "40")
)
QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS = int(os.getenv("QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS", "8000"))
QWEN3_TTS_REQUEST_TIMEOUT = float(os.getenv("QWEN3_TTS_REQUEST_TIMEOUT", "600"))
QWEN3_TTS_WORKER_SCRIPT = os.path.join(API_DIR, "qwen3_tts_worker.py")
QWEN3_TTS_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "qwen3_tts_worker")
QWEN3_TTS_USE_QWEN_LIBS = env_bool("QWEN3_TTS_USE_QWEN_LIBS", False)
QWEN_LIBS_PATH = expand_path(os.getenv("QWEN_LIBS", os.path.join(API_DIR, "vendor/qwen_libs")))

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
os.makedirs(QWEN3_TTS_WORKER_TMP_DIR, exist_ok=True)
gpu_lock_dir = os.path.dirname(GPU_LOCK_FILE)
if gpu_lock_dir:
    os.makedirs(gpu_lock_dir, exist_ok=True)

app = FastAPI(title="Unitale Qwen3-TTS Voice Clone API")


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


def wait_after_cuda_release(label: str = "") -> None:
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


def normalize_synthesis_text(text: str) -> str:
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise ValueError("text 不能为空。")
    return normalized


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "Qwen3-TTS worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class Qwen3TtsSynthesizeRequest(CloneSynthesisRequest):

    text: str
    audio_path: str
    prompt_text: Optional[str] = None
    language: Optional[str] = None
    x_vector_only: Optional[bool] = None
    device_map: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    max_chars_per_chunk: Optional[int] = None
    pause_ms: Optional[int] = None
    trim_leading_silence: Optional[bool] = None
    trim_leading_silence_threshold_db: Optional[float] = None
    trim_leading_silence_min_ms: Optional[int] = None
    trim_leading_silence_analysis_window_ms: Optional[int] = None
    trim_leading_silence_pre_roll_ms: Optional[int] = None
    trim_leading_silence_max_ms: Optional[int] = None


class Qwen3TtsWorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: Qwen3TtsSynthesizeRequest) -> dict:
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
            "model_path": QWEN3_TTS_MODEL_DIR,
            "language": normalize_optional_text(request.language) if request.language is not None else QWEN3_TTS_LANGUAGE,
            "x_vector_only": (
                request.x_vector_only if request.x_vector_only is not None else QWEN3_TTS_X_VECTOR_ONLY
            ),
            "device_map": request.device_map or QWEN3_TTS_DEVICE_MAP,
            "dtype": request.dtype or QWEN3_TTS_DTYPE,
            "attn_implementation": request.attn_implementation or QWEN3_TTS_ATTN_IMPLEMENTATION,
            "max_new_tokens": (
                request.max_new_tokens if request.max_new_tokens is not None else QWEN3_TTS_MAX_NEW_TOKENS
            ),
            "top_p": request.top_p if request.top_p is not None else QWEN3_TTS_TOP_P,
            "temperature": request.temperature if request.temperature is not None else QWEN3_TTS_TEMPERATURE,
            "max_chars_per_chunk": (
                request.max_chars_per_chunk
                if request.max_chars_per_chunk is not None
                else QWEN3_TTS_MAX_CHARS_PER_CHUNK
            ),
            "pause_ms": request.pause_ms if request.pause_ms is not None else QWEN3_TTS_PAUSE_MS,
            "trim_leading_silence": (
                request.trim_leading_silence
                if request.trim_leading_silence is not None
                else QWEN3_TTS_TRIM_LEADING_SILENCE
            ),
            "trim_leading_silence_threshold_db": (
                request.trim_leading_silence_threshold_db
                if request.trim_leading_silence_threshold_db is not None
                else QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB
            ),
            "trim_leading_silence_min_ms": (
                request.trim_leading_silence_min_ms
                if request.trim_leading_silence_min_ms is not None
                else QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS
            ),
            "trim_leading_silence_analysis_window_ms": (
                request.trim_leading_silence_analysis_window_ms
                if request.trim_leading_silence_analysis_window_ms is not None
                else QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS
            ),
            "trim_leading_silence_pre_roll_ms": (
                request.trim_leading_silence_pre_roll_ms
                if request.trim_leading_silence_pre_roll_ms is not None
                else QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS
            ),
            "trim_leading_silence_max_ms": (
                request.trim_leading_silence_max_ms
                if request.trim_leading_silence_max_ms is not None
                else QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS
            ),
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
            "qwen_libs_path": (
                QWEN_LIBS_PATH
                if QWEN3_TTS_USE_QWEN_LIBS and os.path.isdir(QWEN_LIBS_PATH)
                else None
            ),
        }

    def run_worker(self, payload: dict) -> bytes:
        conda_exe = resolve_conda_executable()
        if not conda_exe:
            raise RuntimeError("未找到 conda 命令，无法调用 Qwen3-TTS worker。")
        if not os.path.isfile(QWEN3_TTS_WORKER_SCRIPT):
            raise RuntimeError(f"Qwen3-TTS worker 脚本不存在: {QWEN3_TTS_WORKER_SCRIPT}")
        if not os.path.isdir(QWEN3_TTS_MODEL_DIR):
            raise RuntimeError(f"Qwen3-TTS 模型目录不存在: {QWEN3_TTS_MODEL_DIR}")

        request_fd, request_path = tempfile.mkstemp(
            dir=QWEN3_TTS_WORKER_TMP_DIR,
            prefix="qwen3_tts_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=QWEN3_TTS_WORKER_TMP_DIR,
            prefix="qwen3_tts_out_",
            suffix=".wav",
        )
        os.close(request_fd)
        os.close(output_fd)
        proc: Optional[subprocess.Popen] = None

        try:
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            command = [
                conda_exe,
                "run",
                "--no-capture-output",
                "-n",
                QWEN3_TTS_CONDA_ENV,
                "python",
                QWEN3_TTS_WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[Qwen3-TTS] 启动 worker: env={QWEN3_TTS_CONDA_ENV}")
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
                stdout, stderr = proc.communicate(timeout=QWEN3_TTS_REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                terminate_process_group(proc, "Qwen3-TTS")
                stdout, stderr = proc.communicate()
                raise RuntimeError(f"Qwen3-TTS worker 超时（>{QWEN3_TTS_REQUEST_TIMEOUT:.0f}s）")

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[Qwen3-TTS] worker 退出码={proc.returncode}，耗时 {elapsed:.2f}s")

            if proc.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Qwen3-TTS worker 未生成音频文件。")

            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(proc, "Qwen3-TTS")
            for path in (request_path, output_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass


manager = Qwen3TtsWorkerManager()


@app.get("/v1/health")
async def health():
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "qwen3_tts_model_dir": QWEN3_TTS_MODEL_DIR,
            "qwen_libs_path": QWEN_LIBS_PATH,
            "prompts_dir": PROMPTS_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": QWEN3_TTS_WORKER_SCRIPT,
            "worker_tmp_dir": QWEN3_TTS_WORKER_TMP_DIR,
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "worker_script": os.path.isfile(QWEN3_TTS_WORKER_SCRIPT),
            "qwen3_tts_model_dir": os.path.isdir(QWEN3_TTS_MODEL_DIR),
            "qwen_libs_path": os.path.isdir(QWEN_LIBS_PATH),
            "torch": module_available("torch"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_env": QWEN3_TTS_CONDA_ENV,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": QWEN3_TTS_REQUEST_TIMEOUT,
            "device_map": QWEN3_TTS_DEVICE_MAP,
            "dtype": QWEN3_TTS_DTYPE,
            "attn_implementation": QWEN3_TTS_ATTN_IMPLEMENTATION,
            "language": QWEN3_TTS_LANGUAGE,
            "x_vector_only": QWEN3_TTS_X_VECTOR_ONLY,
            "max_new_tokens": QWEN3_TTS_MAX_NEW_TOKENS,
            "top_p": QWEN3_TTS_TOP_P,
            "temperature": QWEN3_TTS_TEMPERATURE,
            "max_chars_per_chunk": QWEN3_TTS_MAX_CHARS_PER_CHUNK,
            "pause_ms": QWEN3_TTS_PAUSE_MS,
            "trim_leading_silence": QWEN3_TTS_TRIM_LEADING_SILENCE,
            "trim_leading_silence_threshold_db": QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB,
            "trim_leading_silence_min_ms": QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS,
            "trim_leading_silence_analysis_window_ms": QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS,
            "trim_leading_silence_pre_roll_ms": QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS,
            "trim_leading_silence_max_ms": QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS,
            "use_qwen_libs_sidecar": QWEN3_TTS_USE_QWEN_LIBS,
            "prompt_text_fallback": "upload sidecar -> x-vector-only",
        },
        "last_errors": {
            "qwen3_tts": manager.last_error,
        },
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    with gpu_runtime_lock("qwen3_tts/unload"):
        with manager.lock:
            pass
    return JSONResponse({"code": 200, "msg": "qwen3_tts worker 已退出，无常驻模型"})


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
async def synthesize_v2(request: Qwen3TtsSynthesizeRequest):
    with gpu_runtime_lock("qwen3_tts/synthesize"):
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
                wait_after_cuda_release("after qwen3_tts worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 Qwen3-TTS Voice Clone")
    print("==================================================")
    print(f"[配置] Qwen3-TTS worker env: {QWEN3_TTS_CONDA_ENV}")
    print(f"[配置] Qwen3-TTS 模型目录: {QWEN3_TTS_MODEL_DIR}")
    print(f"[配置] Qwen sidecar libs: {QWEN_LIBS_PATH}")
    print(f"[配置] prompts 目录: {PROMPTS_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] worker 脚本: {QWEN3_TTS_WORKER_SCRIPT}")
    print(
        f"[配置] device_map={QWEN3_TTS_DEVICE_MAP}, dtype={QWEN3_TTS_DTYPE}, "
        f"attn_implementation={QWEN3_TTS_ATTN_IMPLEMENTATION}, language={QWEN3_TTS_LANGUAGE or 'auto'}"
    )
    print(
        f"[配置] x_vector_only={QWEN3_TTS_X_VECTOR_ONLY}, max_new_tokens={QWEN3_TTS_MAX_NEW_TOKENS}, "
        f"max_chars_per_chunk={QWEN3_TTS_MAX_CHARS_PER_CHUNK}, pause_ms={QWEN3_TTS_PAUSE_MS}"
    )
    print(
        f"[配置] trim_leading_silence={QWEN3_TTS_TRIM_LEADING_SILENCE}, "
        f"threshold_db={QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB}, "
        f"min_ms={QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS}"
    )
    print(f"[配置] use_qwen_libs_sidecar={QWEN3_TTS_USE_QWEN_LIBS}")
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={QWEN3_TTS_REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
