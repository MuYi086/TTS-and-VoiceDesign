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
from typing import Optional, List, Any

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


def env_optional_text(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() == "none":
        return None
    return normalized


def env_optional_int(name: str) -> Optional[int]:
    value = env_optional_text(name)
    return int(value) if value is not None else None


def env_optional_float(name: str) -> Optional[float]:
    value = env_optional_text(name)
    return float(value) if value is not None else None


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def optional_expand_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return expand_path(normalized)


def default_moss_codec_path(hf_mirror_dir: str) -> str:
    local_path = expand_path(os.path.join(hf_mirror_dir, "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"))
    return local_path if os.path.exists(local_path) else "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"


def normalize_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.lower() == "none":
        return None
    return normalized


def codec_reference_info(reference: str) -> dict[str, Any]:
    expanded = os.path.expandvars(os.path.expanduser(reference))
    exists = os.path.exists(expanded)
    return {
        "value": os.path.abspath(expanded) if exists else reference,
        "local_path_exists": exists,
        "kind": "local_path" if exists else "model_id",
    }


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", os.path.join(API_DIR, "prompts")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", os.path.join(API_DIR, ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8303"))

MOSS_CONDA_ENV = os.getenv("MOSS_CONDA_ENV", "moss-tts-py310")
MOSS_MODEL_DIR = expand_path(
    os.getenv("MOSS_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"))
)
MOSS_CODEC_PATH = os.getenv("MOSS_CODEC_PATH", default_moss_codec_path(HF_MIRROR_DIR))
MOSS_LANGUAGE = os.getenv("MOSS_LANGUAGE", "Chinese")
MOSS_INSTRUCTION = env_optional_text("MOSS_INSTRUCTION")
MOSS_QUALITY = env_optional_text("MOSS_QUALITY")
MOSS_TOKENS = env_optional_int("MOSS_TOKENS")
MOSS_MAX_NEW_TOKENS = int(os.getenv("MOSS_MAX_NEW_TOKENS", "4096"))
MOSS_AUTO_LIMIT_MAX_NEW_TOKENS = env_bool("MOSS_AUTO_LIMIT_MAX_NEW_TOKENS", True)
MOSS_MIN_NEW_TOKENS = int(os.getenv("MOSS_MIN_NEW_TOKENS", "256"))
MOSS_NEW_TOKENS_PER_CHAR = float(os.getenv("MOSS_NEW_TOKENS_PER_CHAR", "10"))
MOSS_N_VQ_FOR_INFERENCE = env_optional_int("MOSS_N_VQ_FOR_INFERENCE")
MOSS_AUDIO_TEMPERATURE = float(os.getenv("MOSS_AUDIO_TEMPERATURE", "1.7"))
MOSS_AUDIO_TOP_P = float(os.getenv("MOSS_AUDIO_TOP_P", "0.8"))
MOSS_AUDIO_TOP_K = int(os.getenv("MOSS_AUDIO_TOP_K", "25"))
MOSS_AUDIO_REPETITION_PENALTY = float(os.getenv("MOSS_AUDIO_REPETITION_PENALTY", "1.0"))
MOSS_TEXT_TEMPERATURE = env_optional_float("MOSS_TEXT_TEMPERATURE")
MOSS_TEXT_TOP_P = env_optional_float("MOSS_TEXT_TOP_P")
MOSS_TEXT_TOP_K = env_optional_int("MOSS_TEXT_TOP_K")
MOSS_TEXT_REPETITION_PENALTY = env_optional_float("MOSS_TEXT_REPETITION_PENALTY")
MOSS_ATTN_IMPLEMENTATION = os.getenv("MOSS_ATTN_IMPLEMENTATION", "auto")
MOSS_SDPA_BACKEND = os.getenv("MOSS_SDPA_BACKEND", "math")
MOSS_DTYPE = os.getenv("MOSS_DTYPE", "auto")
MOSS_MAX_CHARS_PER_CHUNK = int(os.getenv("MOSS_MAX_CHARS_PER_CHUNK", "80"))
MOSS_PAUSE_MS = int(os.getenv("MOSS_PAUSE_MS", "250"))
MOSS_REQUEST_TIMEOUT = float(os.getenv("MOSS_REQUEST_TIMEOUT", "600"))
MOSS_CUDA_RETRY_COUNT = max(0, int(os.getenv("MOSS_CUDA_RETRY_COUNT", "1")))
MOSS_CUDA_RETRY_MAX_NEW_TOKENS = int(os.getenv("MOSS_CUDA_RETRY_MAX_NEW_TOKENS", "512"))

MOSS_WORKER_SCRIPT = os.path.join(API_DIR, "moss_tts_worker.py")
MOSS_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "moss_worker")

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
os.makedirs(MOSS_WORKER_TMP_DIR, exist_ok=True)
gpu_lock_dir = os.path.dirname(GPU_LOCK_FILE)
if gpu_lock_dir:
    os.makedirs(gpu_lock_dir, exist_ok=True)

app = FastAPI(title="Unitale MOSS-TTS API")


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
        return "MOSS worker 未输出错误信息。"
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


class MossSynthesizeRequest(CloneSynthesisRequest):

    text: str
    audio_path: str
    prompt_text: Optional[str] = None
    language: Optional[str] = None
    instruction: Optional[str] = None
    quality: Optional[str] = None
    tokens: Optional[int] = None
    codec_path: Optional[str] = None
    max_new_tokens: Optional[int] = None
    n_vq_for_inference: Optional[int] = None
    audio_temperature: Optional[float] = None
    audio_top_p: Optional[float] = None
    audio_top_k: Optional[int] = None
    audio_repetition_penalty: Optional[float] = None
    text_temperature: Optional[float] = None
    text_top_p: Optional[float] = None
    text_top_k: Optional[int] = None
    text_repetition_penalty: Optional[float] = None
    attn_implementation: Optional[str] = None
    dtype: Optional[str] = None
    max_chars_per_chunk: Optional[int] = None
    pause_ms: Optional[int] = None
    emo_text: Optional[str] = None
    emo_vector: Optional[List[float]] = None


class MossWorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: MossSynthesizeRequest) -> dict:
        ref_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        codec_path = normalize_optional_str(request.codec_path) or MOSS_CODEC_PATH

        return {
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "model_path": MOSS_MODEL_DIR,
            "codec_path": codec_path,
            "language": normalize_language(request.language) if request.language is not None else normalize_language(MOSS_LANGUAGE),
            "instruction": normalize_optional_str(request.instruction) or MOSS_INSTRUCTION,
            "quality": normalize_optional_str(request.quality) or MOSS_QUALITY,
            "tokens": request.tokens if request.tokens is not None else MOSS_TOKENS,
            "max_new_tokens": request.max_new_tokens if request.max_new_tokens is not None else MOSS_MAX_NEW_TOKENS,
            "auto_limit_max_new_tokens": (
                MOSS_AUTO_LIMIT_MAX_NEW_TOKENS and request.max_new_tokens is None
            ),
            "min_new_tokens": MOSS_MIN_NEW_TOKENS,
            "new_tokens_per_char": MOSS_NEW_TOKENS_PER_CHAR,
            "n_vq_for_inference": request.n_vq_for_inference if request.n_vq_for_inference is not None else MOSS_N_VQ_FOR_INFERENCE,
            "audio_temperature": request.audio_temperature if request.audio_temperature is not None else MOSS_AUDIO_TEMPERATURE,
            "audio_top_p": request.audio_top_p if request.audio_top_p is not None else MOSS_AUDIO_TOP_P,
            "audio_top_k": request.audio_top_k if request.audio_top_k is not None else MOSS_AUDIO_TOP_K,
            "audio_repetition_penalty": (
                request.audio_repetition_penalty
                if request.audio_repetition_penalty is not None
                else MOSS_AUDIO_REPETITION_PENALTY
            ),
            "text_temperature": request.text_temperature if request.text_temperature is not None else MOSS_TEXT_TEMPERATURE,
            "text_top_p": request.text_top_p if request.text_top_p is not None else MOSS_TEXT_TOP_P,
            "text_top_k": request.text_top_k if request.text_top_k is not None else MOSS_TEXT_TOP_K,
            "text_repetition_penalty": (
                request.text_repetition_penalty
                if request.text_repetition_penalty is not None
                else MOSS_TEXT_REPETITION_PENALTY
            ),
            "attn_implementation": request.attn_implementation or MOSS_ATTN_IMPLEMENTATION,
            "sdpa_backend": MOSS_SDPA_BACKEND,
            "dtype": request.dtype or MOSS_DTYPE,
            "max_chars_per_chunk": request.max_chars_per_chunk if request.max_chars_per_chunk is not None else MOSS_MAX_CHARS_PER_CHUNK,
            "pause_ms": request.pause_ms if request.pause_ms is not None else MOSS_PAUSE_MS,
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def run_worker(self, payload: dict) -> bytes:
        attempt_payload = dict(payload)
        try:
            for attempt in range(MOSS_CUDA_RETRY_COUNT + 1):
                try:
                    audio_bytes = self._run_worker_once(attempt_payload)
                    self.last_error = None
                    return audio_bytes
                except RuntimeError as exc:
                    if attempt >= MOSS_CUDA_RETRY_COUNT or not is_retryable_cuda_error(exc):
                        raise

                    retry_number = attempt + 1
                    print(
                        f"[MOSS] 检测到可恢复的 CUDA 异常，准备重试 "
                        f"{retry_number}/{MOSS_CUDA_RETRY_COUNT}: {exc}"
                    )
                    attempt_payload = dict(attempt_payload)
                    attempt_payload["attn_implementation"] = "eager"
                    attempt_payload["sdpa_backend"] = "math"
                    attempt_payload["auto_limit_max_new_tokens"] = True
                    attempt_payload["max_new_tokens"] = min(
                        int(attempt_payload.get("max_new_tokens") or MOSS_MAX_NEW_TOKENS),
                        MOSS_CUDA_RETRY_MAX_NEW_TOKENS,
                    )
                    wait_after_cuda_release("before MOSS CUDA retry")
        except Exception as exc:
            self.last_error = str(exc)
            raise

        raise RuntimeError("MOSS worker 重试流程异常结束。")

    def _run_worker_once(self, payload: dict) -> bytes:
        conda_exe = resolve_conda_executable()
        if not conda_exe:
            raise RuntimeError("未找到 conda 命令，无法调用 MOSS worker。")
        if not os.path.isfile(MOSS_WORKER_SCRIPT):
            raise RuntimeError(f"MOSS worker 脚本不存在: {MOSS_WORKER_SCRIPT}")
        if not os.path.isdir(MOSS_MODEL_DIR):
            raise RuntimeError(f"MOSS 模型目录不存在: {MOSS_MODEL_DIR}")

        request_fd, request_path = tempfile.mkstemp(dir=MOSS_WORKER_TMP_DIR, prefix="moss_req_", suffix=".json")
        output_fd, output_path = tempfile.mkstemp(dir=MOSS_WORKER_TMP_DIR, prefix="moss_out_", suffix=".wav")
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
                MOSS_CONDA_ENV,
                "python",
                MOSS_WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[MOSS] 启动 worker: env={MOSS_CONDA_ENV}")
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
                stdout, stderr = proc.communicate(timeout=MOSS_REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                terminate_process_group(proc, "MOSS")
                stdout, stderr = proc.communicate()
                raise RuntimeError(f"MOSS worker 超时（>{MOSS_REQUEST_TIMEOUT:.0f}s）")

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[MOSS] worker 退出码={proc.returncode}，耗时 {elapsed:.2f}s")

            if proc.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("MOSS worker 未生成音频文件。")

            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes
        finally:
            terminate_process_group(proc, "MOSS")
            for path in (request_path, output_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass


manager = MossWorkerManager()


@app.get("/v1/health")
async def health():
    codec_info = codec_reference_info(MOSS_CODEC_PATH)
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "moss_model_dir": MOSS_MODEL_DIR,
            "moss_codec_path": codec_info["value"],
            "moss_helper_script": MOSS_WORKER_SCRIPT,
            "prompts_dir": PROMPTS_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": MOSS_WORKER_SCRIPT,
            "worker_tmp_dir": MOSS_WORKER_TMP_DIR,
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "worker_script": os.path.isfile(MOSS_WORKER_SCRIPT),
            "moss_helper_script": os.path.isfile(MOSS_WORKER_SCRIPT),
            "moss_model_dir": os.path.isdir(MOSS_MODEL_DIR),
            "moss_codec_reference": bool(MOSS_CODEC_PATH),
            "moss_codec_local_path": codec_info["local_path_exists"],
            "torch": module_available("torch"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_env": MOSS_CONDA_ENV,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "helper_source": "bundled in moss_tts_worker.py",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": MOSS_REQUEST_TIMEOUT,
            "language": MOSS_LANGUAGE,
            "instruction": MOSS_INSTRUCTION,
            "quality": MOSS_QUALITY,
            "tokens": MOSS_TOKENS,
            "max_new_tokens": MOSS_MAX_NEW_TOKENS,
            "auto_limit_max_new_tokens": MOSS_AUTO_LIMIT_MAX_NEW_TOKENS,
            "min_new_tokens": MOSS_MIN_NEW_TOKENS,
            "new_tokens_per_char": MOSS_NEW_TOKENS_PER_CHAR,
            "n_vq_for_inference": MOSS_N_VQ_FOR_INFERENCE,
            "audio_temperature": MOSS_AUDIO_TEMPERATURE,
            "audio_top_p": MOSS_AUDIO_TOP_P,
            "audio_top_k": MOSS_AUDIO_TOP_K,
            "audio_repetition_penalty": MOSS_AUDIO_REPETITION_PENALTY,
            "text_temperature": MOSS_TEXT_TEMPERATURE,
            "text_top_p": MOSS_TEXT_TOP_P,
            "text_top_k": MOSS_TEXT_TOP_K,
            "text_repetition_penalty": MOSS_TEXT_REPETITION_PENALTY,
            "attn_implementation": MOSS_ATTN_IMPLEMENTATION,
            "sdpa_backend": MOSS_SDPA_BACKEND,
            "dtype": MOSS_DTYPE,
            "max_chars_per_chunk": MOSS_MAX_CHARS_PER_CHUNK,
            "pause_ms": MOSS_PAUSE_MS,
            "cuda_retry_count": MOSS_CUDA_RETRY_COUNT,
            "cuda_retry_max_new_tokens": MOSS_CUDA_RETRY_MAX_NEW_TOKENS,
            "codec_reference_kind": codec_info["kind"],
        },
        "last_errors": {
            "moss_tts": manager.last_error,
        },
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    with gpu_runtime_lock("moss_tts/unload"):
        with manager.lock:
            pass
    return JSONResponse({"code": 200, "msg": "moss_tts worker 已退出，无常驻模型"})


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
async def synthesize_v2(request: MossSynthesizeRequest):
    with gpu_runtime_lock("moss_tts/synthesize"):
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
                wait_after_cuda_release("after moss worker")


if __name__ == "__main__":
    codec_info = codec_reference_info(MOSS_CODEC_PATH)
    print("==================================================")
    print("   Unitale AI 本地后端 MOSS-TTS Voice Clone")
    print("==================================================")
    print(f"[配置] MOSS worker env: {MOSS_CONDA_ENV}")
    print(f"[配置] MOSS 模型目录: {MOSS_MODEL_DIR}")
    print(f"[配置] MOSS codec: {codec_info['value']} ({codec_info['kind']})")
    print(f"[配置] MOSS helpers: bundled in {MOSS_WORKER_SCRIPT}")
    print(f"[配置] prompts 目录: {PROMPTS_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] worker 脚本: {MOSS_WORKER_SCRIPT}")
    print(
        f"[配置] language={MOSS_LANGUAGE}, max_new_tokens={MOSS_MAX_NEW_TOKENS}, "
        f"auto_limit={MOSS_AUTO_LIMIT_MAX_NEW_TOKENS}, "
        f"min/per_char={MOSS_MIN_NEW_TOKENS}/{MOSS_NEW_TOKENS_PER_CHAR:g}, "
        f"audio_temperature={MOSS_AUDIO_TEMPERATURE}, audio_top_p={MOSS_AUDIO_TOP_P}, "
        f"audio_top_k={MOSS_AUDIO_TOP_K}"
    )
    print(
        f"[配置] attn_implementation={MOSS_ATTN_IMPLEMENTATION}, dtype={MOSS_DTYPE}, "
        f"sdpa_backend={MOSS_SDPA_BACKEND}, max_chars_per_chunk={MOSS_MAX_CHARS_PER_CHUNK}, "
        f"pause_ms={MOSS_PAUSE_MS}"
    )
    print(
        f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={MOSS_REQUEST_TIMEOUT}, "
        f"cuda_retry={MOSS_CUDA_RETRY_COUNT}, retry_cap={MOSS_CUDA_RETRY_MAX_NEW_TOKENS}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
