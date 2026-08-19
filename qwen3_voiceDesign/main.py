#!/usr/bin/env python3
"""Standalone HTTP service for Qwen3-TTS VoiceDesign."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from audio_output import persist_audio_bytes
from voicedesign_runtime import cuda_status, terminate_process_group

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
TIMBRE_STORAGE_DIR = Path(expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(STORAGE_DIR / "timbre"))))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
QWEN_VOICEDESIGN_MODEL_DIR = expand_path(
    os.getenv(
        "QWEN_VOICEDESIGN_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    )
)
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
WORKER_TMP_DIR = expand_path(
    os.getenv(
        "QWEN_VOICEDESIGN_WORKER_TMP_DIR",
        os.path.join(RUNTIME_CACHE_DIR, "qwen_voicedesign_worker"),
    )
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("QWEN_VOICEDESIGN_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("QWEN_VOICEDESIGN_PORT", os.getenv("PORT", "8301")))
REQUEST_TIMEOUT = float(os.getenv("QWEN_VOICEDESIGN_REQUEST_TIMEOUT", "900"))
DEVICE_MAP = os.getenv("QWEN_VOICEDESIGN_DEVICE_MAP", "cuda:0")
DTYPE = os.getenv("QWEN_VOICEDESIGN_DTYPE", "auto")
ATTN_IMPLEMENTATION = os.getenv(
    "QWEN_VOICEDESIGN_ATTN_IMPLEMENTATION",
    "auto",
)
LANGUAGE = os.getenv("QWEN_VOICEDESIGN_LANGUAGE", "Chinese")
MAX_CHARS_PER_CHUNK = int(os.getenv("QWEN_VOICEDESIGN_MAX_CHARS_PER_CHUNK", "0"))
PAUSE_MS = int(os.getenv("QWEN_VOICEDESIGN_PAUSE_MS", "250"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_VOICEDESIGN_MAX_NEW_TOKENS", "2048"))

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(RUNTIME_CACHE_DIR, "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(RUNTIME_CACHE_DIR, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RUNTIME_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RUNTIME_CACHE_DIR, "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for path in (
    WORKER_TMP_DIR,
    TIMBRE_STORAGE_DIR,
    os.environ["HF_MODULES_CACHE"],
    os.environ["NUMBA_CACHE_DIR"],
):
    os.makedirs(path, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.path.dirname(GPU_LOCK_FILE) or ".", exist_ok=True)

app = FastAPI(title="Unitale Qwen3-TTS VoiceDesign API")


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


class QwenDesignRequest(BaseModel):
    """Request schema for the standalone VoiceDesign service."""

    voice_description: str
    text: str = "这是生成的参考音频预览。"
    save_as: str | None = "designed_voice.wav"
    language: str | None = "Chinese"
    max_chars_per_chunk: int | None = 0
    pause_ms: int | None = 250
    max_new_tokens: int | None = 2048
    top_p: float | None = None
    temperature: float | None = None
    dtype: str | None = "auto"
    attn_implementation: str | None = "auto"
    device_map: str | None = "cuda:0"


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


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


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "Qwen3-TTS VoiceDesign worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class QwenVoiceDesignWorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: str | None = None

    @staticmethod
    def _value(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    def build_worker_payload(self, request: QwenDesignRequest) -> dict[str, Any]:
        payload = request.model_dump()
        payload.update(
            {
                "model_path": QWEN_VOICEDESIGN_MODEL_DIR,
                "local_files_only": LOCAL_FILES_ONLY,
                "language": self._value(request.language, LANGUAGE),
                "max_chars_per_chunk": self._value(
                    request.max_chars_per_chunk, MAX_CHARS_PER_CHUNK
                ),
                "pause_ms": self._value(request.pause_ms, PAUSE_MS),
                "max_new_tokens": self._value(request.max_new_tokens, MAX_NEW_TOKENS),
                "dtype": self._value(request.dtype, DTYPE),
                "attn_implementation": self._value(
                    request.attn_implementation, ATTN_IMPLEMENTATION
                ),
                "device_map": self._value(request.device_map, DEVICE_MAP),
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 qwen3_voiceDesign uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"Qwen3-TTS VoiceDesign worker 脚本不存在: {WORKER_SCRIPT}")
        if not os.path.isdir(QWEN_VOICEDESIGN_MODEL_DIR):
            raise RuntimeError(
                f"Qwen3-TTS VoiceDesign 模型目录不存在: {QWEN_VOICEDESIGN_MODEL_DIR}"
            )

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="qwen_voicedesign_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR,
            prefix="qwen_voicedesign_out_",
            suffix=".wav",
        )
        os.close(request_fd)
        os.close(output_fd)
        process: subprocess.Popen | None = None
        try:
            with open(request_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)

            command = [
                python_executable,
                WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            worker_env = os.environ.copy()
            if LOCAL_FILES_ONLY:
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            print(f"[Qwen3-TTS VoiceDesign] 启动 worker: python={python_executable}")
            started = time.perf_counter()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=worker_env,
            )
            try:
                stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(process, "Qwen3-TTS VoiceDesign")
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"Qwen3-TTS VoiceDesign worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(
                f"[Qwen3-TTS VoiceDesign] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s"
            )
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Qwen3-TTS VoiceDesign worker 未生成音频文件。")
            with open(output_path, "rb") as file:
                audio_bytes = file.read()
            if not audio_bytes:
                raise RuntimeError("Qwen3-TTS VoiceDesign worker 返回空 WAV。")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Qwen3-TTS VoiceDesign")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = QwenVoiceDesignWorkerManager()


@app.get("/v1/health")
async def health():
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "qwen_voicedesign_model_dir": QWEN_VOICEDESIGN_MODEL_DIR,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
            "timbre_storage_dir": str(TIMBRE_STORAGE_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "qwen_voicedesign_model_dir": os.path.isdir(QWEN_VOICEDESIGN_MODEL_DIR),
            "torch": module_available("torch"),
            "qwen_tts": module_available("qwen_tts"),
            "soundfile": module_available("soundfile"),
            "flash_attn": module_available("flash_attn"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": REQUEST_TIMEOUT,
            "device_map": DEVICE_MAP,
            "dtype": DTYPE,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "flash_attention_policy": "optional; auto falls back to sdpa when unavailable",
            "language": LANGUAGE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_chars_per_chunk": MAX_CHARS_PER_CHUNK,
            "pause_ms": PAUSE_MS,
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"qwen_voicedesign": manager.last_error},
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="仅允许本机访问内部接口")
    return {"code": 200, "msg": "qwen3_voiceDesign worker 已退出，无常驻模型"}


@app.post("/v1/qwen/timbre")
def qwen_design(request: QwenDesignRequest):
    with gpu_runtime_lock("qwen/design"):
        with manager.lock:
            try:
                audio_bytes = manager.run_worker(manager.build_worker_payload(request))
                saved_output_path = persist_audio_bytes(
                    audio_bytes,
                    "qwen_voicedesign",
                    TIMBRE_STORAGE_DIR,
                )
                print(f"[Qwen3-TTS VoiceDesign] 已保存音色音频: {saved_output_path}")
                return Response(content=audio_bytes, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as exc:
                LOGGER.exception("Qwen3 VoiceDesign request failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                wait_after_cuda_release("after Qwen VoiceDesign worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 Qwen3-TTS VoiceDesign")
    print("==================================================")
    print(f"[配置] 模型目录: {QWEN_VOICEDESIGN_MODEL_DIR}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, device_map={DEVICE_MAP}, "
        f"dtype={DTYPE}, attn_implementation={ATTN_IMPLEMENTATION}"
    )
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
