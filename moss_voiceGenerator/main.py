#!/usr/bin/env python3
"""独立 MOSS-VoiceGenerator VoiceDesign HTTP 服务。"""

from __future__ import annotations

# 学习入口：MOSS VoiceGenerator 的模型只在一次性 worker 中加载，API 进程保持轻量。
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import (
    AudioReferenceStore,
    GpuLockTimeoutError,
)
from unitale_runtime import (
    gpu_runtime_lock as shared_gpu_runtime_lock,
)

from audio_output import persist_audio_bytes
from moss_voice_design_compat import is_moss_codec_path_ready
from runtime import cuda_status, terminate_process_group

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent


def env_bool(name: str, default: bool = False) -> bool:
    """解析启动配置中的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    """展开环境变量和用户目录，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
TIMBRE_STORAGE_DIR = Path(expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(STORAGE_DIR / "timbre"))))
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
MOSS_VOICEGENERATOR_MODEL_DIR = expand_path(
    os.getenv(
        "MOSS_VOICEGENERATOR_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "OpenMOSS-Team/MOSS-VoiceGenerator"),
    )
)
MOSS_AUDIO_TOKENIZER_PATH = expand_path(
    os.getenv(
        "MOSS_AUDIO_TOKENIZER_PATH",
        os.path.join(HF_MIRROR_DIR, "OpenMOSS-Team/MOSS-Audio-Tokenizer"),
    )
)
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
WORKER_TMP_DIR = expand_path(
    os.getenv(
        "MOSS_VOICEGENERATOR_WORKER_TMP_DIR",
        os.path.join(RUNTIME_CACHE_DIR, "moss_voicegenerator_worker"),
    )
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("MOSS_VOICEGENERATOR_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("MOSS_VOICEGENERATOR_PORT", os.getenv("PORT", "8302")))
REQUEST_TIMEOUT = float(os.getenv("MOSS_VOICEGENERATOR_REQUEST_TIMEOUT", "900"))
MAX_CHARS_PER_CHUNK = int(os.getenv("MOSS_VOICEGENERATOR_MAX_CHARS_PER_CHUNK", "0"))
PAUSE_MS = int(os.getenv("MOSS_VOICEGENERATOR_PAUSE_MS", "250"))
MAX_NEW_TOKENS = int(os.getenv("MOSS_VOICEGENERATOR_MAX_NEW_TOKENS", "4096"))
AUDIO_TEMPERATURE = float(os.getenv("MOSS_VOICEGENERATOR_AUDIO_TEMPERATURE", "1.5"))
AUDIO_TOP_P = float(os.getenv("MOSS_VOICEGENERATOR_AUDIO_TOP_P", "0.6"))
AUDIO_TOP_K = int(os.getenv("MOSS_VOICEGENERATOR_AUDIO_TOP_K", "50"))
AUDIO_REPETITION_PENALTY = float(os.getenv("MOSS_VOICEGENERATOR_AUDIO_REPETITION_PENALTY", "1.1"))
DTYPE = os.getenv("MOSS_VOICEGENERATOR_DTYPE", "auto")
ATTN_IMPLEMENTATION = os.getenv("MOSS_VOICEGENERATOR_ATTN_IMPLEMENTATION", "auto")

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

reference_store = AudioReferenceStore(STORAGE_DIR / "clone", TIMBRE_STORAGE_DIR)

app = FastAPI(title="Unitale MOSS-VoiceGenerator VoiceDesign API")


class ForceCORS(BaseHTTPMiddleware):
    """为本地 WebUI 提供跨域响应和预检处理。"""

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


class MossDesignRequest(BaseModel):
    """独立 MOSS VoiceGenerator 音色设计接口的请求参数。"""

    voice_description: str = Field(min_length=1, max_length=2_000)
    text: str = Field(default="这是生成的参考音频预览。", min_length=1, max_length=12_000)
    max_chars_per_chunk: int | None = Field(default=0, ge=0, le=2_000)
    pause_ms: int | None = Field(default=250, ge=0, le=10_000)
    max_new_tokens: int | None = Field(default=4096, ge=1, le=8_192)
    audio_temperature: float | None = Field(default=1.5, ge=0, le=5)
    audio_top_p: float | None = Field(default=0.6, gt=0, le=1)
    audio_top_k: int | None = Field(default=50, ge=1, le=500)
    audio_repetition_penalty: float | None = Field(default=1.1, ge=0.1, le=3)
    dtype: str | None = Field(default="auto", max_length=32)
    attn_implementation: str | None = Field(default="auto", max_length=64)


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def gpu_runtime_lock(label: str):
    """通过共享文件锁串行化 MOSS 的 GPU 推理。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release(label: str = "") -> None:
    """worker 退出后等待 CUDA 释放，再允许下一个请求进入。"""
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "MOSS-VoiceGenerator worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class MossVoiceGeneratorWorkerManager:
    """组装音色设计请求，并管理一次性 worker 的临时文件。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: str | None = None

    @staticmethod
    def _value(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    def build_worker_payload(self, request: MossDesignRequest) -> dict[str, Any]:
        """将请求字段与环境默认值组合成 worker JSON。"""
        payload = request.model_dump()
        payload.update(
            {
                "model_path": MOSS_VOICEGENERATOR_MODEL_DIR,
                "codec_path": MOSS_AUDIO_TOKENIZER_PATH,
                "local_files_only": LOCAL_FILES_ONLY,
                "max_chars_per_chunk": self._value(
                    request.max_chars_per_chunk, MAX_CHARS_PER_CHUNK
                ),
                "pause_ms": self._value(request.pause_ms, PAUSE_MS),
                "max_new_tokens": self._value(request.max_new_tokens, MAX_NEW_TOKENS),
                "audio_temperature": self._value(request.audio_temperature, AUDIO_TEMPERATURE),
                "audio_top_p": self._value(request.audio_top_p, AUDIO_TOP_P),
                "audio_top_k": self._value(request.audio_top_k, AUDIO_TOP_K),
                "audio_repetition_penalty": self._value(
                    request.audio_repetition_penalty, AUDIO_REPETITION_PENALTY
                ),
                "dtype": self._value(request.dtype, DTYPE),
                "attn_implementation": self._value(
                    request.attn_implementation, ATTN_IMPLEMENTATION
                ),
            }
        )
        return payload

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        """在 MOSS 项目解释器中执行一次 VoiceGenerator 推理。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 moss_voiceGenerator uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"MOSS VoiceGenerator worker 脚本不存在: {WORKER_SCRIPT}")
        if not os.path.isdir(MOSS_VOICEGENERATOR_MODEL_DIR):
            raise RuntimeError(
                f"MOSS VoiceGenerator 模型目录不存在: {MOSS_VOICEGENERATOR_MODEL_DIR}"
            )
        if not is_moss_codec_path_ready(MOSS_AUDIO_TOKENIZER_PATH):
            raise RuntimeError(
                f"MOSS 音频 tokenizer 目录不可用或不是 v1：{MOSS_AUDIO_TOKENIZER_PATH}"
            )

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR, prefix="moss_voicegenerator_req_", suffix=".json"
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR, prefix="moss_voicegenerator_out_", suffix=".wav"
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
            print(f"[MOSS VoiceGenerator] 启动 worker: python={python_executable}")
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
                terminate_process_group(process, "MOSS VoiceGenerator")
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"MOSS VoiceGenerator worker 超时（>{REQUEST_TIMEOUT:.0f}s）"
                ) from exc
            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[MOSS VoiceGenerator] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s")
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("MOSS VoiceGenerator worker 未生成音频文件。")
            with open(output_path, "rb") as file:
                audio_bytes = file.read()
            if not audio_bytes:
                raise RuntimeError("MOSS VoiceGenerator worker 返回空 WAV。")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "MOSS VoiceGenerator")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = MossVoiceGeneratorWorkerManager()


@app.get("/v1/health")
def health():
    """返回 MOSS 模型、tokenizer、worker 和 GPU 的状态。"""
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "moss_voicegenerator_model_dir": MOSS_VOICEGENERATOR_MODEL_DIR,
            "moss_audio_tokenizer_path": MOSS_AUDIO_TOKENIZER_PATH,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
            "timbre_storage_dir": str(TIMBRE_STORAGE_DIR),
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "moss_voicegenerator_model_dir": os.path.isdir(MOSS_VOICEGENERATOR_MODEL_DIR),
            "moss_audio_tokenizer": is_moss_codec_path_ready(MOSS_AUDIO_TOKENIZER_PATH),
            "torch": module_available("torch"),
            "transformers": module_available("transformers"),
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
            "dtype": DTYPE,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "flash_attention_policy": "optional; auto falls back to sdpa when unavailable",
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_chars_per_chunk": MAX_CHARS_PER_CHUNK,
            "pause_ms": PAUSE_MS,
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"moss_voicegenerator": manager.last_error},
    }


@app.post("/v1/moss/timbre")
def moss_design(request: MossDesignRequest):
    """串行执行 MOSS 音色设计，并将 WAV 保存到 timbre 目录。"""
    try:
        payload = manager.build_worker_payload(request)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("MOSS VoiceGenerator request preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("moss/design"):
            with manager.lock:
                try:
                    audio_bytes = manager.run_worker(payload)
                    saved_output_path = persist_audio_bytes(
                        audio_bytes,
                        "moss_voicegenerator",
                        TIMBRE_STORAGE_DIR,
                    )
                    reference_store.register_timbre_file(saved_output_path)
                    print(f"[MOSS VoiceGenerator] 已保存音色音频: {saved_output_path}")
                    return Response(
                        content=audio_bytes,
                        media_type="audio/wav",
                    )
                except HTTPException:
                    raise
                except Exception as exc:
                    LOGGER.exception("MOSS VoiceGenerator request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release("after MOSS VoiceGenerator worker")
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 MOSS-VoiceGenerator VoiceDesign")
    print("==================================================")
    print(f"[配置] 模型目录: {MOSS_VOICEGENERATOR_MODEL_DIR}")
    print(f"[配置] 音频 tokenizer: {MOSS_AUDIO_TOKENIZER_PATH}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(
        f"[配置] host={API_HOST}, port={API_PORT}, dtype={DTYPE}, "
        f"attn_implementation={ATTN_IMPLEMENTATION}"
    )
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
