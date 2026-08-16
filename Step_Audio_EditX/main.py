#!/usr/bin/env python3
"""Standalone HTTP service for Step-Audio-EditX.

The API process deliberately imports no Torch, vLLM, ONNX Runtime, or upstream
model code.  Those heavy dependencies are loaded only by ``worker.py`` in a
one-shot child process so health checks remain useful on machines without the
model environment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.base import BaseHTTPMiddleware

from audio_output import persist_audio_bytes
from step_audio_editx_runtime import (
    cuda_status,
    env_bool,
    expand_path,
    gpu_runtime_lock,
    terminate_process_group,
    wait_after_cuda_release,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage")))
CLONE_STORAGE_DIR = expand_path(
    os.getenv("CLONE_STORAGE_DIR", str(Path(STORAGE_DIR) / "clone"))
)
PROMPTS_DIR = expand_path(
    os.getenv("PROMPTS_DIR", CLONE_STORAGE_DIR)
)
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", str(Path(STORAGE_DIR) / ".cache/runtime"))
)
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
WORKER_TMP_DIR = expand_path(
    os.getenv(
        "STEP_AUDIO_EDITX_WORKER_TMP_DIR",
        os.path.join(RUNTIME_CACHE_DIR, "step_audio_editx_worker"),
    )
)
WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("STEP_AUDIO_EDITX_HOST", os.getenv("HOST", "0.0.0.0"))
API_PORT = int(os.getenv("STEP_AUDIO_EDITX_PORT", os.getenv("PORT", "8331")))
REQUEST_TIMEOUT = float(os.getenv("STEP_AUDIO_EDITX_REQUEST_TIMEOUT", "900"))
MODEL_DIR = expand_path(
    os.getenv(
        "STEP_AUDIO_EDITX_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "stepfun-ai/Step-Audio-EditX"),
    )
)
TOKENIZER_PATH = expand_path(
    os.getenv(
        "STEP_AUDIO_TOKENIZER_PATH",
        os.path.join(HF_MIRROR_DIR, "stepfun-ai/Step-Audio-Tokenizer"),
    )
)
CODE_PATH = expand_path(
    os.getenv("STEP_AUDIO_EDITX_CODE_PATH", "~/tts-depency/Step-Audio-EditX")
)
STEP_AUDIO_EDITX_OUTPUT_DIR = expand_path(
    os.getenv("STEP_AUDIO_EDITX_OUTPUT_DIR", CLONE_STORAGE_DIR)
)
DTYPE = os.getenv("STEP_AUDIO_EDITX_DTYPE", "bfloat16")
MAX_MODEL_LEN = int(os.getenv("STEP_AUDIO_EDITX_MAX_MODEL_LEN", "3072"))
GPU_MEMORY_UTILIZATION = float(
    os.getenv("STEP_AUDIO_EDITX_GPU_MEMORY_UTILIZATION", "0.5")
)
MAX_NUM_SEQS = int(os.getenv("STEP_AUDIO_EDITX_MAX_NUM_SEQS", "1"))
COSYVOICE_DTYPE = os.getenv("STEP_AUDIO_EDITX_COSYVOICE_DTYPE", "bfloat16")
ENFORCE_EAGER = env_bool("STEP_AUDIO_EDITX_ENFORCE_EAGER", True)
COSYVOICE_CUDA_GRAPH = env_bool("STEP_AUDIO_EDITX_COSYVOICE_CUDA_GRAPH", False)
EDIT_TYPES = frozenset(
    {"emotion", "style", "paralinguistic", "denoise", "vad", "speed"}
)

for directory in (PROMPTS_DIR, WORKER_TMP_DIR, STEP_AUDIO_EDITX_OUTPUT_DIR):
    os.makedirs(directory, exist_ok=True)
os.makedirs(os.path.dirname(GPU_LOCK_FILE) or ".", exist_ok=True)

app = FastAPI(title="Unitale Step-Audio-EditX API")


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


class StepAudioEditXEditRequest(BaseModel):
    """Request schema for the standalone Step-Audio-EditX route."""

    prompt_text: Optional[str] = None
    prompt_audio: str = Field(
        min_length=1, description="经 /v1/upload_audio 上传后的 prompt 音频路径"
    )
    generated_text: Optional[str] = None
    edit_type: Literal["emotion", "style", "paralinguistic", "denoise", "vad", "speed"]
    edit_info: str = ""

    @model_validator(mode="after")
    def validate_edit_contract(self):
        self.prompt_audio = self.prompt_audio.strip()
        self.prompt_text = self.prompt_text.strip() if self.prompt_text else None
        self.generated_text = self.generated_text.strip() if self.generated_text else None
        self.edit_info = self.edit_info.strip()
        if not self.prompt_audio:
            raise ValueError("prompt_audio 不能为空。")
        if self.edit_type not in {"denoise", "vad"} and not self.prompt_text:
            raise ValueError(f"edit_type={self.edit_type} 需要与 prompt_audio 一致的 prompt_text。")
        if self.edit_type not in {"denoise", "vad"} and not self.generated_text:
            self.generated_text = self.prompt_text
        if self.edit_type in {"emotion", "style", "speed"} and not self.edit_info:
            raise ValueError(f"edit_type={self.edit_type} 需要 edit_info。")
        return self


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def hash_filename(filename: str) -> str:
    extension = os.path.splitext(filename)[1] or ".wav"
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{digest}{extension}"


def prompt_audio_path(filename: str) -> Path:
    return Path(PROMPTS_DIR) / hash_filename(filename)


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-8:]) if lines else "Step-Audio-EditX worker 未输出错误信息。"


class StepAudioEditXWorkerManager:
    """Run one upstream model process per request using this uv interpreter."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    @staticmethod
    def _value(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    def build_worker_payload(
        self,
        request: StepAudioEditXEditRequest,
        prompt_wav_path: Path,
    ) -> dict[str, Any]:
        if not prompt_wav_path.is_file():
            raise FileNotFoundError(f"Step-Audio-EditX prompt 音频不存在：{prompt_wav_path}")
        return {
            "prompt_wav_path": str(prompt_wav_path),
            "prompt_text": request.prompt_text,
            "generated_text": request.generated_text,
            "edit_type": request.edit_type,
            "edit_info": request.edit_info,
            "model_path": MODEL_DIR,
            "tokenizer_path": TOKENIZER_PATH,
            "code_path": CODE_PATH,
            "dtype": DTYPE,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_seqs": MAX_NUM_SEQS,
            "cosyvoice_dtype": COSYVOICE_DTYPE,
            "enforce_eager": ENFORCE_EAGER,
            "cosyvoice_cuda_graph": COSYVOICE_CUDA_GRAPH,
            "local_files_only": LOCAL_FILES_ONLY,
        }

    def run_worker(self, payload: dict[str, Any]) -> bytes:
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 Step_Audio_EditX uv 环境的 Python 解释器。")
        if not os.path.isfile(WORKER_SCRIPT):
            raise RuntimeError(f"Step-Audio-EditX worker 脚本不存在：{WORKER_SCRIPT}")
        for path, label in (
            (MODEL_DIR, "Step-Audio-EditX 模型目录"),
            (TOKENIZER_PATH, "Step-Audio-Tokenizer 模型目录"),
            (CODE_PATH, "Step-Audio-EditX 源码目录"),
        ):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"{label}不存在：{path}")

        request_fd, request_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR, prefix="step_audio_editx_req_", suffix=".json"
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=WORKER_TMP_DIR, prefix="step_audio_editx_out_", suffix=".wav"
        )
        os.close(request_fd)
        os.close(output_fd)
        process: Optional[subprocess.Popen] = None
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
            worker_env.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
            print(f"[Step-Audio-EditX] 启动 worker: python={python_executable}")
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
            except subprocess.TimeoutExpired:
                terminate_process_group(process, "Step-Audio-EditX")
                stdout, stderr = process.communicate()
                raise RuntimeError(f"Step-Audio-EditX worker 超时（>{REQUEST_TIMEOUT:.0f}s）")
            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(
                f"[Step-Audio-EditX] worker 退出码={process.returncode}，耗时 {elapsed:.2f}s"
            )
            if process.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Step-Audio-EditX worker 未生成音频文件。")
            with open(output_path, "rb") as file:
                audio_bytes = file.read()
            if not audio_bytes.startswith(b"RIFF"):
                raise RuntimeError("Step-Audio-EditX worker 返回的不是 WAV 文件。")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "Step-Audio-EditX")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = StepAudioEditXWorkerManager()


def step_audio_editx_is_ready() -> bool:
    return (
        os.path.isdir(MODEL_DIR)
        and os.path.isdir(TOKENIZER_PATH)
        and os.path.isfile(os.path.join(CODE_PATH, "tts.py"))
        and os.path.isfile(os.path.join(CODE_PATH, "tokenizer.py"))
        and os.path.isfile(WORKER_SCRIPT)
    )


@app.get("/v1/health")
async def health():
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "step_audio_editx_model_dir": MODEL_DIR,
            "step_audio_tokenizer_path": TOKENIZER_PATH,
            "step_audio_editx_code_path": CODE_PATH,
            "worker_script": WORKER_SCRIPT,
            "worker_tmp_dir": WORKER_TMP_DIR,
            "prompts_dir": PROMPTS_DIR,
            "clone_storage_dir": STEP_AUDIO_EDITX_OUTPUT_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(WORKER_SCRIPT),
            "step_audio_editx": step_audio_editx_is_ready(),
            "torch": module_available("torch"),
            "torchaudio": module_available("torchaudio"),
            "vllm": module_available("vllm"),
            "onnxruntime": module_available("onnxruntime"),
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
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_seqs": MAX_NUM_SEQS,
            "enforce_eager": ENFORCE_EAGER,
            "cosyvoice_dtype": COSYVOICE_DTYPE,
            "cosyvoice_cuda_graph": COSYVOICE_CUDA_GRAPH,
            "flash_attention_policy": "not required; VLLM_ATTENTION_BACKEND=TRITON_ATTN",
            "gpu_scheduling": "shared exclusive file lock",
        },
        "last_errors": {"step_audio_editx": manager.last_error},
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="仅允许本机访问内部接口")
    return {"code": 200, "msg": "Step-Audio-EditX worker 已退出，无常驻模型"}


@app.post("/v1/upload_audio")
async def upload_audio(audio: UploadFile = File(...), full_path: str = Form(...)):
    content = await audio.read()
    save_path = prompt_audio_path(full_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)
    return {"code": 200, "msg": "上传成功", "filename": full_path}


@app.get("/v1/check/audio")
async def check_audio_exists(file_name: str):
    exists = prompt_audio_path(file_name).is_file()
    return {"code": 200 if exists else 404, "exists": exists}


@app.post("/v1/stepAudioEditx/edit")
def step_audio_editx_edit(request: StepAudioEditXEditRequest):
    prompt_path = prompt_audio_path(request.prompt_audio)
    with gpu_runtime_lock(GPU_LOCK_FILE, "step-audio-editx/edit"):
        with manager.lock:
            try:
                payload = manager.build_worker_payload(request, prompt_path)
                audio_bytes = manager.run_worker(payload)
                saved_output_path = persist_audio_bytes(
                    audio_bytes,
                    "step_audio_editx",
                    STEP_AUDIO_EDITX_OUTPUT_DIR,
                )
                print(f"[Step-Audio-EditX] 已保存语音音频: {saved_output_path}")
                return Response(content=audio_bytes, media_type="audio/wav")
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except HTTPException:
                raise
            except Exception as exc:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                wait_after_cuda_release(CUDA_RELEASE_DELAY, "after Step-Audio-EditX worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 Step-Audio-EditX (uv)")
    print("==================================================")
    print(f"[配置] 模型目录: {MODEL_DIR}")
    print(f"[配置] Tokenizer 目录: {TOKENIZER_PATH}")
    print(f"[配置] 上游源码: {CODE_PATH}")
    print(f"[配置] worker: {WORKER_SCRIPT}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] host={API_HOST}, port={API_PORT}, local_files_only={LOCAL_FILES_ONLY}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
