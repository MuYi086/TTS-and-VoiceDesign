"""Standalone uv HTTP service for LongCat-AudioDiT-3.5B voice cloning.

This module owns the LongCat API and launches worker.py with the current uv
Python. The former Conda API/worker implementation has been removed after the
uv migration was confirmed by the real GPU and HTTP canaries.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.base import BaseHTTPMiddleware

from runtime import cuda_status, terminate_process_group


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
API_DIR = REPOSITORY_DIR / "api"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", str(API_DIR / "prompts")))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", str(API_DIR / ".cache/runtime"))
)
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", str(Path(RUNTIME_CACHE_DIR) / "gpu-runtime.lock"))
)
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8307"))

LONGCAT_AUDIODIT_MODEL_DIR = expand_path(
    os.getenv(
        "LONGCAT_AUDIODIT_MODEL_DIR",
        os.getenv(
            "LONGCAT_AUDIODIT_35B_BF16_MODEL_PATH",
            str(Path(HF_MIRROR_DIR) / "drbaph/LongCat-AudioDiT-3.5B-bf16"),
        ),
    )
)
LONGCAT_AUDIODIT_REPO_PATH = expand_path(
    os.getenv("LONGCAT_AUDIODIT_REPO_PATH", "~/tts-depency/LongCat-AudioDiT")
)
LONGCAT_AUDIODIT_TOKENIZER_PATH = expand_path(
    os.getenv("LONGCAT_AUDIODIT_TOKENIZER_PATH", "~/hf-mirror/google/umt5-base")
)
LONGCAT_AUDIODIT_WORKER_SCRIPT = str(PROJECT_DIR / "worker.py")
LONGCAT_AUDIODIT_WORKER_TMP_DIR = expand_path(
    os.getenv(
        "LONGCAT_AUDIODIT_WORKER_TMP_DIR",
        str(Path(RUNTIME_CACHE_DIR) / "longcat_audiodit_worker"),
    )
)
LONGCAT_AUDIODIT_OUTPUT_DIR = expand_path(
    os.getenv(
        "LONGCAT_AUDIODIT_OUTPUT_DIR",
        os.getenv("TTS_OUTPUT_DIR", str(API_DIR / "tempAudio")),
    )
)

# LongCat clone defaults. Environment variables remain available for deployment
# and canary experiments without changing the WebUI contract.
LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK_DEFAULT = 180
LONGCAT_AUDIODIT_PAUSE_MS_DEFAULT = 250
LONGCAT_AUDIODIT_NFE_DEFAULT = 16
LONGCAT_AUDIODIT_GUIDANCE_STRENGTH_DEFAULT = 4.0
LONGCAT_AUDIODIT_GUIDANCE_METHOD_DEFAULT = "apg"
LONGCAT_AUDIODIT_SEED_DEFAULT = 20260614
LONGCAT_AUDIODIT_DURATION_SCALE_DEFAULT = 1.0
LONGCAT_AUDIODIT_VAE_DTYPE_DEFAULT = "float16"
LONGCAT_AUDIODIT_REQUEST_TIMEOUT_DEFAULT = 900.0

LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK = int(
    os.getenv(
        "LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK",
        str(LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK_DEFAULT),
    )
)
LONGCAT_AUDIODIT_PAUSE_MS = int(
    os.getenv("LONGCAT_AUDIODIT_PAUSE_MS", str(LONGCAT_AUDIODIT_PAUSE_MS_DEFAULT))
)
LONGCAT_AUDIODIT_NFE = int(
    os.getenv("LONGCAT_AUDIODIT_NFE", str(LONGCAT_AUDIODIT_NFE_DEFAULT))
)
LONGCAT_AUDIODIT_GUIDANCE_STRENGTH = float(
    os.getenv(
        "LONGCAT_AUDIODIT_GUIDANCE_STRENGTH",
        str(LONGCAT_AUDIODIT_GUIDANCE_STRENGTH_DEFAULT),
    )
)
LONGCAT_AUDIODIT_GUIDANCE_METHOD = os.getenv(
    "LONGCAT_AUDIODIT_GUIDANCE_METHOD", LONGCAT_AUDIODIT_GUIDANCE_METHOD_DEFAULT
)
LONGCAT_AUDIODIT_SEED = int(
    os.getenv("LONGCAT_AUDIODIT_SEED", str(LONGCAT_AUDIODIT_SEED_DEFAULT))
)
LONGCAT_AUDIODIT_DURATION_SCALE = float(
    os.getenv(
        "LONGCAT_AUDIODIT_DURATION_SCALE",
        str(LONGCAT_AUDIODIT_DURATION_SCALE_DEFAULT),
    )
)
LONGCAT_AUDIODIT_VAE_DTYPE = os.getenv(
    "LONGCAT_AUDIODIT_VAE_DTYPE", LONGCAT_AUDIODIT_VAE_DTYPE_DEFAULT
)
LONGCAT_AUDIODIT_REQUEST_TIMEOUT = float(
    os.getenv(
        "LONGCAT_AUDIODIT_REQUEST_TIMEOUT",
        str(LONGCAT_AUDIODIT_REQUEST_TIMEOUT_DEFAULT),
    )
)

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", str(Path(RUNTIME_CACHE_DIR) / "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(RUNTIME_CACHE_DIR) / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(RUNTIME_CACHE_DIR) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(RUNTIME_CACHE_DIR) / "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for directory in (
    PROMPTS_DIR,
    os.environ["HF_MODULES_CACHE"],
    os.environ["NUMBA_CACHE_DIR"],
    os.environ["MPLCONFIGDIR"],
    os.environ["XDG_CACHE_HOME"],
    LONGCAT_AUDIODIT_WORKER_TMP_DIR,
    os.path.dirname(GPU_LOCK_FILE),
):
    if directory:
        os.makedirs(directory, exist_ok=True)


app = FastAPI(title="Unitale LongCat-AudioDiT-3.5B Voice Clone API")


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
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{digest}{ext}"


def prompt_text_sidecar_path(filename: str) -> str:
    return os.path.join(PROMPTS_DIR, f"{hash_filename(filename)}.prompt.txt")


def load_prompt_text_sidecar(filename: str) -> Optional[str]:
    path = prompt_text_sidecar_path(filename)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        text = file.read().strip()
    return text or None


def save_prompt_text_sidecar(filename: str, prompt_text: Optional[str]) -> None:
    path = prompt_text_sidecar_path(filename)
    normalized = prompt_text.strip() if prompt_text and prompt_text.strip() else None
    if normalized is None:
        if os.path.exists(path):
            os.remove(path)
        return
    with open(path, "w", encoding="utf-8") as file:
        file.write(normalized)


def persist_audio_bytes(audio_bytes: bytes, model_prefix: str, output_dir: str) -> Path:
    if not audio_bytes:
        raise ValueError("cannot persist empty audio")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_fd, temporary_path = tempfile.mkstemp(
        dir=output_directory,
        prefix=f".{model_prefix}_{timestamp}_",
        suffix=".tmp",
    )
    temporary_file = Path(temporary_path)
    output_path = output_directory / f"{temporary_file.name[1:-4]}.wav"
    try:
        with os.fdopen(output_fd, "wb") as destination:
            destination.write(audio_bytes)
        os.replace(temporary_file, output_path)
        return output_path
    except Exception:
        try:
            os.close(output_fd)
        except OSError:
            pass
        try:
            temporary_file.unlink()
        except OSError:
            pass
        try:
            output_path.unlink()
        except OSError:
            pass
        raise


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_synthesis_text(text: str) -> str:
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise HTTPException(status_code=400, detail="text 不能为空。")
    return normalized


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


class CloneSynthesisRequest(BaseModel):
    """Compatibility base for local reference-audio clone endpoints."""

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def reject_style_prompt(cls, value):
        if isinstance(value, dict) and "style_prompt" in value:
            raise ValueError(
                "style_prompt 不适用于 /v2/synthesize；该接口仅用于参考音频克隆。"
            )
        return value


class LongCatAudioDitSynthesizeRequest(CloneSynthesisRequest):
    text: str
    audio_path: str
    prompt_text: Optional[str] = None
    max_chars_per_chunk: Optional[int] = Field(default=None, ge=0)
    pause_ms: Optional[int] = Field(default=None, ge=0)
    nfe: Optional[int] = Field(default=None, ge=2)
    guidance_strength: Optional[float] = Field(default=None, ge=0)
    guidance_method: Optional[Literal["cfg", "apg"]] = None
    seed: Optional[int] = None
    duration_scale: Optional[float] = Field(default=None, gt=0)
    vae_dtype: Optional[Literal["float16", "float32"]] = None


class LongCatAudioDitWorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(
        self, request: LongCatAudioDitSynthesizeRequest
    ) -> dict[str, object]:
        ref_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        prompt_text = normalize_optional_text(request.prompt_text)
        if prompt_text is None:
            prompt_text = load_prompt_text_sidecar(request.audio_path)
        if prompt_text is None:
            raise HTTPException(
                status_code=400,
                detail="LongCat-AudioDiT 克隆需要参考音频的准确 prompt_text。",
            )

        return {
            "operation": "clone",
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "model_path": LONGCAT_AUDIODIT_MODEL_DIR,
            "repo_path": LONGCAT_AUDIODIT_REPO_PATH,
            "tokenizer_path": LONGCAT_AUDIODIT_TOKENIZER_PATH,
            "default_tokenizer_path": LONGCAT_AUDIODIT_TOKENIZER_PATH,
            "max_chars_per_chunk": (
                request.max_chars_per_chunk
                if request.max_chars_per_chunk is not None
                else LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK
            ),
            "pause_ms": (
                request.pause_ms
                if request.pause_ms is not None
                else LONGCAT_AUDIODIT_PAUSE_MS
            ),
            "nfe": request.nfe if request.nfe is not None else LONGCAT_AUDIODIT_NFE,
            "guidance_strength": (
                request.guidance_strength
                if request.guidance_strength is not None
                else LONGCAT_AUDIODIT_GUIDANCE_STRENGTH
            ),
            "guidance_method": (
                request.guidance_method
                if request.guidance_method is not None
                else LONGCAT_AUDIODIT_GUIDANCE_METHOD
            ),
            "seed": request.seed if request.seed is not None else LONGCAT_AUDIODIT_SEED,
            "duration_scale": (
                request.duration_scale
                if request.duration_scale is not None
                else LONGCAT_AUDIODIT_DURATION_SCALE
            ),
            "vae_dtype": request.vae_dtype or LONGCAT_AUDIODIT_VAE_DTYPE,
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def run_worker(self, payload: dict[str, object]) -> bytes:
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 LongCat uv 环境的 Python 解释器。")
        if not os.path.isfile(LONGCAT_AUDIODIT_WORKER_SCRIPT):
            raise RuntimeError(
                f"LongCat worker 脚本不存在: {LONGCAT_AUDIODIT_WORKER_SCRIPT}"
            )
        if not os.path.isdir(LONGCAT_AUDIODIT_WORKER_TMP_DIR):
            os.makedirs(LONGCAT_AUDIODIT_WORKER_TMP_DIR, exist_ok=True)

        request_fd, request_path = tempfile.mkstemp(
            dir=LONGCAT_AUDIODIT_WORKER_TMP_DIR,
            prefix="longcat_audiodit_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=LONGCAT_AUDIODIT_WORKER_TMP_DIR,
            prefix="longcat_audiodit_out_",
            suffix=".wav",
        )
        os.close(request_fd)
        os.close(output_fd)
        process: Optional[subprocess.Popen] = None

        try:
            with open(request_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)

            command = [
                python_executable,
                LONGCAT_AUDIODIT_WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[LongCat-AudioDiT] 启动 uv worker: python={python_executable}")
            started = time.perf_counter()
            worker_env = os.environ.copy()
            worker_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            worker_env.pop("CUDA_MODULE_LOADING", None)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=worker_env,
            )
            try:
                stdout, stderr = process.communicate(
                    timeout=LONGCAT_AUDIODIT_REQUEST_TIMEOUT
                )
            except subprocess.TimeoutExpired:
                terminate_process_group(process, "LongCat-AudioDiT")
                process.communicate()
                raise RuntimeError(
                    f"LongCat-AudioDiT worker 超时（>{LONGCAT_AUDIODIT_REQUEST_TIMEOUT:.0f}s）"
                )

            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(
                "[LongCat-AudioDiT] worker 退出码="
                f"{process.returncode}，耗时 {time.perf_counter() - started:.2f}s"
            )

            if process.returncode != 0:
                output = stderr or stdout
                lines = [line.strip() for line in output.splitlines() if line.strip()]
                detail = " | ".join(lines[-8:]) if lines else "worker 未输出错误信息"
                raise RuntimeError(detail)
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("LongCat-AudioDiT worker 未生成音频文件。")

            with open(output_path, "rb") as file:
                audio_bytes = file.read()
            saved_output_path = persist_audio_bytes(
                audio_bytes,
                "longcat_audiodit",
                LONGCAT_AUDIODIT_OUTPUT_DIR,
            )
            print(f"[LongCat-AudioDiT] 已保存生成音频: {saved_output_path}")
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            terminate_process_group(process, "LongCat-AudioDiT")
            for path in (request_path, output_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass


manager = LongCatAudioDitWorkerManager()


@app.get("/v1/health")
async def health():
    cuda = cuda_status()
    model_required = all(
        os.path.isfile(os.path.join(LONGCAT_AUDIODIT_MODEL_DIR, name))
        for name in ("config.json", "model.safetensors")
    )
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "model_dir": LONGCAT_AUDIODIT_MODEL_DIR,
            "repo_path": LONGCAT_AUDIODIT_REPO_PATH,
            "tokenizer_path": LONGCAT_AUDIODIT_TOKENIZER_PATH,
            "prompts_dir": PROMPTS_DIR,
            "tts_output_dir": LONGCAT_AUDIODIT_OUTPUT_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": LONGCAT_AUDIODIT_WORKER_SCRIPT,
            "worker_tmp_dir": LONGCAT_AUDIODIT_WORKER_TMP_DIR,
        },
        "available": {
            "python": sys.executable,
            "conda": bool(shutil.which("conda")),
            "worker_script": os.path.isfile(LONGCAT_AUDIODIT_WORKER_SCRIPT),
            "model_dir": os.path.isdir(LONGCAT_AUDIODIT_MODEL_DIR),
            "model_required_files": model_required,
            "repo_path": os.path.isdir(LONGCAT_AUDIODIT_REPO_PATH),
            "tokenizer_path": os.path.isdir(LONGCAT_AUDIODIT_TOKENIZER_PATH),
            "torch": module_available("torch"),
            "cuda": cuda["available"],
            "flash_attn": module_available("flash_attn"),
        },
        "cuda": cuda,
        "runtime": {
            "port": API_PORT,
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "model": "LongCat-AudioDiT-3.5B-bf16",
            "model_lifecycle": (
                "one request -> one worker -> explicit CUDA cleanup -> "
                "process exit releases VRAM"
            ),
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": LONGCAT_AUDIODIT_REQUEST_TIMEOUT,
            "sample_rate": 24000,
            "max_chars_per_chunk": LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK,
            "pause_ms": LONGCAT_AUDIODIT_PAUSE_MS,
            "nfe": LONGCAT_AUDIODIT_NFE,
            "guidance_strength": LONGCAT_AUDIODIT_GUIDANCE_STRENGTH,
            "guidance_method": LONGCAT_AUDIODIT_GUIDANCE_METHOD,
            "seed": LONGCAT_AUDIODIT_SEED,
            "duration_scale": LONGCAT_AUDIODIT_DURATION_SCALE,
            "vae_dtype": LONGCAT_AUDIODIT_VAE_DTYPE,
            "clone_contract": (
                "accurate prompt_text + 24 kHz mono prompt_audio; "
                "model max_wav_duration applies"
            ),
            "flash_attention_policy": (
                "not required; official audiodit uses native "
                "PyTorch/Transformers attention"
            ),
        },
        "last_errors": {"longcat_audiodit": manager.last_error},
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    with gpu_runtime_lock("longcat_audiodit/unload"):
        with manager.lock:
            pass
    return JSONResponse({"code": 200, "msg": "LongCat worker 已退出，无常驻模型"})


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...),
    prompt_text: Optional[str] = Form(None),
):
    content = await audio.read()
    save_path = os.path.join(PROMPTS_DIR, hash_filename(full_path))
    with open(save_path, "wb") as file:
        file.write(content)

    normalized_prompt_text = normalize_optional_text(prompt_text)
    save_prompt_text_sidecar(full_path, normalized_prompt_text)
    return {
        "code": 200,
        "msg": "上传成功",
        "filename": full_path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "has_prompt_text": bool(normalized_prompt_text),
    }


@app.get("/v1/check/audio")
async def check_audio_exists(file_name: str):
    audio_path = os.path.join(PROMPTS_DIR, hash_filename(file_name))
    exists = os.path.isfile(audio_path)
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "size_bytes": os.path.getsize(audio_path) if exists else None,
        "sha256": sha256_file(audio_path) if exists else None,
        "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
    }


@app.post("/v2/synthesize")
async def synthesize_v2(request: LongCatAudioDitSynthesizeRequest):
    with gpu_runtime_lock("longcat_audiodit/synthesize"):
        with manager.lock:
            try:
                payload = manager.build_worker_payload(request)
                audio_bytes = manager.run_worker(payload)
                return Response(content=audio_bytes, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as exc:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                wait_after_cuda_release("after LongCat-AudioDiT worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 LongCat-AudioDiT-3.5B uv")
    print("==================================================")
    print(f"[配置] project: {PROJECT_DIR}")
    print(f"[配置] model: {LONGCAT_AUDIODIT_MODEL_DIR}")
    print(f"[配置] official repo: {LONGCAT_AUDIODIT_REPO_PATH}")
    print(f"[配置] tokenizer: {LONGCAT_AUDIODIT_TOKENIZER_PATH}")
    print(f"[配置] worker: {LONGCAT_AUDIODIT_WORKER_SCRIPT}")
    print(f"[配置] port: {API_PORT}")
    print(
        f"[配置] local_files_only={LOCAL_FILES_ONLY}, "
        f"request_timeout={LONGCAT_AUDIODIT_REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
