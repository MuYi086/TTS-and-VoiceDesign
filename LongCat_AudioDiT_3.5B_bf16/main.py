"""独立 uv HTTP 服务，提供 LongCat-AudioDiT-3.5B 语音克隆。

本模块负责 LongCat API，并使用当前 uv Python 启动 worker.py。旧的 Conda
API/worker 实现已移除，uv 迁移已经通过真实 GPU 和 HTTP 金丝雀验证。
"""

from __future__ import annotations

# 学习入口：LongCat API 只管理参考音频、GPU 锁和 uv worker，不在父进程加载模型。
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from unitale_runtime import (
    AudioReferenceStore,
    AudioUploadError,
    GpuLockTimeoutError,
    StagedUpload,
    stage_audio_upload,
)
from unitale_runtime import (
    gpu_runtime_lock as shared_gpu_runtime_lock,
)
from unitale_runtime.storage import sha256_file

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


def expand_path(path: str | os.PathLike[str]) -> str:
    """展开环境变量和用户目录，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def normalize_optional_text(value: str | None) -> str | None:
    """把可选文本清理成字符串或 ``None``。"""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


STORAGE_DIR = Path(expand_path(os.getenv("STORAGE_DIR", str(REPOSITORY_DIR / "storage"))))
CLONE_STORAGE_DIR = Path(expand_path(os.getenv("CLONE_STORAGE_DIR", str(STORAGE_DIR / "clone"))))
TIMBRE_STORAGE_DIR = Path(expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(STORAGE_DIR / "timbre"))))
TIMBRE_REFERENCE_DIR = TIMBRE_STORAGE_DIR / ".references"
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", str(CLONE_STORAGE_DIR)))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", str(STORAGE_DIR / ".cache/runtime")))
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", str(Path(RUNTIME_CACHE_DIR) / "gpu-runtime.lock"))
)
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8323"))

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
        str(CLONE_STORAGE_DIR),
    )
)

# LongCat 克隆默认值。环境变量仍可用于部署和金丝雀实验，但不会改变 WebUI 契约。
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
LONGCAT_AUDIODIT_NFE = int(os.getenv("LONGCAT_AUDIODIT_NFE", str(LONGCAT_AUDIODIT_NFE_DEFAULT)))
LONGCAT_AUDIODIT_GUIDANCE_STRENGTH = float(
    os.getenv(
        "LONGCAT_AUDIODIT_GUIDANCE_STRENGTH",
        str(LONGCAT_AUDIODIT_GUIDANCE_STRENGTH_DEFAULT),
    )
)
LONGCAT_AUDIODIT_GUIDANCE_METHOD = os.getenv(
    "LONGCAT_AUDIODIT_GUIDANCE_METHOD", LONGCAT_AUDIODIT_GUIDANCE_METHOD_DEFAULT
)
LONGCAT_AUDIODIT_SEED = int(os.getenv("LONGCAT_AUDIODIT_SEED", str(LONGCAT_AUDIODIT_SEED_DEFAULT)))
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
    TIMBRE_STORAGE_DIR,
    TIMBRE_REFERENCE_DIR,
    os.environ["HF_MODULES_CACHE"],
    os.environ["NUMBA_CACHE_DIR"],
    os.environ["MPLCONFIGDIR"],
    os.environ["XDG_CACHE_HOME"],
    LONGCAT_AUDIODIT_WORKER_TMP_DIR,
    os.path.dirname(GPU_LOCK_FILE),
):
    if directory:
        os.makedirs(directory, exist_ok=True)

reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)


app = FastAPI(title="Unitale LongCat-AudioDiT-3.5B Voice Clone API")


class ForceCORS(BaseHTTPMiddleware):
    """为本地 WebUI 请求提供跨域响应和预检处理。"""

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
    """把 WebUI 逻辑路径哈希成安全的本地文件名。"""
    return reference_store.clone_path(filename).name


def clone_prompt_audio_path(filename: str) -> str:
    return str(reference_store.clone_path(filename))


def timbre_reference_map_path(filename: str) -> str:
    return str(reference_store.reference_path(filename))


def prompt_audio_path(filename: str) -> str:
    """解析克隆上传，或解析只保存在音色目录中的设计音频。"""
    return str(reference_store.prompt_audio_path(filename))


def prompt_text_sidecar_path(filename: str) -> str:
    return str(reference_store.prompt_sidecar_path(filename))


def load_prompt_text_sidecar(filename: str) -> str | None:
    return reference_store.load_prompt_text(filename)


def store_uploaded_audio(
    staged: StagedUpload,
    full_path: str,
    prompt_text: str | None,
) -> dict[str, object]:
    """原子提交流式上传，并为设计音色创建相对引用。"""
    return reference_store.commit_staged_upload(staged, full_path, prompt_text)


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


def normalize_synthesis_text(text: str) -> str:
    """去除 Markdown 标题标记，减少 WebUI 文本被误读的机会。"""
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", (text or "").strip())
    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", normalized)
    if not normalized:
        raise HTTPException(status_code=400, detail="text 不能为空。")
    return normalized


def gpu_runtime_lock(label: str):
    """通过共享文件锁串行化 LongCat 的 GPU 推理。"""
    return shared_gpu_runtime_lock(GPU_LOCK_FILE, label)


def wait_after_cuda_release(label: str = "") -> None:
    """worker 退出后等待显存释放，再释放共享 GPU 锁。"""
    if CUDA_RELEASE_DELAY <= 0:
        return
    if label:
        print(f"[CUDA] 等待 {CUDA_RELEASE_DELAY:.1f}s 释放显存: {label}")
    time.sleep(CUDA_RELEASE_DELAY)


class CloneSynthesisRequest(BaseModel):
    """本地参考音频克隆接口共用的兼容请求基类。"""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_style_prompt(cls, value):
        if isinstance(value, dict) and "style_prompt" in value:
            raise ValueError("style_prompt 不适用于 /v1/longCat/clone；该接口仅用于参考音频克隆。")
        return value


class LongCatAudioDitSynthesizeRequest(CloneSynthesisRequest):
    text: str = Field(min_length=1, max_length=12_000)
    audio_path: str = Field(min_length=1, max_length=1_024)
    backend: Literal["longcat-audiodit"] | None = None
    prompt_text: str | None = Field(default=None, max_length=12_000)
    max_chars_per_chunk: int | None = Field(default=None, ge=0, le=2_000)
    pause_ms: int | None = Field(default=None, ge=0, le=10_000)
    nfe: int | None = Field(default=None, ge=2, le=200)
    guidance_strength: float | None = Field(default=None, ge=0, le=20)
    guidance_method: Literal["cfg", "apg"] | None = None
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    duration_scale: float | None = Field(default=None, gt=0, le=10)
    vae_dtype: Literal["float16", "float32"] | None = None


class LongCatAudioDitWorkerManager:
    """管理 LongCat 参考音频解析、worker 启动和临时文件清理。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: str | None = None

    def build_worker_payload(self, request: LongCatAudioDitSynthesizeRequest) -> dict[str, object]:
        """把克隆请求和服务默认值转换为 worker JSON。"""
        ref_audio_path = prompt_audio_path(request.audio_path)
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
                request.pause_ms if request.pause_ms is not None else LONGCAT_AUDIODIT_PAUSE_MS
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
        """在 LongCat uv 项目中执行一次隔离推理并读取 WAV。"""
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError("未找到 LongCat uv 环境的 Python 解释器。")
        if not os.path.isfile(LONGCAT_AUDIODIT_WORKER_SCRIPT):
            raise RuntimeError(f"LongCat worker 脚本不存在: {LONGCAT_AUDIODIT_WORKER_SCRIPT}")
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
        process: subprocess.Popen | None = None

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
                stdout, stderr = process.communicate(timeout=LONGCAT_AUDIODIT_REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(process, "LongCat-AudioDiT")
                process.communicate()
                raise RuntimeError(
                    f"LongCat-AudioDiT worker 超时（>{LONGCAT_AUDIODIT_REQUEST_TIMEOUT:.0f}s）"
                ) from exc

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
def health():
    """返回 LongCat 模型、源码、worker 和 GPU 的就绪信息。"""
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
                "one request -> one worker -> explicit CUDA cleanup -> process exit releases VRAM"
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
                "accurate prompt_text + 24 kHz mono prompt_audio; model max_wav_duration applies"
            ),
            "flash_attention_policy": (
                "not required; official audiodit uses native PyTorch/Transformers attention"
            ),
        },
        "last_errors": {"longcat_audiodit": manager.last_error},
    }


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...),
    prompt_text: str | None = Form(None),
):
    """在线程池中保存克隆参考音频和可选文本 sidecar。"""
    try:
        staged = await stage_audio_upload(audio, Path(RUNTIME_CACHE_DIR) / "uploads")
        return await run_in_threadpool(store_uploaded_audio, staged, full_path, prompt_text)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    """检查逻辑参考路径是否已保存，并返回 sidecar 状态。"""
    audio_path = prompt_audio_path(file_name)
    exists = os.path.isfile(audio_path)
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "size_bytes": os.path.getsize(audio_path) if exists else None,
        "sha256": sha256_file(audio_path) if exists else None,
        "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
    }


@app.post("/v1/longCat/clone")
def synthesize_v2(request: LongCatAudioDitSynthesizeRequest):
    """串行执行 LongCat 克隆并返回生成 WAV。"""
    try:
        payload = manager.build_worker_payload(request)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("LongCat-AudioDiT request preflight failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        with gpu_runtime_lock("longcat_audiodit/synthesize"):
            with manager.lock:
                try:
                    audio_bytes = manager.run_worker(payload)
                    saved_output_path = persist_audio_bytes(
                        audio_bytes,
                        "longcat_audiodit",
                        LONGCAT_AUDIODIT_OUTPUT_DIR,
                    )
                    print(f"[LongCat-AudioDiT] 已保存生成音频: {saved_output_path}")
                    return Response(content=audio_bytes, media_type="audio/wav")
                except HTTPException:
                    raise
                except Exception as exc:
                    LOGGER.exception("LongCat-AudioDiT request failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                finally:
                    wait_after_cuda_release("after LongCat-AudioDiT worker")
    except GpuLockTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
