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
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
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


def optional_expand_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return expand_path(stripped)


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", os.path.join(PROJECT_DIR, "prompts")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", os.path.join(PROJECT_DIR, ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8302"))

LONGCAT_CONDA_ENV = os.getenv("LONGCAT_CONDA_ENV", "longcat_audiodit")
LONGCAT_MODEL_DIR = expand_path(
    os.getenv("LONGCAT_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "meituan-longcat/LongCat-AudioDiT-1B"))
)
LONGCAT_TOKENIZER_PATH = optional_expand_path(
    os.getenv("LONGCAT_TOKENIZER_PATH", os.path.join(HF_MIRROR_DIR, "google/umt5-base"))
)
LONGCAT_REPO_ENV = "LONGCAT_REPO_PATH"
LONGCAT_REPO_PATH = optional_expand_path(os.getenv(LONGCAT_REPO_ENV))
DEFAULT_LONGCAT_REPO_CANDIDATES = (
    Path(PROJECT_DIR) / "vendor/LongCat-AudioDiT",
    Path("/tmp/LongCat-AudioDiT"),
)

LONGCAT_MAX_CHARS_PER_CHUNK = int(os.getenv("LONGCAT_MAX_CHARS_PER_CHUNK", "90"))
LONGCAT_PAUSE_MS = int(os.getenv("LONGCAT_PAUSE_MS", "250"))
LONGCAT_NFE = int(os.getenv("LONGCAT_NFE", "16"))
LONGCAT_GUIDANCE_STRENGTH = float(os.getenv("LONGCAT_GUIDANCE_STRENGTH", "4.0"))
LONGCAT_GUIDANCE_METHOD = os.getenv("LONGCAT_GUIDANCE_METHOD", "apg")
LONGCAT_SEED = int(os.getenv("LONGCAT_SEED", "1024"))
LONGCAT_DURATION_SCALE = float(os.getenv("LONGCAT_DURATION_SCALE", "1.0"))
LONGCAT_VAE_DTYPE = os.getenv("LONGCAT_VAE_DTYPE", "float16")
LONGCAT_REQUEST_TIMEOUT = float(os.getenv("LONGCAT_REQUEST_TIMEOUT", "600"))
LONGCAT_TRIM_LEADING_SILENCE = env_bool("LONGCAT_TRIM_LEADING_SILENCE", True)
LONGCAT_TRIM_LEADING_SILENCE_THRESHOLD_DB = float(os.getenv("LONGCAT_TRIM_LEADING_SILENCE_THRESHOLD_DB", "-42"))
LONGCAT_TRIM_LEADING_SILENCE_MIN_MS = int(os.getenv("LONGCAT_TRIM_LEADING_SILENCE_MIN_MS", "120"))
LONGCAT_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS = int(
    os.getenv("LONGCAT_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS", "30")
)
LONGCAT_TRIM_LEADING_SILENCE_PRE_ROLL_MS = int(os.getenv("LONGCAT_TRIM_LEADING_SILENCE_PRE_ROLL_MS", "40"))
LONGCAT_TRIM_LEADING_SILENCE_MAX_MS = int(os.getenv("LONGCAT_TRIM_LEADING_SILENCE_MAX_MS", "8000"))
LONGCAT_AUTO_PROMPT_TEXT = env_bool("LONGCAT_AUTO_PROMPT_TEXT", True)
LONGCAT_ASR_MODEL_DIR = expand_path(
    os.getenv("LONGCAT_ASR_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "FunAudioLLM/SenseVoiceSmall"))
)
LONGCAT_ASR_DEVICE = os.getenv("LONGCAT_ASR_DEVICE", "cpu")
LONGCAT_ASR_LANGUAGE = os.getenv("LONGCAT_ASR_LANGUAGE", "auto")
LONGCAT_ASR_TIMEOUT = float(os.getenv("LONGCAT_ASR_TIMEOUT", "180"))

LONGCAT_WORKER_SCRIPT = os.path.join(PROJECT_DIR, "longcat_audiodit_worker.py")
LONGCAT_PROMPT_TRANSCRIBE_SCRIPT = os.path.join(PROJECT_DIR, "longcat_prompt_transcribe_worker.py")
LONGCAT_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "longcat_worker")

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
os.makedirs(LONGCAT_WORKER_TMP_DIR, exist_ok=True)
gpu_lock_dir = os.path.dirname(GPU_LOCK_FILE)
if gpu_lock_dir:
    os.makedirs(gpu_lock_dir, exist_ok=True)

app = FastAPI(title="Unitale LongCat AudioDiT API")


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


def unique_resolved_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(Path(resolved))
    return unique_paths


def iter_longcat_repo_candidates(explicit_repo_path: Optional[str] = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_repo_path:
        candidates.append(Path(explicit_repo_path))
    if LONGCAT_REPO_PATH:
        candidates.append(Path(LONGCAT_REPO_PATH))

    env_repo_path = os.environ.get(LONGCAT_REPO_ENV)
    if env_repo_path:
        candidates.append(Path(env_repo_path))

    for path_text in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if path_text:
            candidates.append(Path(path_text))

    for path_text in sys.path:
        if path_text:
            candidates.append(Path(path_text))

    candidates.extend(DEFAULT_LONGCAT_REPO_CANDIDATES)
    return unique_resolved_paths(candidates)


def resolve_longcat_repo_path(explicit_repo_path: Optional[str] = None) -> Optional[str]:
    for candidate in iter_longcat_repo_candidates(explicit_repo_path):
        resolved = candidate.expanduser().resolve()
        if (resolved / "audiodit").is_dir():
            return str(resolved)
    return None


def worker_error_excerpt(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "LongCat worker 未输出错误信息。"
    return " | ".join(lines[-8:])


class LongCatSynthesizeRequest(CloneSynthesisRequest):

    text: str
    audio_path: str
    prompt_text: Optional[str] = None
    tokenizer_path: Optional[str] = None
    repo_path: Optional[str] = None
    max_chars_per_chunk: Optional[int] = None
    pause_ms: Optional[int] = None
    nfe: Optional[int] = None
    guidance_strength: Optional[float] = None
    guidance_method: Optional[str] = None
    seed: Optional[int] = None
    duration_scale: Optional[float] = None
    vae_dtype: Optional[str] = None
    emo_text: Optional[str] = None
    emo_vector: Optional[List[float]] = None


class LongCatWorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None
        self.last_asr_error: Optional[str] = None

    def transcribe_prompt_audio(self, audio_path: str, ref_audio_path: str) -> str:
        cached_prompt_text = load_prompt_text_sidecar(audio_path)
        if cached_prompt_text:
            return cached_prompt_text

        conda_exe = resolve_conda_executable()
        if not conda_exe:
            raise RuntimeError("未找到 conda 命令，无法自动转写 LongCat 参考音频。")
        if not os.path.isfile(LONGCAT_PROMPT_TRANSCRIBE_SCRIPT):
            raise RuntimeError(f"LongCat prompt 转写脚本不存在: {LONGCAT_PROMPT_TRANSCRIBE_SCRIPT}")
        if not os.path.isdir(LONGCAT_ASR_MODEL_DIR):
            raise RuntimeError(f"LongCat prompt 转写模型目录不存在: {LONGCAT_ASR_MODEL_DIR}")

        result_fd, result_path = tempfile.mkstemp(dir=LONGCAT_WORKER_TMP_DIR, prefix="longcat_prompt_", suffix=".json")
        os.close(result_fd)

        try:
            command = [
                conda_exe,
                "run",
                "--no-capture-output",
                "-n",
                LONGCAT_CONDA_ENV,
                "python",
                LONGCAT_PROMPT_TRANSCRIBE_SCRIPT,
                "--input-audio",
                ref_audio_path,
                "--output-json",
                result_path,
                "--model-dir",
                LONGCAT_ASR_MODEL_DIR,
                "--device",
                LONGCAT_ASR_DEVICE,
                "--language",
                LONGCAT_ASR_LANGUAGE,
            ]
            print(
                f"[LongCat ASR] 自动转写参考音频: audio={audio_path}, "
                f"model={LONGCAT_ASR_MODEL_DIR}, device={LONGCAT_ASR_DEVICE}"
            )
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
                stdout, stderr = proc.communicate(timeout=LONGCAT_ASR_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                stdout, stderr = proc.communicate(timeout=10)
                raise RuntimeError(f"LongCat prompt 自动转写超时（>{LONGCAT_ASR_TIMEOUT:.0f}s）")

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[LongCat ASR] 退出码={proc.returncode}，耗时 {elapsed:.2f}s")

            if proc.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(result_path) or os.path.getsize(result_path) == 0:
                raise RuntimeError("LongCat prompt 自动转写未生成结果文件。")

            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            prompt_text = str(result.get("text") or "").strip()
            if not prompt_text:
                raise RuntimeError("LongCat prompt 自动转写结果为空。")

            save_prompt_text_sidecar(audio_path, prompt_text)
            self.last_asr_error = None
            return prompt_text
        except Exception as exc:
            self.last_asr_error = str(exc)
            raise
        finally:
            try:
                if os.path.exists(result_path):
                    os.remove(result_path)
            except Exception:
                pass

    def resolve_prompt_text(self, audio_path: str, ref_audio_path: str, prompt_text: Optional[str]) -> str:
        normalized = prompt_text.strip() if prompt_text and prompt_text.strip() else None
        if normalized:
            save_prompt_text_sidecar(audio_path, normalized)
            return normalized

        cached_prompt_text = load_prompt_text_sidecar(audio_path)
        if cached_prompt_text:
            return cached_prompt_text

        if not LONGCAT_AUTO_PROMPT_TEXT:
            raise HTTPException(
                status_code=422,
                detail=(
                    "LongCat-AudioDiT 克隆需要参考音频转写文本。"
                    "请在 /v1/upload_audio 上传时传入 prompt_text，或开启 LONGCAT_AUTO_PROMPT_TEXT 自动转写。"
                ),
            )

        try:
            return self.transcribe_prompt_audio(audio_path, ref_audio_path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LongCat 参考音频自动转写失败: {exc}") from exc

    def build_worker_payload(self, request: LongCatSynthesizeRequest) -> dict:
        ref_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        prompt_text = self.resolve_prompt_text(request.audio_path, ref_audio_path, request.prompt_text)

        explicit_repo_path = request.repo_path.strip() if request.repo_path and request.repo_path.strip() else None
        repo_path = resolve_longcat_repo_path(explicit_repo_path) or explicit_repo_path
        tokenizer_path = request.tokenizer_path.strip() if request.tokenizer_path and request.tokenizer_path.strip() else LONGCAT_TOKENIZER_PATH

        return {
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "model_path": LONGCAT_MODEL_DIR,
            "tokenizer_path": tokenizer_path,
            "repo_path": repo_path,
            "max_chars_per_chunk": request.max_chars_per_chunk if request.max_chars_per_chunk is not None else LONGCAT_MAX_CHARS_PER_CHUNK,
            "pause_ms": request.pause_ms if request.pause_ms is not None else LONGCAT_PAUSE_MS,
            "nfe": request.nfe if request.nfe is not None else LONGCAT_NFE,
            "guidance_strength": request.guidance_strength if request.guidance_strength is not None else LONGCAT_GUIDANCE_STRENGTH,
            "guidance_method": request.guidance_method or LONGCAT_GUIDANCE_METHOD,
            "seed": request.seed if request.seed is not None else LONGCAT_SEED,
            "duration_scale": request.duration_scale if request.duration_scale is not None else LONGCAT_DURATION_SCALE,
            "vae_dtype": request.vae_dtype or LONGCAT_VAE_DTYPE,
            "trim_leading_silence": LONGCAT_TRIM_LEADING_SILENCE,
            "trim_leading_silence_threshold_db": LONGCAT_TRIM_LEADING_SILENCE_THRESHOLD_DB,
            "trim_leading_silence_min_ms": LONGCAT_TRIM_LEADING_SILENCE_MIN_MS,
            "trim_leading_silence_analysis_window_ms": LONGCAT_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS,
            "trim_leading_silence_pre_roll_ms": LONGCAT_TRIM_LEADING_SILENCE_PRE_ROLL_MS,
            "trim_leading_silence_max_ms": LONGCAT_TRIM_LEADING_SILENCE_MAX_MS,
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def run_worker(self, payload: dict) -> bytes:
        conda_exe = resolve_conda_executable()
        if not conda_exe:
            raise RuntimeError("未找到 conda 命令，无法调用 LongCat worker。")
        if not os.path.isfile(LONGCAT_WORKER_SCRIPT):
            raise RuntimeError(f"LongCat worker 脚本不存在: {LONGCAT_WORKER_SCRIPT}")
        if not os.path.isdir(LONGCAT_MODEL_DIR):
            raise RuntimeError(f"LongCat 模型目录不存在: {LONGCAT_MODEL_DIR}")

        request_fd, request_path = tempfile.mkstemp(dir=LONGCAT_WORKER_TMP_DIR, prefix="longcat_req_", suffix=".json")
        output_fd, output_path = tempfile.mkstemp(dir=LONGCAT_WORKER_TMP_DIR, prefix="longcat_out_", suffix=".wav")
        os.close(request_fd)
        os.close(output_fd)

        try:
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            env = os.environ.copy()
            repo_path = payload.get("repo_path")
            if repo_path:
                env[LONGCAT_REPO_ENV] = repo_path
                current_pythonpath = env.get("PYTHONPATH")
                env["PYTHONPATH"] = repo_path if not current_pythonpath else repo_path + os.pathsep + current_pythonpath

            command = [
                conda_exe,
                "run",
                "--no-capture-output",
                "-n",
                LONGCAT_CONDA_ENV,
                "python",
                LONGCAT_WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[LongCat] 启动 worker: env={LONGCAT_CONDA_ENV}, repo={repo_path or '未解析'}")
            started = time.perf_counter()
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=LONGCAT_REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                stdout, stderr = proc.communicate(timeout=10)
                raise RuntimeError(f"LongCat worker 超时（>{LONGCAT_REQUEST_TIMEOUT:.0f}s）")

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[LongCat] worker 退出码={proc.returncode}，耗时 {elapsed:.2f}s")

            if proc.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("LongCat worker 未生成音频文件。")

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


manager = LongCatWorkerManager()


@app.get("/v1/health")
async def health():
    resolved_repo_path = resolve_longcat_repo_path()
    repo_candidates = [str(path) for path in iter_longcat_repo_candidates()]
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "longcat_model_dir": LONGCAT_MODEL_DIR,
            "longcat_tokenizer_path": LONGCAT_TOKENIZER_PATH,
            "longcat_repo_path": resolved_repo_path,
            "prompts_dir": PROMPTS_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": LONGCAT_WORKER_SCRIPT,
            "prompt_transcribe_script": LONGCAT_PROMPT_TRANSCRIBE_SCRIPT,
            "worker_tmp_dir": LONGCAT_WORKER_TMP_DIR,
        },
        "available": {
            "conda": bool(resolve_conda_executable()),
            "worker_script": os.path.isfile(LONGCAT_WORKER_SCRIPT),
            "prompt_transcribe_script": os.path.isfile(LONGCAT_PROMPT_TRANSCRIBE_SCRIPT),
            "longcat_model_dir": os.path.isdir(LONGCAT_MODEL_DIR),
            "longcat_tokenizer_path": bool(
                LONGCAT_TOKENIZER_PATH is None or os.path.exists(LONGCAT_TOKENIZER_PATH)
            ),
            "longcat_repo_path": bool(resolved_repo_path),
            "longcat_asr_model_dir": os.path.isdir(LONGCAT_ASR_MODEL_DIR),
            "torch": module_available("torch"),
            "cuda": cuda_status()["available"],
        },
        "cuda": cuda_status(),
        "runtime": {
            "worker_env": LONGCAT_CONDA_ENV,
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": LONGCAT_REQUEST_TIMEOUT,
            "max_chars_per_chunk": LONGCAT_MAX_CHARS_PER_CHUNK,
            "pause_ms": LONGCAT_PAUSE_MS,
            "nfe": LONGCAT_NFE,
            "guidance_strength": LONGCAT_GUIDANCE_STRENGTH,
            "guidance_method": LONGCAT_GUIDANCE_METHOD,
            "seed": LONGCAT_SEED,
            "duration_scale": LONGCAT_DURATION_SCALE,
            "vae_dtype": LONGCAT_VAE_DTYPE,
            "trim_leading_silence": LONGCAT_TRIM_LEADING_SILENCE,
            "trim_leading_silence_threshold_db": LONGCAT_TRIM_LEADING_SILENCE_THRESHOLD_DB,
            "trim_leading_silence_min_ms": LONGCAT_TRIM_LEADING_SILENCE_MIN_MS,
            "trim_leading_silence_analysis_window_ms": LONGCAT_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS,
            "trim_leading_silence_pre_roll_ms": LONGCAT_TRIM_LEADING_SILENCE_PRE_ROLL_MS,
            "trim_leading_silence_max_ms": LONGCAT_TRIM_LEADING_SILENCE_MAX_MS,
            "auto_prompt_text": LONGCAT_AUTO_PROMPT_TEXT,
            "asr_model_dir": LONGCAT_ASR_MODEL_DIR,
            "asr_device": LONGCAT_ASR_DEVICE,
            "asr_language": LONGCAT_ASR_LANGUAGE,
            "asr_timeout": LONGCAT_ASR_TIMEOUT,
            "repo_candidates": repo_candidates,
        },
        "last_errors": {
            "longcat": manager.last_error,
            "longcat_asr": manager.last_asr_error,
        },
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    clear_cuda_cache("longcat api internal unload")
    wait_after_cuda_release("longcat api internal unload")
    return JSONResponse({"code": 200, "msg": "longcat wrapper 无常驻模型，已完成显存清理等待"})


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

    prompt_text_source = "request" if normalized_prompt_text else "none"
    auto_prompt_text_error = None
    if normalized_prompt_text is None and LONGCAT_AUTO_PROMPT_TEXT:
        try:
            normalized_prompt_text = manager.transcribe_prompt_audio(full_path, save_path)
            prompt_text_source = "auto"
        except Exception as exc:
            auto_prompt_text_error = str(exc)

    return {
        "code": 200,
        "msg": "上传成功",
        "filename": full_path,
        "has_prompt_text": bool(normalized_prompt_text),
        "prompt_text_source": prompt_text_source,
        "prompt_text": normalized_prompt_text,
        "auto_prompt_text_error": auto_prompt_text_error,
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
async def synthesize_v2(request: LongCatSynthesizeRequest):
    with gpu_runtime_lock("longcat/synthesize"):
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
                clear_cuda_cache("after longcat worker")
                wait_after_cuda_release("after longcat worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 LongCat-AudioDiT")
    print("==================================================")
    print(f"[配置] LongCat worker env: {LONGCAT_CONDA_ENV}")
    print(f"[配置] LongCat 模型目录: {LONGCAT_MODEL_DIR}")
    print(f"[配置] LongCat tokenizer: {LONGCAT_TOKENIZER_PATH or 'model.config.text_encoder_model'}")
    print(f"[配置] LongCat repo: {resolve_longcat_repo_path() or LONGCAT_REPO_PATH or '未找到'}")
    print(f"[配置] prompts 目录: {PROMPTS_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] worker 脚本: {LONGCAT_WORKER_SCRIPT}")
    print(
        f"[配置] auto_prompt_text={LONGCAT_AUTO_PROMPT_TEXT}, "
        f"asr_model={LONGCAT_ASR_MODEL_DIR}, asr_device={LONGCAT_ASR_DEVICE}, "
        f"asr_language={LONGCAT_ASR_LANGUAGE}, asr_timeout={LONGCAT_ASR_TIMEOUT}"
    )
    print(
        f"[配置] trim_leading_silence={LONGCAT_TRIM_LEADING_SILENCE}, "
        f"threshold_db={LONGCAT_TRIM_LEADING_SILENCE_THRESHOLD_DB}, "
        f"min_ms={LONGCAT_TRIM_LEADING_SILENCE_MIN_MS}, "
        f"pre_roll_ms={LONGCAT_TRIM_LEADING_SILENCE_PRE_ROLL_MS}, "
        f"max_ms={LONGCAT_TRIM_LEADING_SILENCE_MAX_MS}"
    )
    print(
        f"[配置] nfe={LONGCAT_NFE}, guidance_method={LONGCAT_GUIDANCE_METHOD}, "
        f"guidance_strength={LONGCAT_GUIDANCE_STRENGTH}, duration_scale={LONGCAT_DURATION_SCALE}, "
        f"vae_dtype={LONGCAT_VAE_DTYPE}"
    )
    print(
        f"[配置] max_chars_per_chunk={LONGCAT_MAX_CHARS_PER_CHUNK}, "
        f"pause_ms={LONGCAT_PAUSE_MS}, seed={LONGCAT_SEED}"
    )
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={LONGCAT_REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
