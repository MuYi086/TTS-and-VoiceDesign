"""Qwen3-TTS 本地语音克隆服务。

HTTP 进程只负责校验请求、管理参考音频和启动一次性 worker；模型及
CUDA 上下文始终留在独立子进程中，便于请求结束后彻底释放显存。
"""

from __future__ import annotations

# 学习入口：上传参考音频后，API 解析 prompt，持有 GPU 锁，再启动一次性 worker。
import fcntl
import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from audio_output import persist_audio_bytes
from gpu_runtime import cuda_status, terminate_process_group
from synthesis_request import CloneSynthesisRequest

LOGGER = logging.getLogger(__name__)

# ==========================================
# 0. 系统配置
# ==========================================
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SERVICE_DIR)


def env_bool(name: str, default: bool = False) -> bool:
    """解析服务启动所需的布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_optional_text(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        value = default
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def env_optional_float(name: str, default: float | None = None) -> float | None:
    value = env_optional_text(name)
    return float(value) if value is not None else default


def expand_path(path: str) -> str:
    """展开环境变量和用户目录，返回绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def normalize_optional_text(value: str | None) -> str | None:
    """把可选文本统一成去首尾空白或 ``None``。"""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", os.path.join(PROJECT_DIR, "storage")))
CLONE_STORAGE_DIR = expand_path(os.getenv("CLONE_STORAGE_DIR", os.path.join(STORAGE_DIR, "clone")))
TIMBRE_STORAGE_DIR = expand_path(
    os.getenv("TIMBRE_STORAGE_DIR", os.path.join(STORAGE_DIR, "timbre"))
)
TIMBRE_REFERENCE_DIR = os.path.join(TIMBRE_STORAGE_DIR, ".references")
HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", CLONE_STORAGE_DIR))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", os.path.join(STORAGE_DIR, ".cache/runtime"))
)
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock"))
)
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8321"))

QWEN3_TTS_MODEL_DIR = expand_path(
    os.getenv("QWEN3_TTS_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
)
# ============================================================================
# Qwen3-TTS 克隆调试默认值
#
# 调试参考音频克隆效果时，优先直接修改下面带 _DEFAULT 后缀的值，然后
# 重启服务即可生效。环境变量仍可覆盖默认值，方便部署时统一配置。
# ============================================================================
# 推理设备映射：通常使用 cuda:0；多卡部署时可按模型加载方式调整。
QWEN3_TTS_DEVICE_MAP_DEFAULT = "cuda:0"
# 模型计算精度：auto 会根据设备自动选择，显存或兼容性异常时可显式指定。
QWEN3_TTS_DTYPE_DEFAULT = "auto"
# 参考音频对应的语言；Qwen3-TTS 使用官方语言名称，如 Chinese、English。
QWEN3_TTS_LANGUAGE_DEFAULT = "Chinese"
# 最大生成 token 数：过小会截断长语音，过大会增加耗时和显存占用。
QWEN3_TTS_MAX_NEW_TOKENS_DEFAULT = 2048
# 采样 top-p；None 表示不向模型额外传入该参数，便于使用模型默认值。
QWEN3_TTS_TOP_P_DEFAULT: float | None = None
# 采样温度；None 表示不向模型额外传入该参数，便于使用模型默认值。
QWEN3_TTS_TEMPERATURE_DEFAULT: float | None = None
# 注意力实现：auto 由 transformers 选择，也可按环境指定 eager 或 flash_attention_2。
QWEN3_TTS_ATTN_IMPLEMENTATION_DEFAULT = "auto"
# 是否强制使用仅音色向量克隆；有准确参考文本时通常保持 False。
QWEN3_TTS_X_VECTOR_ONLY_DEFAULT = False
# 文本分片字符数：0 表示不分片；分片可降低显存压力，但会插入停顿。
QWEN3_TTS_MAX_CHARS_PER_CHUNK_DEFAULT = 120
# 分片之间的停顿时长，单位为毫秒；仅在发生分片时生效。
QWEN3_TTS_PAUSE_MS_DEFAULT = 250
# 是否裁剪生成音频开头的静音；关闭可保留模型原始前导空间。
QWEN3_TTS_TRIM_LEADING_SILENCE_DEFAULT = True
# 判定静音的 RMS 阈值，单位为 dB；数值越低越不容易误裁正常弱音。
QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB_DEFAULT = -42.0
# 至少达到该时长才认为是需要裁剪的前导静音，单位为毫秒。
QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS_DEFAULT = 120
# 静音分析窗口，单位为毫秒；窗口越大越平滑，但定位会更粗。
QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS_DEFAULT = 30
# 裁剪时保留的前滚时间，单位为毫秒，避免切掉起始辅音。
QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS_DEFAULT = 40
# 单次最多裁剪的前导静音，单位为毫秒，避免异常音频被过度裁剪。
QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS_DEFAULT = 8000
# 单次请求超时时间，单位为秒；包含 worker 启动和完整合成。
QWEN3_TTS_REQUEST_TIMEOUT_DEFAULT = 600.0

QWEN3_TTS_DEVICE_MAP = os.getenv("QWEN3_TTS_DEVICE_MAP", QWEN3_TTS_DEVICE_MAP_DEFAULT)
QWEN3_TTS_DTYPE = os.getenv("QWEN3_TTS_DTYPE", QWEN3_TTS_DTYPE_DEFAULT)
QWEN3_TTS_LANGUAGE = env_optional_text("QWEN3_TTS_LANGUAGE", QWEN3_TTS_LANGUAGE_DEFAULT)
QWEN3_TTS_MAX_NEW_TOKENS = int(
    os.getenv("QWEN3_TTS_MAX_NEW_TOKENS", str(QWEN3_TTS_MAX_NEW_TOKENS_DEFAULT))
)
QWEN3_TTS_TOP_P = env_optional_float("QWEN3_TTS_TOP_P", QWEN3_TTS_TOP_P_DEFAULT)
QWEN3_TTS_TEMPERATURE = env_optional_float("QWEN3_TTS_TEMPERATURE", QWEN3_TTS_TEMPERATURE_DEFAULT)
QWEN3_TTS_ATTN_IMPLEMENTATION = os.getenv(
    "QWEN3_TTS_ATTN_IMPLEMENTATION", QWEN3_TTS_ATTN_IMPLEMENTATION_DEFAULT
)
QWEN3_TTS_X_VECTOR_ONLY = env_bool("QWEN3_TTS_X_VECTOR_ONLY", QWEN3_TTS_X_VECTOR_ONLY_DEFAULT)
QWEN3_TTS_MAX_CHARS_PER_CHUNK = int(
    os.getenv(
        "QWEN3_TTS_MAX_CHARS_PER_CHUNK",
        str(QWEN3_TTS_MAX_CHARS_PER_CHUNK_DEFAULT),
    )
)
QWEN3_TTS_PAUSE_MS = int(os.getenv("QWEN3_TTS_PAUSE_MS", str(QWEN3_TTS_PAUSE_MS_DEFAULT)))
QWEN3_TTS_TRIM_LEADING_SILENCE = env_bool(
    "QWEN3_TTS_TRIM_LEADING_SILENCE", QWEN3_TTS_TRIM_LEADING_SILENCE_DEFAULT
)
QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB = float(
    os.getenv(
        "QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB",
        str(QWEN3_TTS_TRIM_LEADING_SILENCE_THRESHOLD_DB_DEFAULT),
    )
)
QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS = int(
    os.getenv(
        "QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS",
        str(QWEN3_TTS_TRIM_LEADING_SILENCE_MIN_MS_DEFAULT),
    )
)
QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS = int(
    os.getenv(
        "QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS",
        str(QWEN3_TTS_TRIM_LEADING_SILENCE_ANALYSIS_WINDOW_MS_DEFAULT),
    )
)
QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS = int(
    os.getenv(
        "QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS",
        str(QWEN3_TTS_TRIM_LEADING_SILENCE_PRE_ROLL_MS_DEFAULT),
    )
)
QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS = int(
    os.getenv(
        "QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS",
        str(QWEN3_TTS_TRIM_LEADING_SILENCE_MAX_MS_DEFAULT),
    )
)
QWEN3_TTS_REQUEST_TIMEOUT = float(
    os.getenv("QWEN3_TTS_REQUEST_TIMEOUT", str(QWEN3_TTS_REQUEST_TIMEOUT_DEFAULT))
)
QWEN3_TTS_WORKER_SCRIPT = os.path.join(SERVICE_DIR, "worker.py")
QWEN3_TTS_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "qwen3_tts_worker")
QWEN3_TTS_OUTPUT_DIR = expand_path(
    os.getenv(
        "QWEN3_TTS_OUTPUT_DIR",
        CLONE_STORAGE_DIR,
    )
)
QWEN3_TTS_USE_QWEN_LIBS = env_bool("QWEN3_TTS_USE_QWEN_LIBS", False)
QWEN_LIBS_PATH = expand_path(os.getenv("QWEN_LIBS", os.path.join(SERVICE_DIR, "vendor/qwen_libs")))

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(RUNTIME_CACHE_DIR, "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(RUNTIME_CACHE_DIR, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RUNTIME_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RUNTIME_CACHE_DIR, "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(TIMBRE_STORAGE_DIR, exist_ok=True)
os.makedirs(TIMBRE_REFERENCE_DIR, exist_ok=True)
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
    """为本地 WebUI 请求添加跨域响应头。"""
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


def clone_prompt_audio_path(filename: str) -> str:
    return os.path.join(PROMPTS_DIR, hash_filename(filename))


def timbre_reference_map_path(filename: str) -> str:
    return os.path.join(TIMBRE_REFERENCE_DIR, f"{hash_filename(filename)}.path")


def prompt_audio_path(filename: str) -> str:
    """解析克隆上传，或解析克隆预览引用的音色设计资产。"""
    clone_path = clone_prompt_audio_path(filename)
    if os.path.isfile(clone_path):
        return clone_path

    reference_path = timbre_reference_map_path(filename)
    if os.path.isfile(reference_path):
        with open(reference_path, encoding="utf-8") as reference_file:
            timbre_path = reference_file.read().strip()
        if timbre_path and os.path.isfile(timbre_path):
            return timbre_path
    return clone_path


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_matching_timbre_audio(content: bytes) -> str | None:
    """查找已有设计音频，避免将其复制到克隆存储目录。"""
    content_digest = hashlib.sha256(content).hexdigest()
    with os.scandir(TIMBRE_STORAGE_DIR) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.lower().endswith(".wav"):
                continue
            if file_sha256(entry.path) == content_digest:
                return entry.path
    return None


def prompt_text_sidecar_path(filename: str) -> str:
    clone_sidecar_path = os.path.join(PROMPTS_DIR, f"{hash_filename(filename)}.prompt.txt")
    if os.path.isfile(clone_prompt_audio_path(filename)):
        return clone_sidecar_path
    if os.path.isfile(timbre_reference_map_path(filename)):
        return os.path.join(TIMBRE_REFERENCE_DIR, f"{hash_filename(filename)}.prompt.txt")
    return clone_sidecar_path


def load_prompt_text_sidecar(filename: str) -> str | None:
    sidecar_path = prompt_text_sidecar_path(filename)
    if not os.path.isfile(sidecar_path):
        return None
    with open(sidecar_path, encoding="utf-8") as f:
        text = f.read().strip()
    return text or None


def save_prompt_text_sidecar(filename: str, prompt_text: str | None) -> None:
    sidecar_path = prompt_text_sidecar_path(filename)
    normalized = prompt_text.strip() if prompt_text and prompt_text.strip() else None
    if normalized is None:
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
        return
    with open(sidecar_path, "w", encoding="utf-8") as f:
        f.write(normalized)


def store_uploaded_audio(
    content: bytes,
    full_path: str,
    prompt_text: str | None,
) -> dict[str, object]:
    """同步保存参考音频和 sidecar，避免阻塞异步请求处理。"""
    clone_path = clone_prompt_audio_path(full_path)
    timbre_path = find_matching_timbre_audio(content)
    if timbre_path:
        # 同一份设计音色可能会再次上传用于克隆预览；实际 WAV 只保存在音色目录，
        # 这里只持久化一份轻量的解析映射。
        if os.path.isfile(clone_path):
            os.remove(clone_path)
        clone_sidecar_path = os.path.join(PROMPTS_DIR, f"{hash_filename(full_path)}.prompt.txt")
        if os.path.isfile(clone_sidecar_path):
            os.remove(clone_sidecar_path)
        with open(timbre_reference_map_path(full_path), "w", encoding="utf-8") as reference_file:
            reference_file.write(timbre_path)
    else:
        reference_map_path = timbre_reference_map_path(full_path)
        if os.path.isfile(reference_map_path):
            os.remove(reference_map_path)
        with open(clone_path, "wb") as file:
            file.write(content)

    normalized_prompt_text = prompt_text.strip() if prompt_text and prompt_text.strip() else None
    save_prompt_text_sidecar(full_path, normalized_prompt_text)
    return {
        "code": 200,
        "msg": "上传成功",
        "filename": full_path,
        "has_prompt_text": bool(normalized_prompt_text),
    }


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


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
    """Qwen3-TTS 克隆请求，补充模型专用的生成参数。"""
    text: str
    audio_path: str
    prompt_text: str | None = None
    language: str | None = None
    x_vector_only: bool | None = None
    device_map: str | None = None
    dtype: str | None = None
    attn_implementation: str | None = None
    max_new_tokens: int | None = None
    top_p: float | None = None
    temperature: float | None = None
    max_chars_per_chunk: int | None = None
    pause_ms: int | None = None
    trim_leading_silence: bool | None = None
    trim_leading_silence_threshold_db: float | None = None
    trim_leading_silence_min_ms: int | None = None
    trim_leading_silence_analysis_window_ms: int | None = None
    trim_leading_silence_pre_roll_ms: int | None = None
    trim_leading_silence_max_ms: int | None = None


class Qwen3TtsWorkerManager:
    """管理参考音频解析、worker JSON 和一次性推理生命周期。"""
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.last_output_path: str | None = None

    def build_worker_payload(self, request: Qwen3TtsSynthesizeRequest) -> dict:
        """将请求字段、参考音频 sidecar 和服务默认值组合为 worker 输入。"""
        ref_audio_path = prompt_audio_path(request.audio_path)
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        prompt_text = (
            request.prompt_text.strip()
            if request.prompt_text and request.prompt_text.strip()
            else None
        )
        if prompt_text is None:
            prompt_text = load_prompt_text_sidecar(request.audio_path)

        return {
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "ref_text": prompt_text,
            "model_path": QWEN3_TTS_MODEL_DIR,
            "language": normalize_optional_text(request.language)
            if request.language is not None
            else QWEN3_TTS_LANGUAGE,
            "x_vector_only": (
                request.x_vector_only
                if request.x_vector_only is not None
                else QWEN3_TTS_X_VECTOR_ONLY
            ),
            "device_map": request.device_map or QWEN3_TTS_DEVICE_MAP,
            "dtype": request.dtype or QWEN3_TTS_DTYPE,
            "attn_implementation": request.attn_implementation or QWEN3_TTS_ATTN_IMPLEMENTATION,
            "max_new_tokens": (
                request.max_new_tokens
                if request.max_new_tokens is not None
                else QWEN3_TTS_MAX_NEW_TOKENS
            ),
            "top_p": request.top_p if request.top_p is not None else QWEN3_TTS_TOP_P,
            "temperature": request.temperature
            if request.temperature is not None
            else QWEN3_TTS_TEMPERATURE,
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
        """启动当前 uv 环境中的 Qwen3-TTS worker，并读取结果 WAV。"""
        self.last_output_path = None
        python_executable = sys.executable
        if not python_executable or not os.path.isfile(python_executable):
            raise RuntimeError(
                "未找到 qwen3_tts uv 环境的 Python 解释器，无法调用 Qwen3-TTS worker。"
            )
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
        proc: subprocess.Popen | None = None

        try:
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            command = [
                python_executable,
                QWEN3_TTS_WORKER_SCRIPT,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[Qwen3-TTS] 启动 worker: python={python_executable}")
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
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(proc, "Qwen3-TTS")
                stdout, stderr = proc.communicate()
                raise RuntimeError(
                    f"Qwen3-TTS worker 超时（>{QWEN3_TTS_REQUEST_TIMEOUT:.0f}s）"
                ) from exc

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
            saved_output_path = persist_audio_bytes(
                audio_bytes,
                "qwen3_tts",
                QWEN3_TTS_OUTPUT_DIR,
            )
            self.last_output_path = str(saved_output_path)
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
def health():
    """返回服务配置和依赖状态，不加载 Qwen3-TTS 模型。"""
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "qwen3_tts_model_dir": QWEN3_TTS_MODEL_DIR,
            "qwen_libs_path": QWEN_LIBS_PATH,
            "prompts_dir": PROMPTS_DIR,
            "tts_output_dir": QWEN3_TTS_OUTPUT_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": QWEN3_TTS_WORKER_SCRIPT,
            "worker_tmp_dir": QWEN3_TTS_WORKER_TMP_DIR,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(QWEN3_TTS_WORKER_SCRIPT),
            "qwen3_tts_model_dir": os.path.isdir(QWEN3_TTS_MODEL_DIR),
            "qwen_libs_path": os.path.isdir(QWEN_LIBS_PATH),
            "torch": module_available("torch"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
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
def internal_unload_all(request: Request):
    """保留本机控制接口；模型已由一次性 worker 自行释放。"""
    assert_local_request(request)
    with gpu_runtime_lock("qwen3_tts/unload"):
        with manager.lock:
            pass
    return JSONResponse({"code": 200, "msg": "qwen3_tts worker 已退出，无常驻模型"})


@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...),
    prompt_text: str | None = Form(None),
):
    """在线程池中保存上传参考音频，避免阻塞 FastAPI 事件循环。"""
    content = await audio.read()
    return await run_in_threadpool(store_uploaded_audio, content, full_path, prompt_text)


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    """检查逻辑路径对应的参考音频或音色引用是否存在。"""
    exists = os.path.isfile(prompt_audio_path(file_name))
    return {
        "code": 200 if exists else 404,
        "exists": exists,
        "has_prompt_text": bool(load_prompt_text_sidecar(file_name)),
    }


@app.post("/v1/qwen/clone")
def synthesize_v2(request: Qwen3TtsSynthesizeRequest):
    """串行执行一次 Qwen3-TTS 克隆，并返回生成 WAV。"""
    with gpu_runtime_lock("qwen3_tts/synthesize"):
        with manager.lock:
            try:
                payload = manager.build_worker_payload(request)
                manager.last_output_path = None
                audio_bytes = manager.run_worker(payload)
                if manager.last_output_path is None:
                    saved_output_path = persist_audio_bytes(
                        audio_bytes,
                        "qwen3_tts",
                        QWEN3_TTS_OUTPUT_DIR,
                    )
                else:
                    saved_output_path = manager.last_output_path
                print(f"[Qwen3-TTS] 已保存生成音频: {saved_output_path}")
                return Response(content=audio_bytes, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as exc:
                LOGGER.exception("Qwen3-TTS request failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                wait_after_cuda_release("after qwen3_tts worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 Qwen3-TTS Voice Clone")
    print("==================================================")
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
    print(
        f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={QWEN3_TTS_REQUEST_TIMEOUT}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
