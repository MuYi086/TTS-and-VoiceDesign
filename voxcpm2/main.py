import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Literal, Optional

# 官方文档: https://voxcpm.readthedocs.io/zh-cn/latest/cookbook.html

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.base import BaseHTTPMiddleware
from audio_output import persist_audio_bytes, persist_audio_file
from synthesis_request import CloneSynthesisRequest
from gpu_runtime import cuda_status, terminate_process_group

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SERVICE_DIR)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def normalize_device_name(value: Optional[str], default: str = "cuda") -> str:
    normalized = normalize_optional_text(value)
    return normalized.lower() if normalized is not None else default


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", os.path.join(PROJECT_DIR, "storage")))
CLONE_STORAGE_DIR = expand_path(
    os.getenv("CLONE_STORAGE_DIR", os.path.join(STORAGE_DIR, "clone"))
)
TIMBRE_STORAGE_DIR = expand_path(
    os.getenv("TIMBRE_STORAGE_DIR", os.path.join(STORAGE_DIR, "timbre"))
)
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", CLONE_STORAGE_DIR))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", os.path.join(STORAGE_DIR, ".cache/runtime"))
)
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8306"))

VOXCPM2_MODEL_DIR = expand_path(
    os.getenv("VOXCPM2_MODEL_DIR", os.path.join(HF_MIRROR_DIR, "openbmb/VoxCPM2"))
)
VOXCPM2_HELPER_DEFAULT = os.path.join(SERVICE_DIR, "voxcpm2_helpers.py")
VOXCPM2_EXTERNAL_HELPER_PATHS = {
    expand_path(
        os.path.join("~", "github", "timbre-design", "modelScript", "tts_local_voxcpm2.py")
    ),
    expand_path(
        os.path.join("~", "github", "timbre-design", "scripts", "tts_local_voxcpm2.py")
    ),
}


def resolve_voxcpm2_helper_script(configured_path: Optional[str]) -> str:
    helper_path = expand_path(configured_path or VOXCPM2_HELPER_DEFAULT)
    # Earlier versions exported a helper from the adjacent timbre-design
    # project. Fall back only when one of those known stale paths is missing,
    # while keeping arbitrary user-provided helper overrides intact.
    if helper_path in VOXCPM2_EXTERNAL_HELPER_PATHS and not os.path.isfile(helper_path):
        return VOXCPM2_HELPER_DEFAULT
    return helper_path


# VoxCPM2 辅助脚本路径：未通过环境变量指定时使用仓库内受版本控制的实现。
VOXCPM2_HELPER_SCRIPT = resolve_voxcpm2_helper_script(os.getenv("VOXCPM2_HELPER_SCRIPT"))
# 引导强度：通常在 1~3 调整；提高可强化条件约束，但可能降低自然度或稳定性。
# 项目默认使用官方 Demo 的 2.0；
# 所有 VoxCPM2 克隆和音色设计请求未显式覆盖时都读取这个全局值。
VOXCPM2_CFG_DEFAULT = 2.0
VOXCPM2_CFG_VALUE = float(os.getenv("VOXCPM2_CFG_VALUE", str(VOXCPM2_CFG_DEFAULT)))
# 推理步数：越高通常越稳定但越慢；常用范围 4~30，当前以 10 平衡速度与质量。
VOXCPM2_INFERENCE_TIMESTEPS = int(os.getenv("VOXCPM2_INFERENCE_TIMESTEPS", "10"))
# 输出归一化：统一响度，可能改变参考音频原始的动态范围。
VOXCPM2_NORMALIZE = env_bool("VOXCPM2_NORMALIZE", False)
# 降噪：可减弱参考或生成中的噪声，但可能损失细节；启用时自动加载降噪器。
# 不兼容，改成True，会报错
VOXCPM2_DENOISE = env_bool("VOXCPM2_DENOISE", False)
# 坏例重试：模型判定结果异常时重试，提高成功率但会增加耗时。
VOXCPM2_RETRY_BADCASE = env_bool("VOXCPM2_RETRY_BADCASE", True)
# 加载降噪器：仅在启用降噪或需预热降噪器时设为 true，会增加显存和加载时间。
VOXCPM2_LOAD_DENOISER = env_bool("VOXCPM2_LOAD_DENOISER", False)
# 推理优化：启用后可能提升速度，但应先在当前 CUDA/torch 组合验证兼容性。
# 不兼容，改成True，会报错
VOXCPM2_OPTIMIZE = env_bool("VOXCPM2_OPTIMIZE", False)
# 运行设备：VoxCPM2 当前仅支持 CUDA 设备，例如 cuda 或 cuda:0。
VOXCPM2_DEVICE = normalize_device_name(os.getenv("VOXCPM2_DEVICE"), "cuda")
# 随机种子：官方在线推理默认不固定种子；负数表示沿用运行时随机状态。
# 只有复现实验时才通过请求或 VOXCPM2_SEED 显式指定非负整数。
VOXCPM2_SEED_DEFAULT = -1
VOXCPM2_SEED = int(os.getenv("VOXCPM2_SEED", str(VOXCPM2_SEED_DEFAULT)))
# 分片字符数：0 表示不切分；长文本切分可降低单次显存压力，但会在片段间插入停顿。
VOXCPM2_MAX_CHARS_PER_CHUNK = int(os.getenv("VOXCPM2_MAX_CHARS_PER_CHUNK", "0"))
# 分片停顿毫秒数：只在发生文本切分时生效，过大会让语流显得断裂。
VOXCPM2_PAUSE_MS = int(os.getenv("VOXCPM2_PAUSE_MS", "250"))
# 单次请求超时秒数：覆盖 worker 启动和完整生成；过短会中断长文本或冷启动。
VOXCPM2_REQUEST_TIMEOUT = float(os.getenv("VOXCPM2_REQUEST_TIMEOUT", "600"))

VOXCPM2_WORKER_SCRIPT = os.path.join(SERVICE_DIR, "worker.py")
VOXCPM2_WORKER_TMP_DIR = os.path.join(RUNTIME_CACHE_DIR, "voxcpm2_worker")
VOXCPM2_OUTPUT_DIR = expand_path(
    os.getenv(
        "VOXCPM2_OUTPUT_DIR",
        CLONE_STORAGE_DIR,
    )
)
VOXCPM2_TIMBRE_OUTPUT_DIR = expand_path(
    os.getenv("VOXCPM2_TIMBRE_OUTPUT_DIR", TIMBRE_STORAGE_DIR)
)
VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT = os.path.join(SERVICE_DIR, "voice_design_worker.py")

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
os.makedirs(VOXCPM2_WORKER_TMP_DIR, exist_ok=True)
os.makedirs(VOXCPM2_OUTPUT_DIR, exist_ok=True)
os.makedirs(VOXCPM2_TIMBRE_OUTPUT_DIR, exist_ok=True)
gpu_lock_dir = os.path.dirname(GPU_LOCK_FILE)
if gpu_lock_dir:
    os.makedirs(gpu_lock_dir, exist_ok=True)

app = FastAPI(title="Unitale VoxCPM2 Voice Clone API")


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


def persist_generated_audio(source_path: str, operation: str = "clone") -> str:
    """Copy a validated worker WAV into its semantic storage category."""
    output_dir = (
        VOXCPM2_TIMBRE_OUTPUT_DIR if operation == "voice_design" else VOXCPM2_OUTPUT_DIR
    )
    prefix = "voxcpm2_voicedesign" if operation == "voice_design" else "voxcpm2"
    return str(persist_audio_file(source_path, prefix, output_dir))


def persist_generated_audio_bytes(audio_bytes: bytes, operation: str = "clone") -> str:
    """Persist route output, including bytes returned by a mocked worker."""
    output_dir = (
        VOXCPM2_TIMBRE_OUTPUT_DIR if operation == "voice_design" else VOXCPM2_OUTPUT_DIR
    )
    prefix = "voxcpm2_voicedesign" if operation == "voice_design" else "voxcpm2"
    return str(persist_audio_bytes(audio_bytes, prefix, output_dir))


def sha256_file(path: str) -> str:
    """Return the content hash used to detect stale same-name uploads."""
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        return "VoxCPM2 worker 未输出错误信息。"
    return " | ".join(lines[-8:])


VOXCPM2_NONVERBAL_TAGS = frozenset({
    "laughing",
    "sigh",
    "Uhm",
    "Shh",
    "Question-ah",
    "Question-ei",
    "Question-en",
    "Question-oh",
    "Surprise-wa",
    "Surprise-yo",
    "Dissatisfaction-hnn",
})


def normalize_nonverbal_tags(tags: Optional[list[str]]) -> list[str]:
    """只接受 VoxCPM2 官方白名单中的至多一个非语言标签。"""
    normalized = [str(tag).strip() for tag in (tags or [])]
    if len(normalized) > 1:
        raise ValueError("VoxCPM2 nonverbal_tags 最多只能包含一个标签。")
    if normalized and normalized[0] not in VOXCPM2_NONVERBAL_TAGS:
        raise ValueError("VoxCPM2 nonverbal_tags 包含不支持的标签。")
    return normalized


def contains_nonverbal_tag_marker(text: Optional[str]) -> bool:
    """禁止把受支持标签伪装成正文或参考转写的一部分。"""
    value = text or ""
    return any(f"[{tag}]" in value for tag in VOXCPM2_NONVERBAL_TAGS)


class VoxCpm2SynthesizeRequest(CloneSynthesisRequest):

    text: str
    audio_path: str
    prompt_text: Optional[str] = None
    # Ultimate Cloning 使用参考转写；可控克隆使用控制指令，二者按 VoxCPM2 官方接口互斥。
    clone_mode: Optional[Literal["ultimate", "controllable"]] = None
    control_instruction: Optional[str] = None
    # 仅保存官方标签名；worker 会将其拼成 [tag] 并且只写入模型目标文本。
    nonverbal_tags: list[str] = Field(default_factory=list, max_length=1)
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None
    normalize: Optional[bool] = None
    denoise: Optional[bool] = None
    retry_badcase: Optional[bool] = None
    load_denoiser: Optional[bool] = None
    optimize: Optional[bool] = None
    device: Optional[str] = None
    seed: Optional[int] = None
    max_chars_per_chunk: Optional[int] = None
    pause_ms: Optional[int] = None

    @model_validator(mode="after")
    def validate_clone_mode_contract(self):
        self.nonverbal_tags = normalize_nonverbal_tags(self.nonverbal_tags)
        if contains_nonverbal_tag_marker(self.text) or contains_nonverbal_tag_marker(self.prompt_text):
            raise ValueError("VoxCPM2 非语言标签必须使用 nonverbal_tags，不能写入 text 或 prompt_text。")
        has_prompt_text = normalize_optional_text(self.prompt_text) is not None
        has_control_instruction = normalize_optional_text(self.control_instruction) is not None
        if self.clone_mode == "controllable":
            if has_prompt_text:
                raise ValueError("VoxCPM2 可控克隆不能同时传 prompt_text。")
            if not has_control_instruction:
                raise ValueError("VoxCPM2 可控克隆需要 control_instruction。")
        elif self.clone_mode == "ultimate":
            if has_control_instruction:
                raise ValueError("VoxCPM2 极致克隆不能传 control_instruction。")
        elif has_control_instruction:
            raise ValueError("control_instruction 需要显式指定 clone_mode=controllable。")
        if self.nonverbal_tags and self.clone_mode != "controllable":
            raise ValueError("VoxCPM2 非语言标签只能用于 clone_mode=controllable。")
        return self


class VoxCpm2WorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(self, request: VoxCpm2SynthesizeRequest) -> dict:
        ref_audio_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=404, detail="音频不存在")

        clone_mode = request.clone_mode
        control_instruction = normalize_optional_text(request.control_instruction)
        prompt_text = normalize_optional_text(request.prompt_text)
        # 可控克隆严格省略参考转写及其 sidecar，避免进入 Ultimate Cloning 签名。
        if clone_mode == "controllable":
            prompt_text = None
        elif prompt_text is None:
            prompt_text = load_prompt_text_sidecar(request.audio_path)
        settings = self.build_generation_settings(request)

        return {
            "operation": "clone",
            "text": normalize_synthesis_text(request.text),
            "ref_audio_path": ref_audio_path,
            "clone_mode": clone_mode,
            "prompt_text": prompt_text,
            "control_instruction": control_instruction,
            "nonverbal_tags": request.nonverbal_tags,
            **settings,
        }

    def build_generation_settings(self, request) -> dict:
        """统一整理同一 VoxCPM2 运行时使用的生成参数。"""
        device = normalize_device_name(request.device, VOXCPM2_DEVICE)
        if not device.startswith("cuda"):
            raise HTTPException(status_code=400, detail=f"VoxCPM2 仅支持 GPU 设备，当前 device={device}")
        denoise = request.denoise if request.denoise is not None else VOXCPM2_DENOISE
        configured_load_denoiser = (
            request.load_denoiser if request.load_denoiser is not None else VOXCPM2_LOAD_DENOISER
        )
        max_chars_per_chunk = getattr(request, "max_chars_per_chunk", None)
        pause_ms = getattr(request, "pause_ms", None)
        return {
            "model_path": VOXCPM2_MODEL_DIR,
            "voxcpm2_helper_script": VOXCPM2_HELPER_SCRIPT,
            "cfg_value": request.cfg_value if request.cfg_value is not None else VOXCPM2_CFG_VALUE,
            "inference_timesteps": (
                request.inference_timesteps
                if request.inference_timesteps is not None
                else VOXCPM2_INFERENCE_TIMESTEPS
            ),
            "normalize": request.normalize if request.normalize is not None else VOXCPM2_NORMALIZE,
            "denoise": denoise,
            "retry_badcase": (
                request.retry_badcase if request.retry_badcase is not None else VOXCPM2_RETRY_BADCASE
            ),
            # 启用 denoise 时自动补齐模型加载前置条件，避免请求参数互相矛盾。
            "load_denoiser": bool(configured_load_denoiser or denoise),
            "optimize": request.optimize if request.optimize is not None else VOXCPM2_OPTIMIZE,
            "device": device,
            "seed": request.seed if request.seed is not None else VOXCPM2_SEED,
            "max_chars_per_chunk": (
                max_chars_per_chunk
                if max_chars_per_chunk is not None
                else VOXCPM2_MAX_CHARS_PER_CHUNK
            ),
            "pause_ms": (
                pause_ms
                if pause_ms is not None
                else VOXCPM2_PAUSE_MS
            ),
            "local_files_only": LOCAL_FILES_ONLY,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "hf_mirror_dir": HF_MIRROR_DIR,
        }

    def _run_worker_once(self, payload: dict, worker_script: Optional[str] = None) -> bytes:
        selected_worker_script = worker_script or VOXCPM2_WORKER_SCRIPT
        if not os.path.isfile(selected_worker_script):
            raise RuntimeError(f"VoxCPM2 worker 脚本不存在: {selected_worker_script}")
        if not os.path.isdir(VOXCPM2_MODEL_DIR):
            raise RuntimeError(f"VoxCPM2 模型目录不存在: {VOXCPM2_MODEL_DIR}")
        if not os.path.isfile(VOXCPM2_HELPER_SCRIPT):
            raise RuntimeError(f"VoxCPM2 辅助脚本不存在: {VOXCPM2_HELPER_SCRIPT}")

        request_fd, request_path = tempfile.mkstemp(
            dir=VOXCPM2_WORKER_TMP_DIR,
            prefix="voxcpm2_req_",
            suffix=".json",
        )
        output_fd, output_path = tempfile.mkstemp(
            dir=VOXCPM2_WORKER_TMP_DIR,
            prefix="voxcpm2_out_",
            suffix=".wav",
        )
        os.close(request_fd)
        os.close(output_fd)
        proc: Optional[subprocess.Popen] = None

        try:
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            command = [
                sys.executable,
                selected_worker_script,
                "--input-json",
                request_path,
                "--output-wav",
                output_path,
            ]
            print(f"[VoxCPM2] 启动 worker: runtime=uv, python={sys.executable}")
            started = time.perf_counter()
            worker_env = os.environ.copy()
            worker_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            worker_env.pop("CUDA_MODULE_LOADING", None)
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=worker_env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=VOXCPM2_REQUEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                terminate_process_group(proc, "VoxCPM2")
                stdout, stderr = proc.communicate()
                raise RuntimeError(f"VoxCPM2 worker 超时（>{VOXCPM2_REQUEST_TIMEOUT:.0f}s）")

            elapsed = time.perf_counter() - started
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            print(f"[VoxCPM2] worker 退出码={proc.returncode}，耗时 {elapsed:.2f}s")

            if proc.returncode != 0:
                raise RuntimeError(worker_error_excerpt(stderr or stdout))
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("VoxCPM2 worker 未生成音频文件。")

            with open(output_path, "rb") as file:
                return file.read()
        finally:
            terminate_process_group(proc, "VoxCPM2")
            for path in (request_path, output_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

    def run_worker(self, payload: dict, worker_script: Optional[str] = None) -> bytes:
        try:
            audio_bytes = self._run_worker_once(payload, worker_script=worker_script)
            self.last_error = None
            return audio_bytes
        except Exception as exc:
            self.last_error = str(exc)
            raise


manager = VoxCpm2WorkerManager()


class VoxCpm2VoiceDesignRequest(BaseModel):
    """与主 API 兼容的无参考音频音色设计请求。"""

    voice_description: str
    text: str = "这是生成的参考音频预览。"
    save_as: Optional[str] = "designed_voice.wav"
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None
    normalize: Optional[bool] = None
    denoise: Optional[bool] = None
    retry_badcase: Optional[bool] = None
    load_denoiser: Optional[bool] = None
    optimize: Optional[bool] = None
    device: Optional[str] = None
    seed: Optional[int] = None


def build_voice_design_worker_payload(request: VoxCpm2VoiceDesignRequest) -> dict:
    voice_description = normalize_optional_text(request.voice_description)
    if not voice_description:
        raise HTTPException(status_code=422, detail="VoxCPM2 音色设计需要 voice_description。")
    return {
        "operation": "voice_design",
        "text": normalize_synthesis_text(request.text),
        "voice_description": voice_description,
        **manager.build_generation_settings(request),
    }


def voice_design_is_ready() -> bool:
    return (
        os.path.isdir(VOXCPM2_MODEL_DIR)
        and os.path.isfile(VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT)
        and os.path.isfile(VOXCPM2_HELPER_SCRIPT)
    )


@app.get("/v1/health")
async def health():
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "voxcpm2_model_dir": VOXCPM2_MODEL_DIR,
            "voxcpm2_helper_script": VOXCPM2_HELPER_SCRIPT,
            "prompts_dir": PROMPTS_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "worker_script": VOXCPM2_WORKER_SCRIPT,
            "voice_design_worker_script": VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT,
            "worker_tmp_dir": VOXCPM2_WORKER_TMP_DIR,
            "output_dir": VOXCPM2_OUTPUT_DIR,
            "clone_storage_dir": VOXCPM2_OUTPUT_DIR,
            "timbre_storage_dir": VOXCPM2_TIMBRE_OUTPUT_DIR,
        },
        "available": {
            "python": sys.executable,
            "worker_script": os.path.isfile(VOXCPM2_WORKER_SCRIPT),
            "voice_design_worker_script": os.path.isfile(VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT),
            "voxcpm2_model_dir": os.path.isdir(VOXCPM2_MODEL_DIR),
            "voxcpm2_helper_script": os.path.isfile(VOXCPM2_HELPER_SCRIPT),
            "torch": module_available("torch"),
            "flash_attn": module_available("flash_attn"),
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "runtime": {
            "worker_runtime": "uv",
            "worker_python": sys.executable,
            "flash_attention": "not required; VoxCPM uses PyTorch scaled_dot_product_attention",
            "flash_attention_policy": (
                "not required; VoxCPM uses PyTorch scaled_dot_product_attention and does not import flash_attn"
            ),
            "backends": {
                "voxcpm2": "voxcpm2",
            },
            "model_lifecycle": "one request -> one worker -> process exit releases VRAM",
            "local_files_only": LOCAL_FILES_ONLY,
            "request_timeout": VOXCPM2_REQUEST_TIMEOUT,
            "cfg_value": VOXCPM2_CFG_VALUE,
            "inference_timesteps": VOXCPM2_INFERENCE_TIMESTEPS,
            "normalize": VOXCPM2_NORMALIZE,
            "denoise": VOXCPM2_DENOISE,
            "retry_badcase": VOXCPM2_RETRY_BADCASE,
            "load_denoiser": VOXCPM2_LOAD_DENOISER,
            "optimize": VOXCPM2_OPTIMIZE,
            "device": VOXCPM2_DEVICE,
            "seed": VOXCPM2_SEED,
            "max_chars_per_chunk": VOXCPM2_MAX_CHARS_PER_CHUNK,
            "pause_ms": VOXCPM2_PAUSE_MS,
            "clone_modes": {
                "ultimate": "prompt_text -> upload sidecar -> reference-only compatibility fallback",
                "controllable": "control_instruction only; prompt_text and sidecar are omitted",
            },
            "nonverbal_tags": sorted(VOXCPM2_NONVERBAL_TAGS),
        },
        "last_errors": {
            "voxcpm2_tts": manager.last_error,
        },
    }


@app.post("/v1/voxcpm2/design")
async def voxcpm2_design(request: VoxCpm2VoiceDesignRequest):
    with gpu_runtime_lock("voxcpm2/design"):
        try:
            payload = build_voice_design_worker_payload(request)
            with manager.lock:
                audio_bytes = manager.run_worker(
                    payload,
                    worker_script=VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT,
                )
            saved_output_path = persist_generated_audio_bytes(
                audio_bytes,
                operation="voice_design",
            )
            print(f"[VoxCPM2] 已保存音色音频: {saved_output_path}")
            return Response(content=audio_bytes, media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            wait_after_cuda_release("after VoxCPM2 voice design")


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    with gpu_runtime_lock("voxcpm2/unload"):
        with manager.lock:
            pass
    return JSONResponse({"code": 200, "msg": "voxcpm2 worker 已退出，无常驻模型"})


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
    content_sha256 = hashlib.sha256(content).hexdigest()

    return {
        "code": 200,
        "msg": "上传成功",
        "filename": full_path,
        "size_bytes": len(content),
        "sha256": content_sha256,
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
async def synthesize_v2(request: Request):
    with gpu_runtime_lock("voxcpm2/synthesize"):
        try:
            data = await request.json()
            with manager.lock:
                payload = manager.build_worker_payload(VoxCpm2SynthesizeRequest.model_validate(data))
                audio_bytes = manager.run_worker(payload)
            saved_output_path = persist_generated_audio_bytes(audio_bytes, operation="clone")
            print(f"[VoxCPM2] 已保存生成音频: {saved_output_path}")
            return Response(content=audio_bytes, media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            wait_after_cuda_release("after 8306 worker")


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI 本地后端 VoxCPM2 Voice Clone")
    print("==================================================")
    print(f"[配置] VoxCPM2 worker runtime: uv, python={sys.executable}")
    print(f"[配置] VoxCPM2 模型目录: {VOXCPM2_MODEL_DIR}")
    print(f"[配置] VoxCPM2 helper: {VOXCPM2_HELPER_SCRIPT}")
    print(f"[配置] prompts 目录: {PROMPTS_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] worker 脚本: {VOXCPM2_WORKER_SCRIPT}")
    print(f"[配置] VoiceDesign worker 脚本: {VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT}")
    print(
        f"[配置] cfg_value={VOXCPM2_CFG_VALUE}, inference_timesteps={VOXCPM2_INFERENCE_TIMESTEPS}, "
        f"seed={VOXCPM2_SEED}"
    )
    print(
        f"[配置] normalize={VOXCPM2_NORMALIZE}, denoise={VOXCPM2_DENOISE}, "
        f"retry_badcase={VOXCPM2_RETRY_BADCASE}, load_denoiser={VOXCPM2_LOAD_DENOISER}, optimize={VOXCPM2_OPTIMIZE}, "
        f"max_chars_per_chunk={VOXCPM2_MAX_CHARS_PER_CHUNK}, pause_ms={VOXCPM2_PAUSE_MS}"
    )
    print(
        f"[配置] device={VOXCPM2_DEVICE}"
    )
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, request_timeout={VOXCPM2_REQUEST_TIMEOUT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
