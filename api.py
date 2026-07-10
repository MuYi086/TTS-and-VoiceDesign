import fcntl
import base64
import gc
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import wave
from contextlib import contextmanager

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import torch
import uvicorn
import soundfile as sf
from typing import Any, Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
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


HF_MIRROR_DIR = expand_path(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
DEFAULT_QWEN_LIBS = os.path.join(PROJECT_DIR, "vendor/qwen_libs")
DEFAULT_INDEXTTS_CODE_DIR = os.path.join(PROJECT_DIR, "vendor/index-tts")
QWEN_LIBS = os.getenv("QWEN_LIBS", DEFAULT_QWEN_LIBS if os.path.isdir(DEFAULT_QWEN_LIBS) else "")
QWEN_MODEL = expand_path(
    os.getenv(
        "QWEN_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    )
)
INDEXTTS_MODEL_DIR = expand_path(
    os.getenv(
        "INDEXTTS_MODEL_DIR",
        os.path.join(HF_MIRROR_DIR, "IndexTeam/IndexTTS-2"),
    )
)
INDEXTTS_CFG_PATH = expand_path(
    os.getenv("INDEXTTS_CFG_PATH", os.path.join(INDEXTTS_MODEL_DIR, "config.yaml"))
)
INDEXTTS_CODE_DIR = os.getenv(
    "INDEXTTS_CODE_DIR",
    DEFAULT_INDEXTTS_CODE_DIR if os.path.isdir(DEFAULT_INDEXTTS_CODE_DIR) else "",
)
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", os.path.join(PROJECT_DIR, "prompts")))
RUNTIME_CACHE_DIR = expand_path(os.getenv("RUNTIME_CACHE_DIR", os.path.join(PROJECT_DIR, ".cache/runtime")))
GPU_LOCK_FILE = expand_path(os.getenv("GPU_LOCK_FILE", os.path.join(RUNTIME_CACHE_DIR, "gpu-runtime.lock")))
LOCAL_FILES_ONLY = env_bool("LOCAL_FILES_ONLY", True)
PRELOAD_INDEXTTS = env_bool("PRELOAD_INDEXTTS", False)
CLEAN_UNKNOWN_PYTHON_PROCESSES = env_bool("CLEAN_UNKNOWN_PYTHON_PROCESSES", False)
INDEXTTS_DEVICE = os.getenv("INDEXTTS_DEVICE") or None
INDEXTTS_USE_FP16 = env_bool("INDEXTTS_USE_FP16", True)
INDEXTTS_USE_CUDA_KERNEL = env_bool("INDEXTTS_USE_CUDA_KERNEL", False)
INDEXTTS_NUM_BEAMS = int(os.getenv("INDEXTTS_NUM_BEAMS", "1"))
CUDA_RELEASE_DELAY = float(os.getenv("CUDA_RELEASE_DELAY", "2.0"))
QWEN_DEVICE = os.getenv("QWEN_DEVICE") or None
QWEN_DTYPE = os.getenv("QWEN_DTYPE") or None
QWEN_ATTN_IMPLEMENTATION = os.getenv("QWEN_ATTN_IMPLEMENTATION") or None
QWEN_REQUEST_TIMEOUT = float(os.getenv("QWEN_REQUEST_TIMEOUT", "120"))
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2.5-tts-voicedesign")
MIMO_AUTH_HEADER = os.getenv("MIMO_AUTH_HEADER", "api-key")
MIMO_TIMEOUT = float(os.getenv("MIMO_TIMEOUT", "300"))
MIMO_MAX_CHARS_PER_CHUNK = int(os.getenv("MIMO_MAX_CHARS_PER_CHUNK", "300"))
MIMO_PAUSE_MS = int(os.getenv("MIMO_PAUSE_MS", "250"))
MIMO_OPTIMIZE_TEXT_PREVIEW = env_bool("MIMO_OPTIMIZE_TEXT_PREVIEW", False)
MIMO_MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("MIMO_MIN_REQUEST_INTERVAL_SECONDS", "0"))
MIMO_MAX_RETRIES = int(os.getenv("MIMO_MAX_RETRIES", "3"))
MIMO_RETRY_BASE_SECONDS = float(os.getenv("MIMO_RETRY_BASE_SECONDS", "5"))
MIMO_RETRY_MAX_SECONDS = float(os.getenv("MIMO_RETRY_MAX_SECONDS", "60"))
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8300"))

INDEXTTS_AUX_DIR = os.path.join(INDEXTTS_MODEL_DIR, "hf_cache")
INDEXTTS_REQUIRED_FILES = (
    "config.yaml",
    "bpe.model",
    "wav2vec2bert_stats.pt",
    "gpt.pth",
    "s2mel.pth",
    "feat2.pt",
    "feat1.pt",
    "qwen0.6bemo4-merge/config.json",
    "qwen0.6bemo4-merge/model.safetensors",
    "qwen0.6bemo4-merge/tokenizer.json",
    "qwen0.6bemo4-merge/tokenizer_config.json",
)
INDEXTTS_AUX_REQUIRED_FILES = (
    "hf_cache/w2v-bert-2.0/config.json",
    "hf_cache/w2v-bert-2.0/preprocessor_config.json",
    "hf_cache/semantic_codec_model.safetensors",
    "hf_cache/campplus_cn_common.bin",
    "hf_cache/bigvgan/config.json",
    "hf_cache/bigvgan/bigvgan_generator.pt",
)

os.environ.setdefault("HF_HOME", HF_MIRROR_DIR)
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(RUNTIME_CACHE_DIR, "hf_modules"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(RUNTIME_CACHE_DIR, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RUNTIME_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RUNTIME_CACHE_DIR, "xdg"))
if LOCAL_FILES_ONLY:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if INDEXTTS_CODE_DIR:
    indextts_code_path = expand_path(INDEXTTS_CODE_DIR)
    if os.path.isdir(indextts_code_path) and indextts_code_path not in sys.path:
        sys.path.insert(0, indextts_code_path)

os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
gpu_lock_dir = os.path.dirname(GPU_LOCK_FILE)
if gpu_lock_dir:
    os.makedirs(gpu_lock_dir, exist_ok=True)

app = FastAPI(title="Super Unitale Smart API")

class ForceCORS(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return Response(status_code=200, headers={
                "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Credentials": "false",
            })
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

app.add_middleware(ForceCORS)

def hash_filename(filename: str) -> str:
    ext = os.path.splitext(filename)[1] or ".wav"
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{h}{ext}"


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


def missing_relative_files(base_dir: str, relative_paths: tuple[str, ...]) -> List[str]:
    return [
        rel_path
        for rel_path in relative_paths
        if not os.path.isfile(os.path.join(base_dir, rel_path))
    ]


def indextts_aux_paths() -> dict:
    return {
        "w2v_bert": os.path.join(INDEXTTS_AUX_DIR, "w2v-bert-2.0"),
        "semantic_codec": os.path.join(INDEXTTS_AUX_DIR, "semantic_codec_model.safetensors"),
        "campplus": os.path.join(INDEXTTS_AUX_DIR, "campplus_cn_common.bin"),
        "bigvgan": os.path.join(INDEXTTS_AUX_DIR, "bigvgan"),
    }


def indextts_file_status() -> dict:
    main_missing = missing_relative_files(INDEXTTS_MODEL_DIR, INDEXTTS_REQUIRED_FILES)
    aux_missing = missing_relative_files(INDEXTTS_MODEL_DIR, INDEXTTS_AUX_REQUIRED_FILES)
    return {
        "main_missing": main_missing,
        "aux_missing": aux_missing,
        "main_ready": not main_missing,
        "aux_ready": not aux_missing,
        "ready": not main_missing and not aux_missing,
    }


def module_available(module_name: str, search_path: Optional[str] = None) -> bool:
    if search_path:
        path = expand_path(search_path)
        if not os.path.isdir(path):
            return False
        return importlib.machinery.PathFinder.find_spec(module_name, [path]) is not None
    return importlib.util.find_spec(module_name) is not None


CUDA_ERROR_MARKERS = (
    "cuda",
    "cublas",
    "cudnn",
    "device not ready",
    "device-side assert",
    "out of memory",
)


def is_cuda_runtime_error(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in CUDA_ERROR_MARKERS)


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


def assert_cuda_ready(operation: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        probe = torch.empty(1, device="cuda")
        probe.fill_(1)
        del probe
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            f"{operation} 前 CUDA 自检失败: {exc}. "
            "请先停止残留的 python/api.py 进程；如果 nvidia-smi 仍显示已退出进程占用显存，"
            "需要重启 WSL 或宿主机 NVIDIA 驱动后再启动服务。"
        ) from exc


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


def close_mp_queue(q: Optional[multiprocessing.Queue]) -> None:
    if q is None:
        return
    try:
        q.cancel_join_thread()
    except Exception:
        pass
    try:
        q.close()
    except Exception:
        pass

# ==========================================
# 1. Qwen3 守护进程逻辑
# ==========================================
def qwen_daemon(input_q, output_q):
    model = None
    try:
        if QWEN_LIBS:
            qwen_libs_path = expand_path(QWEN_LIBS)
            if os.path.isdir(qwen_libs_path) and qwen_libs_path not in sys.path:
                sys.path.insert(0, qwen_libs_path)

        import torch
        import sox

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        if not os.path.isdir(QWEN_MODEL):
            raise FileNotFoundError(f"Qwen 模型目录不存在: {QWEN_MODEL}")
        
        print(f"🟢 [Qwen Daemon] 子进程启动 (PID: {os.getpid()})，正在加载模型...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device_map = QWEN_DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype_name = QWEN_DTYPE or ("bfloat16" if torch.cuda.is_available() else "float32")
        dtype = getattr(torch, dtype_name)
        load_kwargs = {
            "device_map": device_map,
            "dtype": dtype,
            "local_files_only": LOCAL_FILES_ONLY,
        }
        if QWEN_ATTN_IMPLEMENTATION:
            load_kwargs["attn_implementation"] = QWEN_ATTN_IMPLEMENTATION

        print(f"🟢 [Qwen Daemon] 模型目录: {QWEN_MODEL}")
        print(f"🟢 [Qwen Daemon] device_map={device_map}, dtype={dtype_name}, local_files_only={LOCAL_FILES_ONLY}")
        model = Qwen3TTSModel.from_pretrained(QWEN_MODEL, **load_kwargs)
        print(f"🟢 [Qwen Daemon] 模型加载完毕，等待指令...")
        
        while True:
            try:
                task = input_q.get(timeout=1) 
            except queue.Empty:
                continue

            if task.get("command") == "STOP":
                print("🔴 [Qwen Daemon] 收到停止指令，正在退出...")
                break
            
            if task.get("command") == "DESIGN":
                try:
                    print(f"🔵 [Qwen Daemon] 开始处理音色合成任务...")
                    req_dict = task["data"]
                    wavs, sr = model.generate_voice_design(
                        text=req_dict.get("text", "预览"),
                        language="Chinese",
                        instruct=req_dict["voice_description"]
                    )
                    
                    audio_data = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else wavs[0]
                    buf = io.BytesIO()
                    sf.write(buf, audio_data, sr, format="WAV")
                    buf.seek(0)
                    
                    output_q.put({"success": True, "audio_bytes": buf.read()})
                    print(f"🔵 [Qwen Daemon] 任务完成")
                except Exception as e:
                    traceback.print_exc()
                    output_q.put({"success": False, "error": str(e)})
                    
    except Exception as e:
        print(f"❌ [Qwen Daemon] 致命错误: {e}")
        traceback.print_exc()
        try:
            output_q.put({"success": False, "error": str(e)})
        except Exception:
            pass
    finally:
        if model:
            del model
        clear_cuda_cache("Qwen daemon exit")
        print("🔴 [Qwen Daemon] 进程销毁，显存释放")

# ==========================================
# 2. 全局模型管理器 (精准白名单版)
# ==========================================
class ModelManager:
    def __init__(self):
        self.indextts = None
        self.indextts_error: Optional[str] = None
        self.qwen_process: Optional[multiprocessing.Process] = None
        self.qwen_in_q: Optional[multiprocessing.Queue] = None
        self.qwen_out_q: Optional[multiprocessing.Queue] = None
        self.lock = threading.RLock()
        
        # 记录主进程 PID
        self.main_pid = os.getpid()
        
        if PRELOAD_INDEXTTS and multiprocessing.current_process().name == 'MainProcess':
            print("[启动] PRELOAD_INDEXTTS 已忽略：当前策略为真实请求时加载、请求结束后卸载。")

    def _init_resident_model(self):
        print(f"[启动] 主进程 PID: {self.main_pid}")
        self.ensure_indextts_loaded()

    def ensure_indextts_loaded(self):
        if self.indextts is not None:
            return

        print("[IndexTTS2] 正在载入本地模型...")
        if INDEXTTS_CODE_DIR:
            indextts_code_path = expand_path(INDEXTTS_CODE_DIR)
            if os.path.isdir(indextts_code_path) and indextts_code_path not in sys.path:
                sys.path.insert(0, indextts_code_path)

        missing = []
        if not os.path.isdir(INDEXTTS_MODEL_DIR):
            missing.append(f"模型目录不存在: {INDEXTTS_MODEL_DIR}")
        if not os.path.isfile(INDEXTTS_CFG_PATH):
            missing.append(f"配置文件不存在: {INDEXTTS_CFG_PATH}")
        file_status = indextts_file_status()
        if file_status["main_missing"]:
            missing.append("主模型文件缺失: " + ", ".join(file_status["main_missing"]))
        if file_status["aux_missing"]:
            missing.append("辅助模型文件缺失: " + ", ".join(file_status["aux_missing"]))
        if missing:
            self.indextts_error = "；".join(missing)
            raise RuntimeError(self.indextts_error)

        from indextts.infer_v2 import IndexTTS2

        assert_cuda_ready("加载 IndexTTS2")

        try:
            self.indextts = IndexTTS2(
                model_dir=INDEXTTS_MODEL_DIR,
                cfg_path=INDEXTTS_CFG_PATH,
                aux_paths=indextts_aux_paths(),
                device=INDEXTTS_DEVICE,
                use_fp16=INDEXTTS_USE_FP16,
                use_cuda_kernel=INDEXTTS_USE_CUDA_KERNEL,
            )
            self.indextts_error = None
            print(
                "✅ IndexTTS2 就绪。"
                f" device={self.indextts.device}, fp16={self.indextts.use_fp16},"
                f" cuda_kernel={self.indextts.use_cuda_kernel}"
            )
        except Exception as e:
            self.indextts = None
            self.indextts_error = str(e)
            clear_cuda_cache("IndexTTS2 load failed")
            raise

    def _kill_zombies(self):
        """云端兼容清理逻辑。本地默认关闭，避免误杀用户自己的 Python 任务。"""
        if not CLEAN_UNKNOWN_PYTHON_PROCESSES:
            return

        try:
            # 使用 ps 命令获取更详细的信息：PID 和 命令行
            cmd = "ps -eo pid,args | grep python"
            output = subprocess.check_output(cmd, shell=True, encoding='utf-8')
            
            for line in output.strip().split('\n'):
                if not line: continue
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2: continue
                
                try:
                    pid = int(parts[0])
                    cmdline = parts[1]
                except ValueError:
                    continue

                # 🛑 豁免名单 (Whitelist)
                if pid == self.main_pid: continue
                if self.qwen_process and self.qwen_process.is_alive() and pid == self.qwen_process.pid: continue
                if "resource_tracker" in cmdline: continue
                if "grep" in cmdline or "ps -eo" in cmdline: continue

                # ☠️ 只有不在白名单里的，才是真正的入侵者
                print(f"💀 [内部清洗] 发现未知 Python 进程 PID: {pid} ({cmdline[:15]}...)，执行清理...")
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
        except Exception:
            pass

    def ensure_qwen_loaded(self):
        if self.qwen_process is not None and self.qwen_process.is_alive():
            return 

        print("🟡 [调度器] 准备启动 Qwen3...")
        self._kill_zombies() 
        self.unload_indextts()
        
        print("🧹 [调度器] 正在清理显存缓存...")
        clear_cuda_cache("before Qwen load")
        wait_after_cuda_release("before Qwen load")

        print("🟡 [调度器] 拉起 Qwen3 守护进程...")
        self.qwen_in_q = multiprocessing.Queue()
        self.qwen_out_q = multiprocessing.Queue()
        self.qwen_process = multiprocessing.Process(
            target=qwen_daemon, 
            args=(self.qwen_in_q, self.qwen_out_q),
            daemon=True
        )
        self.qwen_process.start()

    def unload_qwen(self):
        proc = self.qwen_process
        if proc is None:
            return

        if proc.is_alive():
            print("⚠️ [调度器] 正在卸载 Qwen3...")
            try:
                if self.qwen_in_q is not None:
                    self.qwen_in_q.put({"command": "STOP"})
                proc.join(timeout=10)
                if proc.is_alive():
                    print("⚠️ [调度器] Qwen3 未及时退出，强制终止")
                    proc.terminate()
                    proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
            except Exception as e:
                print(f"⚠️ [调度器] 卸载 Qwen3 时发生异常: {e}")

        close_mp_queue(self.qwen_in_q)
        close_mp_queue(self.qwen_out_q)
        self.qwen_process = None
        self.qwen_in_q = None
        self.qwen_out_q = None
        clear_cuda_cache("after Qwen unload")
        wait_after_cuda_release("after Qwen unload")
        print("✅ [调度器] Qwen3 已卸载")

    def unload_indextts(self):
        if self.indextts is None:
            return

        print("⚠️ [调度器] 正在卸载 IndexTTS2...")
        try:
            del self.indextts
        except Exception:
            pass
        self.indextts = None
        clear_cuda_cache("after IndexTTS2 unload")
        wait_after_cuda_release("after IndexTTS2 unload")
        print("✅ [调度器] IndexTTS2 已卸载")

    def unload_all(self):
        self.unload_qwen()
        self.unload_indextts()

manager = ModelManager()

# ==========================================
# 3. 接口定义
# ==========================================
class TextToSpeechRequest(CloneSynthesisRequest):
    text: str 
    audio_path: str 
    emo_text: Optional[str] = None
    emo_vector: Optional[List[float]] = Field(None, min_length=8, max_length=8)

class QwenDesignRequest(BaseModel):
    voice_description: str
    text: str = "这是生成的参考音频预览。"
    save_as: Optional[str] = "designed_voice.wav" 


class MimoDesignRequest(BaseModel):
    voice_description: str
    text: str = "这是生成的参考音频预览。"
    save_as: Optional[str] = "designed_voice.wav"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    auth_header: Optional[str] = None
    timeout: Optional[float] = None
    max_chars_per_chunk: Optional[int] = None
    pause_ms: Optional[int] = None
    optimize_text_preview: Optional[bool] = None
    min_request_interval_seconds: Optional[float] = None
    max_retries: Optional[int] = None
    retry_base_seconds: Optional[float] = None
    retry_max_seconds: Optional[float] = None


class MiMoHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str, retry_after: Optional[float] = None):
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        super().__init__(f"MiMo HTTP {status_code}: {body}")


MIMO_REQUEST_LOCK = threading.Lock()


def split_long_voice_design_text(text: str, max_chars: int) -> list[str]:
    parts = re.findall(r".+?[，,、：:]|.+$", text, flags=re.S)
    chunks: list[str] = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(part[index : index + max_chars] for index in range(0, len(part), max_chars))
            continue
        candidate = current + part
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_voice_design_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    sentences = re.findall(r".+?[。！？；;!?]|.+$", text, flags=re.S)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_voice_design_text(sentence, max_chars))
            continue
        candidate = current + sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def resolve_mimo_api_key(api_key: Optional[str]) -> str:
    resolved = api_key or os.getenv("MIMO_API_KEY")
    if not resolved:
        raise RuntimeError("MiMo API key 缺失。请设置 MIMO_API_KEY，或在请求中传入 api_key。")
    return resolved


def mimo_chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def mimo_request_headers(api_key: str, auth_header: str) -> dict[str, str]:
    normalized = auth_header.strip().lower()
    if normalized not in {"api-key", "bearer", "both"}:
        raise ValueError(f"不支持的 MiMo auth_header: {auth_header}")

    headers = {"Content-Type": "application/json"}
    if normalized in {"api-key", "both"}:
        headers["api-key"] = api_key
    if normalized in {"bearer", "both"}:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def mimo_build_messages(voice_instruction: str, chunk: str) -> list[dict[str, str]]:
    voice_instruction = voice_instruction.strip()
    if not voice_instruction:
        return [{"role": "assistant", "content": chunk}]
    return [
        {"role": "user", "content": voice_instruction},
        {"role": "assistant", "content": chunk},
    ]


def mimo_parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return None


def mimo_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        retry_after = mimo_parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
        raise MiMoHTTPError(exc.code, error_body, retry_after) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"MiMo request failed: {exc.reason}") from exc

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MiMo returned non-JSON response: {response_body[:500]}") from exc


def mimo_is_retryable_http_error(exc: MiMoHTTPError) -> bool:
    return exc.status_code == 429 or 500 <= exc.status_code <= 599


def mimo_retry_delay_seconds(exc: MiMoHTTPError, attempt: int, base: float, maximum: float) -> float:
    if exc.retry_after is not None:
        return min(maximum, exc.retry_after)
    return min(maximum, max(0.0, base) * (2 ** max(0, attempt - 1)))


def mimo_post_json_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    min_request_interval_seconds: float,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    chunk_label: str,
) -> dict[str, Any]:
    with MIMO_REQUEST_LOCK:
        for attempt in range(1, max_retries + 2):
            try:
                response = mimo_post_json(url, payload, headers, timeout)
            except MiMoHTTPError as exc:
                if not mimo_is_retryable_http_error(exc) or attempt > max_retries:
                    raise
                delay = mimo_retry_delay_seconds(exc, attempt, retry_base_seconds, retry_max_seconds)
                print(
                    f"MiMo HTTP {exc.status_code}，{delay:.1f}s 后重试 {chunk_label}，"
                    f"第 {attempt}/{max_retries} 次"
                )
                time.sleep(delay)
                continue

            if min_request_interval_seconds > 0:
                time.sleep(min_request_interval_seconds)
            return response

    raise RuntimeError("MiMo request did not return a response")


def mimo_extract_audio_bytes(response: dict[str, Any]) -> bytes:
    try:
        encoded = response["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"MiMo response 缺少 choices[0].message.audio.data: {response}") from exc
    return base64.b64decode(encoded)


def read_wav_params(audio_bytes: bytes) -> wave._wave_params:
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        params = reader.getparams()
    if params.comptype != "NONE":
        raise RuntimeError(f"不支持压缩 wav 拼接: {params.comptype}")
    return params


def join_wav_bytes(chunks: list[bytes], pause_ms: int) -> bytes:
    if not chunks:
        raise RuntimeError("MiMo 未返回音频片段。")

    first_params = read_wav_params(chunks[0])
    sample_rate = int(first_params.framerate)
    frame_size = first_params.nchannels * first_params.sampwidth
    pause_frames = max(0, int(sample_rate * pause_ms / 1000))
    pause = b"\x00" * pause_frames * frame_size

    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams(first_params)
        for index, chunk in enumerate(chunks):
            with wave.open(io.BytesIO(chunk), "rb") as reader:
                params = reader.getparams()
                if params[:3] != first_params[:3] or params[4:] != first_params[4:]:
                    raise RuntimeError(
                        "MiMo wav 分块参数不一致，无法拼接："
                        f"chunk 1={first_params}, chunk {index + 1}={params}"
                    )
                writer.writeframes(reader.readframes(reader.getnframes()))
            if index < len(chunks) - 1 and pause:
                writer.writeframes(pause)

    return output.getvalue()


def run_mimo_voice_design(request_data: dict[str, Any]) -> bytes:
    text = str(request_data.get("text") or "").strip()
    if not text:
        raise RuntimeError("text 不能为空。")

    voice_instruction = str(request_data.get("voice_description") or "").strip()
    if not voice_instruction:
        raise RuntimeError("voice_description 不能为空。")

    model = str(request_data.get("model") or MIMO_MODEL).strip()
    if model != "mimo-v2.5-tts-voicedesign":
        raise RuntimeError(f"MiMo 音色设计仅支持 mimo-v2.5-tts-voicedesign，当前为: {model}")

    api_key = resolve_mimo_api_key(request_data.get("api_key"))
    base_url = str(request_data.get("base_url") or MIMO_BASE_URL)
    auth_header = str(request_data.get("auth_header") or MIMO_AUTH_HEADER)
    timeout = float(request_data.get("timeout") if request_data.get("timeout") is not None else MIMO_TIMEOUT)
    max_chars_per_chunk = int(
        request_data.get("max_chars_per_chunk")
        if request_data.get("max_chars_per_chunk") is not None
        else MIMO_MAX_CHARS_PER_CHUNK
    )
    pause_ms = int(request_data.get("pause_ms") if request_data.get("pause_ms") is not None else MIMO_PAUSE_MS)
    optimize_text_preview = (
        bool(request_data["optimize_text_preview"])
        if request_data.get("optimize_text_preview") is not None
        else MIMO_OPTIMIZE_TEXT_PREVIEW
    )
    min_request_interval_seconds = float(
        request_data.get("min_request_interval_seconds")
        if request_data.get("min_request_interval_seconds") is not None
        else MIMO_MIN_REQUEST_INTERVAL_SECONDS
    )
    max_retries = max(
        0,
        int(
            request_data.get("max_retries")
            if request_data.get("max_retries") is not None
            else MIMO_MAX_RETRIES
        ),
    )
    retry_base_seconds = float(
        request_data.get("retry_base_seconds")
        if request_data.get("retry_base_seconds") is not None
        else MIMO_RETRY_BASE_SECONDS
    )
    retry_max_seconds = float(
        request_data.get("retry_max_seconds")
        if request_data.get("retry_max_seconds") is not None
        else MIMO_RETRY_MAX_SECONDS
    )

    chunks = split_voice_design_text(text, max_chars_per_chunk)
    audio_payload: dict[str, Any] = {"format": "wav"}
    if optimize_text_preview:
        audio_payload["optimize_text_preview"] = True

    print(f"[MiMo] model={model}, base_url={base_url}, chunks={len(chunks)}")
    url = mimo_chat_completions_url(base_url)
    headers = mimo_request_headers(api_key, auth_header)
    audio_chunks: list[bytes] = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"[MiMo] 合成 chunk {index}/{len(chunks)}: {len(chunk)} 字")
        payload = {
            "model": model,
            "messages": mimo_build_messages(voice_instruction, chunk),
            "audio": audio_payload,
        }
        response = mimo_post_json_with_retry(
            url=url,
            payload=payload,
            headers=headers,
            timeout=timeout,
            min_request_interval_seconds=min_request_interval_seconds,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
            chunk_label=f"MiMo chunk {index}/{len(chunks)}",
        )
        audio_chunks.append(mimo_extract_audio_bytes(response))

    return join_wav_bytes(audio_chunks, pause_ms)

@app.get("/v1/health")
async def health():
    qwen_pkg = module_available("qwen_tts", QWEN_LIBS)
    indextts_pkg = module_available("indextts")
    indextts_files = indextts_file_status()
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "hf_mirror_dir": HF_MIRROR_DIR,
            "qwen_model_dir": QWEN_MODEL,
            "indextts_model_dir": INDEXTTS_MODEL_DIR,
            "indextts_cfg_path": INDEXTTS_CFG_PATH,
            "indextts_aux_dir": INDEXTTS_AUX_DIR,
            "prompts_dir": PROMPTS_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "mimo_base_url": MIMO_BASE_URL,
        },
        "available": {
            "qwen_model_dir": os.path.isdir(QWEN_MODEL),
            "qwen_package": qwen_pkg,
            "mimo_api_key": bool(os.getenv("MIMO_API_KEY")),
            "indextts_model_dir": os.path.isdir(INDEXTTS_MODEL_DIR),
            "indextts_config": os.path.isfile(INDEXTTS_CFG_PATH),
            "indextts_package": indextts_pkg,
            "indextts_main_files": indextts_files["main_ready"],
            "indextts_aux_files": indextts_files["aux_ready"],
            "indextts_ready": indextts_files["ready"],
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "missing": {
            "indextts_main": indextts_files["main_missing"],
            "indextts_aux": indextts_files["aux_missing"],
        },
        "loaded": {
            "qwen_process": bool(manager.qwen_process and manager.qwen_process.is_alive()),
            "indextts": manager.indextts is not None,
        },
        "last_errors": {
            "indextts": manager.indextts_error,
        },
        "offline": {
            "local_files_only": LOCAL_FILES_ONLY,
            "hf_hub_offline": os.getenv("HF_HUB_OFFLINE"),
            "transformers_offline": os.getenv("TRANSFORMERS_OFFLINE"),
        },
        "runtime": {
            "voice_design_providers": ["qwen", "mimo"],
            "qwen_request_timeout": QWEN_REQUEST_TIMEOUT,
            "mimo_model": MIMO_MODEL,
            "mimo_auth_header": MIMO_AUTH_HEADER,
            "mimo_timeout": MIMO_TIMEOUT,
            "mimo_max_chars_per_chunk": MIMO_MAX_CHARS_PER_CHUNK,
            "mimo_pause_ms": MIMO_PAUSE_MS,
            "mimo_optimize_text_preview": MIMO_OPTIMIZE_TEXT_PREVIEW,
            "mimo_min_request_interval_seconds": MIMO_MIN_REQUEST_INTERVAL_SECONDS,
            "mimo_max_retries": MIMO_MAX_RETRIES,
        },
    }


@app.get("/v1/voice-design/providers")
async def voice_design_providers():
    return {
        "code": 200,
        "providers": [
            {
                "id": "qwen",
                "name": "Qwen3-TTS VoiceDesign",
                "route": "/v1/qwen/design",
                "type": "local_model",
                "ready": os.path.isdir(QWEN_MODEL) and module_available("qwen_tts", QWEN_LIBS),
            },
            {
                "id": "mimo",
                "name": "MiMo TTS VoiceDesign",
                "route": "/v1/mimo/design",
                "type": "cloud_api",
                "ready": bool(os.getenv("MIMO_API_KEY")),
            },
        ],
    }


@app.post("/internal/unload_all")
async def internal_unload_all(request: Request):
    assert_local_request(request)
    with manager.lock:
        manager.unload_all()
    return JSONResponse({"code": 200, "msg": "已卸载 qwen 和 indextts"})

@app.post("/v1/upload_audio")
async def upload_audio(audio: UploadFile = File(...), full_path: str = Form(...)):
    content = await audio.read()
    save_path = os.path.join(PROMPTS_DIR, hash_filename(full_path))
    with open(save_path, "wb") as f: f.write(content)
    return {"code": 200, "msg": "上传成功", "filename": full_path}

@app.get("/v1/check/audio")
async def check_audio_exists(file_name: str):
    exists = os.path.isfile(os.path.join(PROMPTS_DIR, hash_filename(file_name)))
    return {"code": 200 if exists else 404, "exists": exists}

@app.post("/v1/qwen/design")
async def qwen_design(request: QwenDesignRequest):
    with gpu_runtime_lock("qwen/design"):
        with manager.lock:
            try:
                manager.ensure_qwen_loaded()
                if manager.qwen_in_q is None or manager.qwen_out_q is None:
                    raise RuntimeError("Qwen 队列未初始化。")
                manager.qwen_in_q.put({"command": "DESIGN", "data": request.model_dump()})
                res = manager.qwen_out_q.get(timeout=QWEN_REQUEST_TIMEOUT)
                if res.get("success"):
                    return Response(content=res["audio_bytes"], media_type="audio/wav")

                error = res.get("error") or "Qwen 推理失败"
                raise HTTPException(status_code=500, detail=error)
            except queue.Empty:
                raise HTTPException(status_code=500, detail="Qwen 推理超时")
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            finally:
                manager.unload_qwen()


@app.post("/v1/mimo/design")
async def mimo_design(request: MimoDesignRequest):
    with gpu_runtime_lock("mimo/design"):
        with manager.lock:
            manager.unload_all()
            manager._kill_zombies()
            try:
                audio_bytes = run_mimo_voice_design(request.model_dump())
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            return Response(content=audio_bytes, media_type="audio/wav")

@app.post("/v2/synthesize")
async def synthesize_v2(request: TextToSpeechRequest):
    with gpu_runtime_lock("indextts/synthesize"):
        with manager.lock:
            manager.unload_qwen()
            manager._kill_zombies() # 双重保险

            real_file_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
            temp_out = os.path.join(PROMPTS_DIR, f"temp_synth_{time.time_ns()}.wav")
            if not os.path.isfile(real_file_path):
                raise HTTPException(status_code=404, detail="音频不存在")

            try:
                try:
                    manager.ensure_indextts_loaded()
                except Exception as e:
                    manager.indextts_error = str(e)
                    raise HTTPException(status_code=503, detail=f"IndexTTS2 未就绪: {e}")

                manager.indextts.infer(
                    spk_audio_prompt=real_file_path,
                    text=request.text,
                    output_path=temp_out,
                    emo_vector=request.emo_vector,
                    emo_text=request.emo_text,
                    use_emo_text=bool(request.emo_text),
                    emo_alpha=0.6,
                    num_beams=INDEXTTS_NUM_BEAMS,
                )
                with open(temp_out, "rb") as f:
                    data = f.read()
                return Response(content=data, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception as e:
                manager.indextts_error = str(e)
                traceback.print_exc()
                if is_cuda_runtime_error(e):
                    manager.unload_indextts()
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"{e}. 已卸载 IndexTTS2 以释放损坏的 CUDA 上下文；"
                            "请重试一次。如果 nvidia-smi 仍显示已退出的 python 进程占用显存，"
                            "需要重启 WSL 或宿主机 NVIDIA 驱动。"
                        ),
                    )
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if os.path.exists(temp_out):
                    os.remove(temp_out)
                manager.unload_indextts()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    print("==================================================")
    print("   Unitale AI 本地后端服务 IndexTTS2 + Qwen3/MiMo VoiceDesign")
    print("==================================================")
    print(f"[配置] Qwen 模型目录: {QWEN_MODEL}")
    print(f"[配置] MiMo base URL: {MIMO_BASE_URL}")
    print(f"[配置] MiMo 模型: {MIMO_MODEL}")
    print(f"[配置] MiMo API key: {'已配置' if os.getenv('MIMO_API_KEY') else '未配置'}")
    print(f"[配置] IndexTTS2 模型目录: {INDEXTTS_MODEL_DIR}")
    print(f"[配置] IndexTTS2 配置: {INDEXTTS_CFG_PATH}")
    print(f"[配置] prompts 目录: {PROMPTS_DIR}")
    print(f"[配置] GPU 锁文件: {GPU_LOCK_FILE}")
    print(f"[配置] local_files_only={LOCAL_FILES_ONLY}, preload_indextts={PRELOAD_INDEXTTS}")
    print(
        f"[配置] indextts_device={INDEXTTS_DEVICE or 'auto'}, "
        f"indextts_fp16={INDEXTTS_USE_FP16}, "
        f"indextts_cuda_kernel={INDEXTTS_USE_CUDA_KERNEL}, "
        f"indextts_num_beams={INDEXTTS_NUM_BEAMS}"
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
