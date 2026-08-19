"""Runtime helpers for the one-shot ACE-Step 1.5 worker service.

The API process intentionally does not import torch or diffusers.  A fresh
worker process owns the model and its CUDA context for one request only.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for one worker invocation."""

    worker_script: Path
    temp_dir: Path
    timeout: float
    label: str
    file_prefix: str


@dataclass(frozen=True)
class WorkerResult:
    """Audio bytes plus lightweight metadata emitted by the worker."""

    audio: bytes
    metadata: dict[str, Any]


def module_available(module_name: str) -> bool:
    """Return whether a module can be found without importing it."""
    try:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def cuda_status() -> dict[str, Any]:
    """Read GPU status with nvidia-smi without creating a CUDA context."""
    status: dict[str, Any] = {"available": False, "source": "nvidia-smi"}
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        status["error"] = "nvidia-smi not found"
        return status

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.free,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not rows:
            status["error"] = "nvidia-smi returned no GPU"
            return status

        fields = [field.strip() for field in rows[0].split(",")]
        if len(fields) != 5:
            status["error"] = f"unexpected nvidia-smi output: {rows[0]}"
            return status

        _, device_name, free_mib, total_mib, used_mib = fields
        status.update(
            {
                "available": True,
                "device_count": len(rows),
                "device_name": device_name,
                "memory": {
                    "free_mib": float(free_mib),
                    "total_mib": float(total_mib),
                    "used_mib": float(used_mib),
                    "allocated_mib": None,
                    "reserved_mib": None,
                },
            }
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        status["error"] = str(exc)
    return status


def process_is_running(process: Any) -> bool:
    """Handle real Popen objects and small process mocks used in tests."""
    if process is None:
        return False
    poll = getattr(process, "poll", None)
    if callable(poll):
        return poll() is None
    return getattr(process, "returncode", None) is None


def terminate_process_group(
    process: Any,
    label: str,
    terminate_timeout: float = 10,
    kill_timeout: float = 5,
) -> None:
    """Terminate a worker and all descendants, escalating to SIGKILL."""
    if not process_is_running(process):
        return

    pid: int | None = getattr(process, "pid", None)
    if pid is None:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            wait = getattr(process, "wait", None)
            if callable(wait):
                try:
                    wait(timeout=kill_timeout)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            return
        except OSError as exc:
            print(f"[{label}] 终止 worker 进程组失败，改为终止主进程: {exc}")
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()

    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        wait(timeout=terminate_timeout)
        return
    except subprocess.TimeoutExpired:
        print(f"[{label}] worker 未及时退出，强制终止进程组")

    if pid is None:
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            try:
                wait(timeout=kill_timeout)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            return
        except OSError:
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()

    wait(timeout=kill_timeout)


def worker_error_excerpt(output: str, label: str) -> str:
    """Keep the useful tail of a worker traceback for an HTTP error."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return f"{label} worker 未输出错误信息。"
    return " | ".join(lines[-8:])


def run_local_worker(payload: dict[str, Any], config: WorkerConfig) -> WorkerResult:
    """Run one worker in this uv project and return non-empty WAV bytes."""
    if not config.worker_script.is_file():
        raise RuntimeError(f"{config.label} worker 脚本不存在: {config.worker_script}")

    config.temp_dir.mkdir(parents=True, exist_ok=True)
    request_fd, request_path = tempfile.mkstemp(
        dir=config.temp_dir,
        prefix=f"{config.file_prefix}_req_",
        suffix=".json",
    )
    output_fd, output_path = tempfile.mkstemp(
        dir=config.temp_dir,
        prefix=f"{config.file_prefix}_out_",
        suffix=".wav",
    )
    metadata_fd, metadata_path = tempfile.mkstemp(
        dir=config.temp_dir,
        prefix=f"{config.file_prefix}_meta_",
        suffix=".json",
    )
    for file_descriptor in (request_fd, output_fd, metadata_fd):
        os.close(file_descriptor)

    process: subprocess.Popen[str] | None = None
    paths = (request_path, output_path, metadata_path)

    try:
        with open(request_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

        command = [
            os.environ.get("ACESTEP_PYTHON", sys.executable),
            str(config.worker_script),
            "--input-json",
            request_path,
            "--output-wav",
            output_path,
            "--metadata-json",
            metadata_path,
        ]
        print(f"[{config.label}] 启动 uv worker: python={command[0]}")
        worker_env = os.environ.copy()
        worker_env.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=worker_env,
        )
        try:
            stdout, stderr = process.communicate(timeout=config.timeout)
        except subprocess.TimeoutExpired as exc:
            terminate_process_group(process, config.label)
            process.communicate()
            raise RuntimeError(f"{config.label} worker 超时（>{config.timeout:.0f}s）") from exc

        if stdout.strip():
            print(stdout.rstrip())
        if stderr.strip():
            print(stderr.rstrip())
        if process.returncode != 0:
            raise RuntimeError(worker_error_excerpt(stderr or stdout, config.label))
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"{config.label} worker 未生成音频文件。")

        metadata: dict[str, Any] = {}
        if os.path.isfile(metadata_path) and os.path.getsize(metadata_path) > 0:
            with open(metadata_path, encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                metadata = loaded

        with open(output_path, "rb") as file:
            return WorkerResult(audio=file.read(), metadata=metadata)
    finally:
        terminate_process_group(process, config.label)
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass


@contextmanager
def gpu_runtime_lock(lock_path: Path, label: str) -> Iterator[None]:
    """Serialize GPU workers with the repository-wide advisory lock."""
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        print(f"[GPU 锁] 等待进入: {label}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(f"[GPU 锁] 已进入: {label}")
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(f"[GPU 锁] 已退出: {label}")


def persist_audio_bytes(audio_bytes: bytes, model_prefix: str, output_dir: Path) -> Path:
    """Atomically persist a successful WAV response for local inspection."""
    if not audio_bytes:
        raise ValueError("cannot persist empty audio")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_fd, temporary_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{model_prefix}_{timestamp}_",
        suffix=".tmp",
    )
    temporary_file = Path(temporary_path)
    output_path = output_dir / f"{temporary_file.name[1:-4]}.wav"
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
        temporary_file.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
