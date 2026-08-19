"""独立 MOSS-SoundEffect uv 服务的运行时辅助函数。"""

from __future__ import annotations

# 声效模型只能在 worker 中加载；运行时工具负责隔离启动和锁释放。
import json
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UvWorkerConfig:
    """当前 uv 环境中一次性 worker 的启动参数。"""

    python_executable: str
    worker_script: str
    model_dir: str
    code_path: str
    temp_dir: str
    timeout: float
    label: str
    file_prefix: str


def cuda_status() -> dict[str, Any]:
    """在不让 API 进程创建 CUDA 上下文的前提下读取 GPU 状态。"""
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
    """判断子进程是否存活，并兼容无真实 PID 的测试替身。"""
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
    """确保一次性 worker 及其子进程全部退出。"""
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

    try:
        wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        # 清理阶段不能覆盖 worker 原始异常；SIGKILL 后仍未回收时交给系统继续回收。
        pass


def worker_error_excerpt(output: str, label: str) -> str:
    """保留 worker traceback 的末尾关键信息，作为 HTTP 错误内容。"""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return f"{label} worker 未输出错误信息。"
    return " | ".join(lines[-8:])


def run_uv_worker(payload: dict[str, Any], config: UvWorkerConfig) -> bytes:
    """启动声效 worker，读取 WAV，并在成功或异常时完成清理。"""
    python_executable = Path(config.python_executable)
    if not python_executable.is_file():
        raise RuntimeError(f"未找到 uv 环境 Python 解释器: {python_executable}")

    worker_script = Path(config.worker_script)
    if not worker_script.is_file():
        raise RuntimeError(f"{config.label} worker 脚本不存在: {worker_script}")
    if config.model_dir and not Path(config.model_dir).is_dir():
        raise RuntimeError(f"{config.label} 模型目录不存在: {config.model_dir}")
    if config.code_path and not Path(config.code_path).is_dir():
        raise RuntimeError(f"{config.label} 上游源码目录不存在: {config.code_path}")

    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    request_fd, request_path = tempfile.mkstemp(
        dir=temp_dir,
        prefix=f"{config.file_prefix}_req_",
        suffix=".json",
    )
    output_fd, output_path = tempfile.mkstemp(
        dir=temp_dir,
        prefix=f"{config.file_prefix}_out_",
        suffix=".wav",
    )
    os.close(request_fd)
    os.close(output_fd)
    process: subprocess.Popen | None = None

    try:
        with open(request_path, "w", encoding="utf-8") as request_file:
            json.dump(payload, request_file, ensure_ascii=False)

        command = [
            str(python_executable),
            str(worker_script),
            "--input-json",
            request_path,
            "--output-wav",
            output_path,
        ]
        print(f"[{config.label}] 启动 worker: python={python_executable}")
        worker_env = os.environ.copy()
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
            stdout, stderr = process.communicate()
            excerpt = worker_error_excerpt(stderr or stdout, config.label)
            raise RuntimeError(
                f"{config.label} worker 超时（>{config.timeout:.0f}s）: {excerpt}"
            ) from exc

        if stdout.strip():
            print(stdout.rstrip())
        if stderr.strip():
            print(stderr.rstrip())
        if process.returncode != 0:
            raise RuntimeError(worker_error_excerpt(stderr or stdout, config.label))
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"{config.label} worker 未生成音频文件。")
        with open(output_path, "rb") as output_file:
            return output_file.read()
    finally:
        terminate_process_group(process, config.label)
        for path in (request_path, output_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
