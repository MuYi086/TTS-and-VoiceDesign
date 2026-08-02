"""Shared one-shot worker runner for model-specific Conda environments.

The HTTP processes deliberately do not import heavyweight model packages.  A
request is serialized to a temporary JSON file, executed in the configured
Conda environment, and the worker's WAV is read back before the process group
is terminated and temporary files are removed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from gpu_runtime import terminate_process_group


@dataclass(frozen=True)
class LocalWorkerConfig:
    """Runtime settings for one model worker family."""

    conda_env: str
    worker_script: str
    model_dir: str
    temp_dir: str
    timeout: float
    label: str
    file_prefix: str


def resolve_conda_executable() -> Optional[str]:
    """Resolve the configured Conda executable without importing a model."""
    configured = os.environ.get("CONDA_EXE")
    if configured:
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
        if os.path.isfile(path):
            return path
    return shutil.which("conda")


def worker_error_excerpt(output: str, label: str) -> str:
    """Keep the useful tail of a worker traceback for an HTTP error response."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return f"{label} worker 未输出错误信息。"
    return " | ".join(lines[-8:])


def run_local_worker(payload: dict[str, Any], config: LocalWorkerConfig) -> bytes:
    """Run one model worker and return its validated non-empty WAV bytes."""
    conda_exe = resolve_conda_executable()
    if not conda_exe:
        raise RuntimeError(f"未找到 conda 命令，无法调用 {config.label} worker。")

    worker_script = Path(config.worker_script)
    if not worker_script.is_file():
        raise RuntimeError(f"{config.label} worker 脚本不存在: {worker_script}")
    if config.model_dir and not Path(config.model_dir).is_dir():
        raise RuntimeError(f"{config.label} 模型目录不存在: {config.model_dir}")

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
    process: Optional[subprocess.Popen] = None

    try:
        with open(request_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

        command = [
            conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            config.conda_env,
            "python",
            str(worker_script),
            "--input-json",
            request_path,
            "--output-wav",
            output_path,
        ]
        print(f"[{config.label}] 启动 worker: env={config.conda_env}")
        worker_env = os.environ.copy()
        # The parent may use allocator settings that are incompatible with a
        # model's own torch build; let each worker start from a clean CUDA env.
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
            stdout, stderr = process.communicate(timeout=config.timeout)
        except subprocess.TimeoutExpired:
            terminate_process_group(process, config.label)
            process.communicate()
            raise RuntimeError(f"{config.label} worker 超时（>{config.timeout:.0f}s）")

        if stdout.strip():
            print(stdout.rstrip())
        if stderr.strip():
            print(stderr.rstrip())
        if process.returncode != 0:
            raise RuntimeError(worker_error_excerpt(stderr or stdout, config.label))
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"{config.label} worker 未生成音频文件。")
        with open(output_path, "rb") as file:
            return file.read()
    finally:
        terminate_process_group(process, config.label)
        for path in (request_path, output_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
