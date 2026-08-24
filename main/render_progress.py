"""维护空间音频长任务的内存进度快照并输出终端日志。"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any

JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,79}$")
TERMINAL_STATES = {"succeeded", "failed"}
LOGGER = logging.getLogger("uvicorn.error")


class RenderProgressError(ValueError):
    """任务 ID 不合法或与仍保留的任务冲突。"""


class RenderProgressRegistry:
    """以线程安全方式保存可轮询的渲染进度，终态按保留期自动淘汰。"""

    def __init__(self, *, retention_seconds: float = 3600) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds 必须大于 0。")
        self.retention_seconds = retention_seconds
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()

    def _prune(self, now: float) -> None:
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job["state"] in TERMINAL_STATES
            and now - job["updated_monotonic"] > self.retention_seconds
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def start(self, job_id: str, message: str) -> dict[str, Any]:
        """注册新任务；拒绝不安全 ID 和保留期内的重复提交。"""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise RenderProgressError("job_id 必须是 8–80 位安全 ASCII 标识。")
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if job_id in self._jobs:
                raise RenderProgressError("job_id 已存在，请为每次导出生成新 ID。")
            snapshot = {
                "job_id": job_id,
                "state": "running",
                "stage": "accepted",
                "progress": 1,
                "message": message,
                "updated_at": self._timestamp(),
                "updated_monotonic": now,
            }
            self._jobs[job_id] = snapshot
        LOGGER.info("[Steam Audio][%s] %3d%% %s", job_id, 1, message)
        return self._public_snapshot(snapshot)

    @staticmethod
    def _public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in snapshot.items() if key != "updated_monotonic"}

    def update(
        self,
        job_id: str,
        *,
        stage: str,
        progress: int,
        message: str,
    ) -> dict[str, Any] | None:
        """推进任务阶段和百分比；进度单调递增且终态不可回退。"""
        now = time.monotonic()
        with self._lock:
            snapshot = self._jobs.get(job_id)
            if snapshot is None or snapshot["state"] in TERMINAL_STATES:
                return self._public_snapshot(snapshot) if snapshot else None
            snapshot.update(
                stage=stage,
                progress=max(snapshot["progress"], min(99, max(1, int(progress)))),
                message=message,
                updated_at=self._timestamp(),
                updated_monotonic=now,
            )
            public = self._public_snapshot(snapshot)
        LOGGER.info(
            "[Steam Audio][%s] %3d%% %s",
            job_id,
            public["progress"],
            message,
        )
        return public

    def succeed(self, job_id: str, message: str) -> dict[str, Any] | None:
        """把已注册任务标记为成功。"""
        return self._finish(
            job_id, state="succeeded", stage="completed", progress=100, message=message
        )

    def fail(self, job_id: str, message: str) -> dict[str, Any] | None:
        """把已注册任务标记为失败并保留可轮询的诊断。"""
        return self._finish(job_id, state="failed", stage="failed", progress=None, message=message)

    def _finish(
        self,
        job_id: str,
        *,
        state: str,
        stage: str,
        progress: int | None,
        message: str,
    ) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            snapshot = self._jobs.get(job_id)
            if snapshot is None or snapshot["state"] in TERMINAL_STATES:
                return self._public_snapshot(snapshot) if snapshot else None
            snapshot.update(
                state=state,
                stage=stage,
                progress=progress if progress is not None else snapshot["progress"],
                message=message,
                updated_at=self._timestamp(),
                updated_monotonic=now,
            )
            public = self._public_snapshot(snapshot)
        log = LOGGER.info if state == "succeeded" else LOGGER.error
        log("[Steam Audio][%s] %s %s", job_id, state.upper(), message)
        return public

    def get(self, job_id: str) -> dict[str, Any] | None:
        """返回任务公开快照；未知或已过保留期任务返回 None。"""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            snapshot = self._jobs.get(job_id)
            return self._public_snapshot(snapshot) if snapshot else None
