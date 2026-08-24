"""空间音频长任务进度状态的无模型回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIR / "main"))

from render_progress import RenderProgressRegistry  # noqa: E402


class RenderProgressRegistryTests(unittest.TestCase):
    def test_progress_is_monotonic_and_terminal_state_cannot_regress(self) -> None:
        registry = RenderProgressRegistry(retention_seconds=60)
        registry.start("job-12345678", "开始处理")
        registry.update("job-12345678", stage="normalizing", progress=40, message="标准化")
        registry.update("job-12345678", stage="rendering", progress=30, message="渲染")
        registry.succeed("job-12345678", "完成")
        registry.update("job-12345678", stage="rendering", progress=80, message="迟到更新")

        snapshot = registry.get("job-12345678")
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["state"], "succeeded")
        self.assertEqual(snapshot["stage"], "completed")
        self.assertEqual(snapshot["progress"], 100)
        self.assertEqual(snapshot["message"], "完成")

    def test_fail_exposes_diagnostic_message(self) -> None:
        registry = RenderProgressRegistry(retention_seconds=60)
        registry.start("job-abcdefgh", "开始处理")
        registry.fail("job-abcdefgh", "renderer 退出码 4")

        snapshot = registry.get("job-abcdefgh")
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["stage"], "failed")
        self.assertEqual(snapshot["message"], "renderer 退出码 4")


if __name__ == "__main__":
    unittest.main()
