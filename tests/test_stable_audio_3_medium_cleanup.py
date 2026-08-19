"""验证 Stable Audio 迁移后不再残留旧的集中式 API/worker 文件。"""

from __future__ import annotations

import unittest
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]


class StableAudio3MediumCleanupTests(unittest.TestCase):
    def test_legacy_api_and_worker_are_removed(self) -> None:
        self.assertFalse((REPOSITORY_DIR / "main/stable_audio_3_medium_api.py").exists())
        self.assertFalse((REPOSITORY_DIR / "main/stable_audio_3_medium_worker.py").exists())

    def test_start_script_uses_only_the_standalone_uv_service(self) -> None:
        script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        self.assertIn(
            'setsid uv run --no-sync --project "$STABLE_AUDIO_3_MEDIUM_PROJECT_DIR"',
            script,
        )
        self.assertIn('python "$STABLE_AUDIO_3_MEDIUM_PROJECT_DIR/main.py"', script)
        self.assertNotIn("STABLE_AUDIO_3_MEDIUM_RUNTIME", script)
        self.assertNotIn("STABLE_AUDIO_3_MEDIUM_CONDA_ENV", script)
        self.assertNotIn('python "$MAIN_DIR/stable_audio_3_medium_api.py"', script)
        self.assertNotIn('python "$MAIN_DIR/stable_audio_3_medium_worker.py"', script)


if __name__ == "__main__":
    unittest.main()
