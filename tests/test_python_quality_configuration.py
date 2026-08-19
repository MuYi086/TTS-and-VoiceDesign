"""No-model regression tests for the repository's Python quality baseline."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
MODEL_PROJECTS = (
    "mimo_tts",
    "qwen3_tts",
    "voxcpm2",
    "LongCat_AudioDiT_3.5B_bf16",
    "dots_tts_soar",
    "moss_soundEffect",
    "stable_audio_3_medium",
    "ace_step_1_5",
    "qwen3_voiceDesign",
    "moss_voiceGenerator",
    "Step_Audio_EditX",
)
EXPECTED_RUFF_RULES = {"E", "F", "I", "UP", "B"}


class PythonQualityConfigurationTests(unittest.TestCase):
    def test_each_model_project_has_a_locked_ruff_baseline(self) -> None:
        for project_name in MODEL_PROJECTS:
            with self.subTest(project=project_name):
                project_dir = REPOSITORY_DIR / project_name
                configuration = tomllib.loads(
                    (project_dir / "pyproject.toml").read_text(encoding="utf-8")
                )

                self.assertEqual(configuration["project"]["requires-python"], "==3.12.13")
                self.assertFalse(configuration["tool"]["uv"]["package"])
                self.assertIn("ruff>=0.15.9", configuration["dependency-groups"]["dev"])
                self.assertEqual(
                    (project_dir / ".python-version").read_text(encoding="utf-8").strip(),
                    "3.12.13",
                )

                ruff_config = configuration["tool"]["ruff"]
                self.assertEqual(ruff_config["line-length"], 100)
                self.assertEqual(ruff_config["target-version"], "py312")
                self.assertEqual(
                    set(ruff_config["lint"]["select"]),
                    EXPECTED_RUFF_RULES,
                )
                self.assertIn("B008", ruff_config["lint"]["ignore"])
                self.assertIn("E501", ruff_config["lint"]["ignore"])
                self.assertIn(
                    'name = "ruff"', (project_dir / "uv.lock").read_text(encoding="utf-8")
                )

    def test_root_quality_rules_and_moss_dependency_use_the_local_checkout(self) -> None:
        root_ruff = tomllib.loads((REPOSITORY_DIR / "ruff.toml").read_text(encoding="utf-8"))
        self.assertEqual(set(root_ruff["lint"]["select"]), EXPECTED_RUFF_RULES)
        self.assertIn("E501", root_ruff["lint"]["ignore"])

        moss_configuration = (REPOSITORY_DIR / "moss_voiceGenerator" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('path = "/home/muyi086/tts-depency/MOSS-TTS"', moss_configuration)
        self.assertIn("editable = true", moss_configuration)
        self.assertNotIn('git = "https://github.com/OpenMOSS/MOSS-TTS"', moss_configuration)


if __name__ == "__main__":
    unittest.main()
