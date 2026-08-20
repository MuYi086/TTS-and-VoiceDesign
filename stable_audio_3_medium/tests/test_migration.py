"""Stable Audio 3 Medium uv 服务的无模型契约测试。"""

from __future__ import annotations

# Stable Audio 测试 mock worker，验证本地权重检查、FlashAttention 策略和路由契约。
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="stable-audio-3-medium-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
os.environ.update(
    {
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "SOUNDEFFECT_STORAGE_DIR": str(TEST_ROOT / "storage" / "soundEffect"),
        "TTS_OUTPUT_DIR": str(TEST_ROOT / "legacy-clone"),
    }
)
os.environ.pop("STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR", None)

import main
import runtime
import worker

FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "


class StableAudioApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app)

    def test_health_keeps_compatibility_sections_without_loading_model(self):
        with patch.object(main, "cuda_status", return_value={"available": False}):
            response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], 200)
        for key in ("paths", "available", "cuda", "runtime", "last_errors"):
            self.assertIn(key, body)
        self.assertIn("model_required_files_detail", body["available"])
        self.assertIn("flash_attention", body["runtime"])

    def test_duration_alias_and_defaults_are_normalized(self):
        request = main.StableAudio3MediumGenerateRequest(
            prompt="  glass shattering  ", duration=1.5
        )
        payload = main.manager.build_worker_payload(request)
        self.assertEqual(request.prompt, "glass shattering")
        self.assertEqual(request.seconds, 1.5)
        self.assertEqual(payload["seconds"], 1.5)
        self.assertEqual(payload["device"], main.STABLE_AUDIO_3_MEDIUM_DEVICE)
        self.assertIn("require_flash_attn", payload)

    def test_mismatched_seconds_and_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            main.StableAudio3MediumGenerateRequest(prompt="sound", seconds=1, duration=2)

    def test_generate_routes_return_audio_wav(self):
        self.assertEqual(main.STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR, main.SOUNDEFFECT_STORAGE_DIR)
        with (
            patch.object(main.manager, "run_worker", return_value=FAKE_WAV) as run_worker,
            patch.object(main, "wait_after_cuda_release"),
        ):
            response = self.client.post(
                "/v1/stableAudio/soundEffect",
                json={"prompt": "a short glass shatter", "seconds": 1},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, FAKE_WAV)
        self.assertEqual(run_worker.call_count, 1)
        self.assertEqual(run_worker.call_args.args[0]["prompt"], "a short glass shatter")
        saved_files = list(
            main.STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR.glob("stable_audio_3_medium_*.wav")
        )
        self.assertEqual(len(saved_files), 1)
        self.assertTrue(all(path.parent == main.SOUNDEFFECT_STORAGE_DIR for path in saved_files))


class WorkerRuntimeTests(unittest.TestCase):
    def test_uv_worker_uses_current_python_and_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            script = temp_dir / "worker.py"
            script.write_text("# fake worker\n", encoding="utf-8")
            config = runtime.WorkerConfig(
                worker_script=script,
                temp_dir=temp_dir,
                timeout=5,
                label="test",
                file_prefix="test",
            )

            class FakeProcess:
                pid = 12345
                returncode = 0

                def communicate(self, timeout=None):
                    Path(fake_command[-1]).write_bytes(FAKE_WAV)
                    return "worker ok", ""

                def poll(self):
                    return self.returncode

            fake_command: list[str] = []

            def fake_popen(command, **kwargs):
                fake_command.extend(command)
                return FakeProcess()

            with patch.object(runtime.subprocess, "Popen", side_effect=fake_popen):
                audio = runtime.run_local_worker({"prompt": "test"}, config)

            self.assertEqual(audio, FAKE_WAV)
            self.assertEqual(fake_command[0], sys.executable)
            self.assertNotIn("conda", fake_command)
            self.assertEqual(list(temp_dir.glob("test_*")), [])

    def test_flash_attention_is_optional_by_default_but_strict_when_requested(self):
        fake_torch = object()
        with patch.dict(sys.modules, {"flash_attn": None}):
            self.assertFalse(worker.check_flash_attention(fake_torch, required=False))
            with self.assertRaises(RuntimeError):
                worker.check_flash_attention(fake_torch, required=True)

    def test_worker_payload_reader_requires_json_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            path.write_text(json.dumps({"prompt": "test"}), encoding="utf-8")
            self.assertEqual(worker.read_payload(str(path))["prompt"], "test")


if __name__ == "__main__":
    unittest.main()
