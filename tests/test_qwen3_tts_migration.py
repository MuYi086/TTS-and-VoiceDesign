"""No-model regression tests for the standalone Qwen3-TTS service."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
QWEN3_TTS_DIR = REPOSITORY_DIR / "qwen3_tts"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="qwen3-tts-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)

os.environ.update(
    {
        "PROMPTS_DIR": str(TEST_ROOT / "prompts"),
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "TIMBRE_STORAGE_DIR": str(TEST_ROOT / "storage" / "timbre"),
        "TTS_OUTPUT_DIR": str(TEST_ROOT / "legacy-clone"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "QWEN3_TTS_OUTPUT_DIR": str(TEST_ROOT / "output"),
        "QWEN3_TTS_MODEL_DIR": str(TEST_ROOT / "model"),
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
    }
)
(TEST_ROOT / "model").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(QWEN3_TTS_DIR))

import main  # noqa: E402


class Qwen3TtsMigrationTests(unittest.TestCase):
    def test_route_and_health_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/internal/unload_all"),
            ("POST", "/v1/upload_audio"),
            ("GET", "/v1/check/audio"),
            ("POST", "/v2/synthesize"),
        }
        actual_routes = {
            (method, route.path)
            for route in main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertTrue(expected_routes.issubset(actual_routes))

        with patch.object(main, "cuda_status", return_value={"available": False, "source": "test"}):
            response = TestClient(main.app).get("/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload),
            {"code", "paths", "available", "cuda", "runtime", "last_errors"},
        )
        self.assertTrue(payload["available"]["worker_script"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["available"]["python"], sys.executable)

    def test_upload_and_check_audio_preserve_prompt_sidecar_contract(self) -> None:
        from fastapi.testclient import TestClient

        filename = "migration-test.wav"
        prompt_text = "这是测试参考文本。"
        client = TestClient(main.app)
        response = client.post(
            "/v1/upload_audio",
            files={"audio": (filename, io.BytesIO(b"RIFF" + b"\0" * 40), "audio/wav")},
            data={"full_path": filename, "prompt_text": prompt_text},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["has_prompt_text"], True)

        check = client.get("/v1/check/audio", params={"file_name": filename})
        self.assertEqual(check.status_code, 200)
        self.assertEqual(
            check.json(),
            {"code": 200, "exists": True, "has_prompt_text": True},
        )
        self.assertEqual(
            (TEST_ROOT / "prompts" / f"{main.hash_filename(filename)}.prompt.txt").read_text(
                encoding="utf-8"
            ),
            prompt_text,
        )

    def test_timbre_reference_upload_does_not_copy_audio_to_clone(self) -> None:
        from fastapi.testclient import TestClient

        filename = "designed-voice.wav"
        content = b"RIFF" + b"\x05\x06" * 32
        timbre_path = TEST_ROOT / "storage" / "timbre" / "qwen_voicedesign_fixture.wav"
        timbre_path.write_bytes(content)

        response = TestClient(main.app).post(
            "/v1/upload_audio",
            files={"audio": (filename, io.BytesIO(content), "audio/wav")},
            data={"full_path": filename, "prompt_text": "设计音色参考文本。"},
        )

        self.assertEqual(response.status_code, 200)
        clone_path = TEST_ROOT / "prompts" / main.hash_filename(filename)
        self.assertFalse(clone_path.exists())
        self.assertEqual(main.prompt_audio_path(filename), str(timbre_path))
        checked = TestClient(main.app).get("/v1/check/audio", params={"file_name": filename})
        self.assertEqual(checked.status_code, 200)
        self.assertTrue(checked.json()["exists"])
        self.assertEqual(
            main.load_prompt_text_sidecar(filename),
            "设计音色参考文本。",
        )
        payload = main.Qwen3TtsSynthesizeRequest(text="你好。", audio_path=filename)
        self.assertEqual(
            main.manager.build_worker_payload(payload)["ref_audio_path"],
            str(timbre_path),
        )

    def test_request_contract_rejects_style_prompt(self) -> None:
        valid = main.Qwen3TtsSynthesizeRequest.model_validate(
            {"text": "你好", "audio_path": "reference.wav"}
        )
        self.assertEqual(valid.audio_path, "reference.wav")
        with self.assertRaises(ValueError):
            main.Qwen3TtsSynthesizeRequest.model_validate(
                {
                    "text": "你好",
                    "audio_path": "reference.wav",
                    "style_prompt": "应该被拒绝",
                }
            )

    def test_worker_uses_current_uv_interpreter_and_one_shot_command(self) -> None:
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0
            pid = None

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                output_path = Path(captured["command"][-1])
                output_path.write_bytes(b"RIFF" + b"\0" * 40)
                return "worker ok", ""

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        manager = main.Qwen3TtsWorkerManager()
        with patch.object(main.subprocess, "Popen", side_effect=fake_popen):
            audio_bytes = manager.run_worker({"text": "mock request"})

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.QWEN3_TTS_WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertTrue(audio_bytes.startswith(b"RIFF"))
        self.assertTrue(list((TEST_ROOT / "output").glob("qwen3_tts_*.wav")))


if __name__ == "__main__":
    unittest.main()
