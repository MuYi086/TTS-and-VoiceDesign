"""No-model regression tests for the standalone MOSS VoiceGenerator service."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "moss_voiceGenerator"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="moss-voicegenerator-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "MOSS_VOICEGENERATOR_MODEL_DIR": str(TEST_ROOT / "model"),
        "MOSS_AUDIO_TOKENIZER_PATH": str(TEST_ROOT / "codec"),
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "TIMBRE_STORAGE_DIR": str(TEST_ROOT / "storage" / "timbre"),
        "TTS_OUTPUT_DIR": str(TEST_ROOT / "legacy-clone"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "MOSS_VOICEGENERATOR_REQUEST_TIMEOUT": "5",
        "MOSS_VOICEGENERATOR_PORT": "8302",
    }
)
(TEST_ROOT / "model").mkdir(parents=True, exist_ok=True)
(TEST_ROOT / "codec").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SERVICE_DIR))

spec = importlib.util.spec_from_file_location(
    "moss_voicegenerator_service_main",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class MossVoiceGeneratorMigrationTests(unittest.TestCase):
    def test_route_and_health_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/internal/unload_all"),
            ("POST", "/v1/moss/timbre"),
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
        self.assertTrue(payload["available"]["moss_voicegenerator_model_dir"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["available"]["python"], sys.executable)
        self.assertEqual(
            payload["runtime"]["flash_attention_policy"].split(";")[0],
            "optional",
        )

    def test_request_defaults_and_worker_payload(self) -> None:
        request = main.MossDesignRequest(voice_description="成年女性，清晰自然。")
        payload = main.manager.build_worker_payload(request)

        self.assertEqual(request.text, "这是生成的参考音频预览。")
        self.assertEqual(request.save_as, "designed_voice.wav")
        self.assertEqual(payload["model_path"], str(TEST_ROOT / "model"))
        self.assertEqual(payload["codec_path"], str(TEST_ROOT / "codec"))
        self.assertEqual(payload["max_new_tokens"], 4096)
        self.assertEqual(payload["local_files_only"], True)

    def test_design_route_returns_wav_without_loading_model(self) -> None:
        from fastapi.testclient import TestClient

        wav = b"RIFF" + b"\0" * 40
        with patch.object(main.manager, "run_worker", return_value=wav) as run_worker:
            response = TestClient(main.app).post(
                "/v1/moss/timbre",
                json={
                    "voice_description": "成年女性，温柔、清晰。",
                    "text": "你好。",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        run_worker.assert_called_once()
        self.assertEqual(run_worker.call_args.args[0]["text"], "你好。")
        saved_files = list(main.TIMBRE_STORAGE_DIR.glob("moss_voicegenerator_*.wav"))
        self.assertTrue(saved_files)
        self.assertTrue(all(path.parent == main.TIMBRE_STORAGE_DIR for path in saved_files))
        self.assertFalse((TEST_ROOT / "legacy-clone").exists())

    def test_worker_uses_uv_interpreter_and_cleans_temporary_files(self) -> None:
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

        with (
            patch.object(main, "is_moss_codec_path_ready", return_value=True),
            patch.object(main.subprocess, "Popen", side_effect=fake_popen),
        ):
            audio_bytes = main.manager.run_worker({"voice_description": "mock"})

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertTrue(audio_bytes.startswith(b"RIFF"))
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

    def test_moss_processor_loader_does_not_forward_local_files_only(self) -> None:
        worker_spec = importlib.util.spec_from_file_location(
            "moss_voicegenerator_worker_for_test",
            SERVICE_DIR / "worker.py",
        )
        assert worker_spec and worker_spec.loader
        moss_worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(moss_worker)

        captured: dict[str, object] = {}

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(model_path, **kwargs):
                captured["model_path"] = model_path
                captured["kwargs"] = kwargs
                return object()

        moss_worker.load_moss_processor(
            FakeAutoProcessor,
            TEST_ROOT / "model",
            TEST_ROOT / "codec",
        )

        self.assertEqual(captured["model_path"], str(TEST_ROOT / "model"))
        self.assertNotIn("local_files_only", captured["kwargs"])
        self.assertTrue(captured["kwargs"]["trust_remote_code"])


if __name__ == "__main__":
    unittest.main()
