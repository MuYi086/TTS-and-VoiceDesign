"""No-model regression tests for the standalone Step-Audio-EditX service."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "Step_Audio_EditX"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="step-audio-editx-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "model"
TOKENIZER_DIR = TEST_ROOT / "tokenizer"
CODE_DIR = TEST_ROOT / "upstream"
PROMPTS_DIR = TEST_ROOT / "prompts"
for directory in (MODEL_DIR, TOKENIZER_DIR, CODE_DIR, PROMPTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
(CODE_DIR / "tts.py").write_text("# mocked upstream\n", encoding="utf-8")
(CODE_DIR / "tokenizer.py").write_text("# mocked upstream\n", encoding="utf-8")

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "STEP_AUDIO_EDITX_MODEL_DIR": str(MODEL_DIR),
        "STEP_AUDIO_TOKENIZER_PATH": str(TOKENIZER_DIR),
        "STEP_AUDIO_EDITX_CODE_PATH": str(CODE_DIR),
        "PROMPTS_DIR": str(PROMPTS_DIR),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "STEP_AUDIO_EDITX_WORKER_TMP_DIR": str(TEST_ROOT / "worker-tmp"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "STEP_AUDIO_EDITX_REQUEST_TIMEOUT": "5",
        "STEP_AUDIO_EDITX_PORT": "8331",
    }
)
sys.path.insert(0, str(SERVICE_DIR))

spec = importlib.util.spec_from_file_location(
    "step_audio_editx_service_main_for_test", SERVICE_DIR / "main.py"
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class StepAudioEditXMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt_name = "migration-test.wav"
        main.prompt_audio_path(self.prompt_name).write_bytes(b"RIFF" + b"\0" * 40)

    def test_routes_health_and_uv_runtime(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/internal/unload_all"),
            ("POST", "/v1/upload_audio"),
            ("GET", "/v1/check/audio"),
            ("POST", "/v1/stepAudioEditx/edit"),
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
            set(payload), {"code", "paths", "available", "cuda", "runtime", "last_errors"}
        )
        self.assertTrue(payload["available"]["worker_script"])
        self.assertTrue(payload["available"]["step_audio_editx"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["runtime"]["flash_attention_policy"].split(";")[0], "not required")

    def test_request_contract_and_mocked_edit_returns_wav(self) -> None:
        from fastapi.testclient import TestClient

        request = main.StepAudioEditXEditRequest(
            prompt_audio=self.prompt_name,
            prompt_text="这是一条测试台词。",
            edit_type="emotion",
            edit_info="calmness",
        )
        self.assertEqual(request.generated_text, request.prompt_text)

        wav = b"RIFF" + b"\0" * 40
        with patch.object(main.manager, "run_worker", return_value=wav) as run_worker:
            response = TestClient(main.app).post(
                "/v1/stepAudioEditx/edit",
                json={
                    "prompt_audio": self.prompt_name,
                    "prompt_text": "这是一条测试台词。",
                    "edit_type": "emotion",
                    "edit_info": "calmness",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        run_worker.assert_called_once()
        self.assertEqual(
            run_worker.call_args.args[0]["prompt_wav_path"],
            str(main.prompt_audio_path(self.prompt_name)),
        )

        denoise = main.StepAudioEditXEditRequest(
            prompt_audio=self.prompt_name,
            edit_type="denoise",
        )
        self.assertIsNone(denoise.prompt_text)
        with self.assertRaises(ValueError):
            main.StepAudioEditXEditRequest(
                prompt_audio=self.prompt_name,
                prompt_text="台词",
                edit_type="emotion",
            )

    def test_worker_uses_current_uv_interpreter_and_cleans_files(self) -> None:
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0
            pid = None

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                command = captured["command"]
                output_path = Path(command[command.index("--output-wav") + 1])
                output_path.write_bytes(b"RIFF" + b"\0" * 40)
                return "worker ok", ""

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        payload = {
            "prompt_wav_path": str(main.prompt_audio_path(self.prompt_name)),
            "model_path": str(MODEL_DIR),
            "tokenizer_path": str(TOKENIZER_DIR),
            "code_path": str(CODE_DIR),
        }
        with patch.object(main.subprocess, "Popen", side_effect=fake_popen):
            audio_bytes = main.manager.run_worker(payload)

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertTrue(audio_bytes.startswith(b"RIFF"))
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

    def test_worker_failure_and_timeout_cleanup(self) -> None:
        class FailedProcess:
            returncode = 1
            pid = None

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                return "", "mock worker failure"

        payload = {
            "prompt_wav_path": str(main.prompt_audio_path(self.prompt_name)),
            "model_path": str(MODEL_DIR),
            "tokenizer_path": str(TOKENIZER_DIR),
            "code_path": str(CODE_DIR),
        }
        with patch.object(main.subprocess, "Popen", return_value=FailedProcess()):
            with self.assertRaisesRegex(RuntimeError, "mock worker failure"):
                main.manager.run_worker(payload)
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

        class TimeoutProcess:
            returncode = None
            pid = None

            def poll(self):
                return None if self.returncode is None else self.returncode

            def communicate(self, timeout=None):
                if self.returncode is None:
                    raise subprocess.TimeoutExpired("worker", timeout)
                return "", ""

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                self.returncode = -15

        timeout_process = TimeoutProcess()
        with patch.object(main.subprocess, "Popen", return_value=timeout_process):
            with self.assertRaisesRegex(RuntimeError, "超时"):
                main.manager.run_worker(payload)
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

    def test_legacy_api_proxy_is_removed(self) -> None:
        main_dir = REPOSITORY_DIR / "main"
        self.assertFalse((main_dir / "step_audio_editx.py").exists())
        self.assertNotIn(
            "/v1/stepAudioEditx/edit",
            (main_dir / "main.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "/v1/stepAudioEditx/edit",
            (SERVICE_DIR / "main.py").read_text(encoding="utf-8"),
        )

    def test_start_script_does_not_resolve_dependencies_at_runtime(self) -> None:
        start_script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        self.assertIn(
            'uv run --no-sync --project "$STEP_AUDIO_EDITX_PROJECT_DIR"',
            start_script,
        )
        self.assertNotIn("STEP_AUDIO_EDITX_UV_BASE_URL", start_script)
        self.assertNotIn("MAIN_DIR/step_audio_editx", start_script)

    def test_audio_processing_dependencies_are_declared(self) -> None:
        pyproject = (SERVICE_DIR / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"sox==1.5.0"', pyproject)
        self.assertIn('"ffmpeg-python==0.2.0"', pyproject)
        self.assertIn('"hdbscan==0.8.41"', pyproject)
        self.assertIn('"rotary-embedding-torch==0.8.9"', pyproject)
        self.assertIn('"torchcodec==0.9.1"', pyproject)


if __name__ == "__main__":
    unittest.main()
