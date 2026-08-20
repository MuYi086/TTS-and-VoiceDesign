"""独立 LongCat uv 服务的无模型回归测试。"""

from __future__ import annotations

# 迁移测试保护 LongCat 的 uv 项目、最终路由和 worker 解释器选择。
import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "LongCat_AudioDiT_3.5B_bf16"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="longcat-audiodit-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "model"
PROMPTS_DIR = TEST_ROOT / "prompts"
REPO_DIR = TEST_ROOT / "repo"
TOKENIZER_DIR = TEST_ROOT / "tokenizer"
for directory in (MODEL_DIR, PROMPTS_DIR, REPO_DIR, TOKENIZER_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "LONGCAT_AUDIODIT_MODEL_DIR": str(MODEL_DIR),
        "LONGCAT_AUDIODIT_REPO_PATH": str(REPO_DIR),
        "LONGCAT_AUDIODIT_TOKENIZER_PATH": str(TOKENIZER_DIR),
        "PROMPTS_DIR": str(PROMPTS_DIR),
        "TIMBRE_STORAGE_DIR": str(TEST_ROOT / "timbre"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "LONGCAT_AUDIODIT_OUTPUT_DIR": str(TEST_ROOT / "output"),
        "LONGCAT_AUDIODIT_WORKER_TMP_DIR": str(TEST_ROOT / "worker-tmp"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "LONGCAT_AUDIODIT_REQUEST_TIMEOUT": "5",
    }
)
sys.path.insert(0, str(SERVICE_DIR))

spec = importlib.util.spec_from_file_location(
    "longcat_audiodit_service_main_for_test", SERVICE_DIR / "main.py"
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class LongCatAudioDitMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filename = "migration-test.wav"
        self.prompt_text = "这是一条准确的参考文本。"
        (PROMPTS_DIR / main.hash_filename(self.filename)).write_bytes(b"RIFF" + b"\0" * 40)
        main.reference_store.prompt_sidecar_path(self.filename).write_text(
            self.prompt_text,
            encoding="utf-8",
        )

    def test_routes_health_uv_runtime_and_flash_policy(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/upload_audio"),
            ("GET", "/v1/check/audio"),
            ("POST", "/v1/longCat/clone"),
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
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["available"]["python"], sys.executable)
        self.assertEqual(
            payload["runtime"]["flash_attention_policy"].split(";")[0],
            "not required",
        )
        self.assertFalse(payload["available"]["flash_attn"])

    def test_upload_and_check_audio_preserve_webui_hash_contract(self) -> None:
        from fastapi.testclient import TestClient

        filename = "upload-test.wav"
        content = b"RIFF" + b"\x01\x02" * 32
        response = TestClient(main.app).post(
            "/v1/upload_audio",
            files={"audio": (filename, io.BytesIO(content), "audio/wav")},
            data={"full_path": filename, "prompt_text": self.prompt_text},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(response.json()["size_bytes"], len(content))

        check = TestClient(main.app).get("/v1/check/audio", params={"file_name": filename})
        self.assertEqual(check.status_code, 200)
        self.assertEqual(check.json()["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(check.json()["size_bytes"], len(content))
        self.assertTrue(check.json()["has_prompt_text"])

    def test_timbre_reference_upload_does_not_copy_audio_to_clone(self) -> None:
        from fastapi.testclient import TestClient

        filename = "designed-voice.wav"
        content = b"RIFF" + b"\x03\x04" * 32
        timbre_path = Path(main.TIMBRE_STORAGE_DIR) / "longcat-designed.wav"
        timbre_path.write_bytes(content)

        response = TestClient(main.app).post(
            "/v1/upload_audio",
            files={"audio": (filename, io.BytesIO(content), "audio/wav")},
            data={"full_path": filename, "prompt_text": "设计音色参考文本。"},
        )

        self.assertEqual(response.status_code, 200)
        clone_path = PROMPTS_DIR / main.hash_filename(filename)
        self.assertFalse(clone_path.exists())
        self.assertEqual(main.prompt_audio_path(filename), str(timbre_path))
        self.assertTrue(Path(main.timbre_reference_map_path(filename)).exists())
        self.assertEqual(main.load_prompt_text_sidecar(filename), "设计音色参考文本。")
        checked = TestClient(main.app).get("/v1/check/audio", params={"file_name": filename})
        self.assertEqual(checked.status_code, 200)
        self.assertTrue(checked.json()["exists"])

    def test_request_contract_and_payload_preserve_longcat_fields(self) -> None:
        request = main.LongCatAudioDitSynthesizeRequest(
            text="# 目标台词",
            audio_path=self.filename,
            guidance_method="apg",
            nfe=20,
            vae_dtype="float16",
        )
        payload = main.manager.build_worker_payload(request)
        self.assertEqual(payload["text"], "目标台词")
        self.assertEqual(payload["prompt_text"], self.prompt_text)
        self.assertEqual(payload["guidance_method"], "apg")
        self.assertEqual(payload["nfe"], 20)
        self.assertEqual(payload["vae_dtype"], "float16")

        with self.assertRaises(ValueError):
            main.LongCatAudioDitSynthesizeRequest.model_validate(
                {
                    "text": "台词",
                    "audio_path": self.filename,
                    "style_prompt": "不能用于克隆",
                }
            )

    def test_synthesize_returns_wav_without_loading_model(self) -> None:
        from fastapi.testclient import TestClient

        wav = b"RIFF" + b"\0" * 40
        with patch.object(main.manager, "run_worker", return_value=wav) as run_worker:
            response = TestClient(main.app).post(
                "/v1/longCat/clone",
                json={
                    "text": "目标台词",
                    "audio_path": self.filename,
                    "prompt_text": self.prompt_text,
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        run_worker.assert_called_once()

    def test_worker_uses_current_uv_interpreter_and_cleans_temp_files(self) -> None:
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

        payload = main.manager.build_worker_payload(
            main.LongCatAudioDitSynthesizeRequest(
                text="目标台词",
                audio_path=self.filename,
                prompt_text=self.prompt_text,
            )
        )
        with patch.object(main.subprocess, "Popen", side_effect=fake_popen):
            audio_bytes = main.manager.run_worker(payload)

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.LONGCAT_AUDIODIT_WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertTrue(audio_bytes.startswith(b"RIFF"))
        self.assertTrue(list((TEST_ROOT / "output").glob("longcat_audiodit_*.wav")))
        self.assertEqual(list((TEST_ROOT / "worker-tmp").iterdir()), [])

    def test_start_script_routes_longcat_only_to_uv(self) -> None:
        script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        self.assertIn(
            'uv run --no-sync --project "$LONGCAT_AUDIODIT_PROJECT_DIR"',
            script,
        )
        self.assertNotIn("main/longcat_audiodit_api.py", script)
        self.assertNotIn("LONGCAT_AUDIODIT_CONDA_ENV", script)
        self.assertFalse((REPOSITORY_DIR / "main/longcat_audiodit_api.py").exists())
        self.assertFalse((REPOSITORY_DIR / "main/longcat_audiodit_worker.py").exists())


if __name__ == "__main__":
    unittest.main()
