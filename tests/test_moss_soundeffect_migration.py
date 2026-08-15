"""No-model regression tests for the standalone MOSS-SoundEffect uv service."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "moss_soundEffect"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="moss-soundeffect-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "model"
CODE_DIR = TEST_ROOT / "MOSS-TTS"
PACKAGE_DIR = CODE_DIR / "moss_soundeffect_v2"
CACHE_DIR = TEST_ROOT / "cache"
for directory in (MODEL_DIR, PACKAGE_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

for relative_path in (
    "model_index.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/vae_128d_48k.pth",
):
    path = MODEL_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")

(PACKAGE_DIR / "__init__.py").write_text(
    "class MossSoundEffectPipeline: pass\n",
    encoding="utf-8",
)

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "MOSS_SOUNDEFFECT_MODEL_DIR": str(MODEL_DIR),
        "MOSS_SOUNDEFFECT_CODE_PATH": str(CODE_DIR),
        "RUNTIME_CACHE_DIR": str(CACHE_DIR),
        "GPU_LOCK_FILE": str(CACHE_DIR / "gpu.lock"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "MOSS_SOUNDEFFECT_REQUEST_TIMEOUT": "5",
        "MOSS_SOUNDEFFECT_PORT": "18311",
    }
)

sys.path.insert(0, str(SERVICE_DIR))
sys.modules.pop("runtime", None)
spec = importlib.util.spec_from_file_location(
    "moss_soundeffect_service_main_for_test",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)
runtime = sys.modules["runtime"]


class MossSoundEffectMigrationTests(unittest.TestCase):
    def test_routes_health_and_flash_policy(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/internal/unload_all"),
            ("POST", "/v1/generate"),
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

        with patch.object(
            main, "cuda_status", return_value={"available": False, "source": "test"}
        ):
            response = TestClient(main.app).get("/v1/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload),
            {"code", "paths", "available", "cuda", "runtime", "last_errors"},
        )
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["available"]["python"], sys.executable)
        self.assertTrue(payload["available"]["source_repo"])
        self.assertTrue(payload["available"]["moss_package"])
        self.assertFalse(payload["available"]["flash_attn"])
        self.assertTrue(
            payload["runtime"]["flash_attention_policy"].startswith("not required")
        )

    def test_request_validation_and_payload_preserve_legacy_fields(self) -> None:
        request = main.SoundEffectGenerateRequest(
            prompt="  木门缓慢推开，铰链发出吱呀声  ",
            seconds=2.5,
            num_inference_steps=80,
            cfg_scale=3.5,
            sigma_shift=4.0,
            seed=12,
            device="cuda:0",
            torch_dtype="float16",
        )
        payload = main.manager.build_worker_payload(request)
        self.assertEqual(payload["prompt"], "木门缓慢推开，铰链发出吱呀声")
        self.assertEqual(payload["seconds"], 2.5)
        self.assertEqual(payload["num_inference_steps"], 80)
        self.assertEqual(payload["cfg_scale"], 3.5)
        self.assertEqual(payload["sigma_shift"], 4.0)
        self.assertEqual(payload["seed"], 12)
        self.assertEqual(payload["code_path"], str(CODE_DIR))

        with self.assertRaises(ValueError):
            main.SoundEffectGenerateRequest(prompt=" ")

        with self.assertRaises(ValueError):
            main.SoundEffectGenerateRequest(prompt="声效", seconds=31)

    def test_generate_alias_returns_wav_without_model(self) -> None:
        from fastapi.testclient import TestClient

        wav = b"RIFF" + b"\0" * 40
        with patch.object(main.manager, "run_worker", return_value=wav) as run_worker:
            response = TestClient(main.app).post(
                "/v2/synthesize",
                json={"prompt": "近距离玻璃碎裂", "seconds": 1},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, wav)
        run_worker.assert_called_once()

    def test_uv_worker_uses_current_interpreter_and_cleans_temp_files(self) -> None:
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

        config = runtime.UvWorkerConfig(
            python_executable=sys.executable,
            worker_script=str(SERVICE_DIR / "worker.py"),
            model_dir=str(MODEL_DIR),
            code_path=str(CODE_DIR),
            temp_dir=str(TEST_ROOT / "worker-tmp"),
            timeout=5,
            label="SoundEffect",
            file_prefix="soundeffect",
        )
        payload = {"prompt": "测试", "seconds": 1}
        with patch.object(runtime.subprocess, "Popen", side_effect=fake_popen):
            audio_bytes = runtime.run_uv_worker(payload, config)

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(SERVICE_DIR / "worker.py"))
        self.assertNotIn("conda", command)
        self.assertTrue(audio_bytes.startswith(b"RIFF"))
        self.assertEqual(list((TEST_ROOT / "worker-tmp").iterdir()), [])

    def test_start_script_routes_soundeffect_to_uv_with_conda_fallback(self) -> None:
        script = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")
        self.assertIn(
            'uv run --no-sync --project "$MOSS_SOUNDEFFECT_PROJECT_DIR"',
            script,
        )
        self.assertIn("MOSS_SOUNDEFFECT_RUNTIME", script)
        self.assertIn('elif [[ "$MOSS_SOUNDEFFECT_RUNTIME" == "conda" ]]', script)
        self.assertIn('python "$API_DIR/soundeffect_api.py"', script)


if __name__ == "__main__":
    unittest.main()
