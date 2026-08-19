"""No-model regression tests for the 8300 control-plane service."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "main"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="main-control-plane-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
CONTROL_ENV = {
    "STORAGE_DIR": str(TEST_ROOT / "storage"),
    "TIMBRE_STORAGE_DIR": str(TEST_ROOT / "storage/timbre"),
    "SOUNDEFFECT_STORAGE_DIR": str(TEST_ROOT / "storage/soundEffect"),
    "CLONE_STORAGE_DIR": str(TEST_ROOT / "storage/clone"),
    "PROMPTS_DIR": str(TEST_ROOT / "storage/clone"),
    "RUNTIME_CACHE_DIR": str(TEST_ROOT / "storage/.cache/runtime"),
    "GPU_LOCK_FILE": str(TEST_ROOT / "storage/.cache/runtime/gpu-runtime.lock"),
    "HOST": "127.0.0.1",
    "PORT": "8300",
    "MIMO_TTS_PROXY_URL": "http://127.0.0.1:8303/v1/mimo/timbre",
}
ORIGINAL_ENV = {key: os.environ.get(key) for key in CONTROL_ENV}
os.environ.update(CONTROL_ENV)
sys.path.insert(0, str(SERVICE_DIR))
sys.modules.pop("gpu_runtime", None)
spec = importlib.util.spec_from_file_location(
    "control_plane_main_for_test", SERVICE_DIR / "main.py"
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)
for key, value in ORIGINAL_ENV.items():
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


class MainControlPlaneMigrationTests(unittest.TestCase):
    def test_control_plane_routes_and_health_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("GET", "/v1/control"),
            ("POST", "/v1/mimo/timbre"),
            ("POST", "/v1/upload_audio"),
            ("GET", "/v1/check/audio"),
        }
        actual_routes = {
            (method, route.path)
            for route in main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method in {"GET", "POST"}
        }
        self.assertTrue(expected_routes.issubset(actual_routes))

        response = TestClient(main.app).get("/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["runtime"]["service_role"], "control_plane")
        self.assertEqual(
            payload["runtime"]["model_inference"],
            "delegated to standalone services",
        )

    def test_upload_and_check_preserve_timbre_reference_without_clone_copy(self) -> None:
        from fastapi.testclient import TestClient

        timbre_path = Path(main.TIMBRE_STORAGE_DIR) / "designed.wav"
        timbre_path.write_bytes(b"RIFF-designed-voice")
        client = TestClient(main.app)

        response = client.post(
            "/v1/upload_audio",
            files={"audio": ("designed.wav", timbre_path.read_bytes(), "audio/wav")},
            data={"full_path": "designed.wav"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse((Path(main.PROMPTS_DIR) / main.hash_filename("designed.wav")).exists())
        self.assertTrue(
            (
                Path(main.TIMBRE_REFERENCE_DIR) / f"{main.hash_filename('designed.wav')}.path"
            ).exists()
        )
        checked = client.get("/v1/check/audio", params={"file_name": "designed.wav"})
        self.assertEqual(checked.status_code, 200)
        self.assertTrue(checked.json()["exists"])

    def test_mimo_route_is_a_proxy_only(self) -> None:
        from fastapi.testclient import TestClient

        with patch.object(
            main,
            "forward_mimo_design_request",
            return_value=(200, b"RIFF-proxy", "audio/wav"),
        ) as forward:
            response = TestClient(main.app).post(
                "/v1/mimo/timbre",
                json={"voice_description": "温柔的女声", "text": "你好。"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"RIFF-proxy")
        forward.assert_called_once()

    def test_api_directory_is_removed_after_migration(self) -> None:
        self.assertFalse((REPOSITORY_DIR / "api").exists())
        self.assertTrue((SERVICE_DIR / "main.py").exists())
        self.assertTrue((SERVICE_DIR / "gpu_runtime.py").exists())


if __name__ == "__main__":
    unittest.main()
