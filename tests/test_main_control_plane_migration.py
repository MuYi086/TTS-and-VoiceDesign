"""8300 控制面服务的无模型回归测试。"""

from __future__ import annotations

# 控制面测试确保 8300 只做存储、健康检查和 MiMo 转发，不加载模型。
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
            ("POST", "/v1/audio/export"),
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
        self.assertEqual(payload["audio_export"]["sample_rate"], 48000)
        self.assertEqual(payload["audio_export"]["profiles"], ["standard", "balanced", "immersive"])

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
                Path(main.TIMBRE_REFERENCE_DIR) / f"{main.hash_filename('designed.wav')}.json"
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

    def test_audio_export_streams_processed_result_and_cleans_response_file(self) -> None:
        from fastapi.testclient import TestClient

        output_dir = TEST_ROOT / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "result.mp3"
        output_path.write_bytes(b"processed-mp3")
        cleanup_called: list[bool] = []

        def fake_export(staged, *, profile: str, output_format: str):
            self.assertEqual(profile, "balanced")
            self.assertEqual(output_format, "mp3")
            self.assertTrue(staged.path.is_file())
            staged.path.unlink()
            return SimpleNamespace(
                path=output_path,
                media_type="audio/mpeg",
                download_name="unitale_balanced.mp3",
                cleanup=lambda: cleanup_called.append(True),
            )

        with patch.object(
            main.spatial_audio_processor, "export", side_effect=fake_export
        ) as export:
            response = TestClient(main.app).post(
                "/v1/audio/export",
                files={"audio": ("mix.wav", b"RIFF-mixed-audio", "audio/wav")},
                data={"profile": "balanced", "output_format": "mp3"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"processed-mp3")
        self.assertEqual(response.headers["x-audio-sample-rate"], "48000")
        self.assertEqual(response.headers["x-audio-profile"], "balanced")
        self.assertIn("unitale_balanced.mp3", response.headers["content-disposition"])
        self.assertIn(
            "Content-Disposition",
            response.headers["access-control-expose-headers"],
        )
        self.assertTrue(cleanup_called)
        export.assert_called_once()

    def test_audio_export_rejects_unknown_profile_before_processing(self) -> None:
        from fastapi.testclient import TestClient

        with patch.object(main.spatial_audio_processor, "export") as export:
            response = TestClient(main.app).post(
                "/v1/audio/export",
                files={"audio": ("mix.wav", b"RIFF-mixed-audio", "audio/wav")},
                data={"profile": "unknown", "output_format": "wav"},
            )

        self.assertEqual(response.status_code, 422)
        export.assert_not_called()

    def test_api_directory_is_removed_after_migration(self) -> None:
        self.assertFalse((REPOSITORY_DIR / "api").exists())
        self.assertTrue((SERVICE_DIR / "main.py").exists())
        self.assertTrue((SERVICE_DIR / "gpu_runtime.py").exists())


if __name__ == "__main__":
    unittest.main()
