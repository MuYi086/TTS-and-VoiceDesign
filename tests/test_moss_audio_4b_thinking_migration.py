"""MOSS-Audio-4B-Thinking 独立服务的无模型回归测试。"""

from __future__ import annotations

# 测试只替换 worker 进程，不下载模型权重、不导入 Torch，也不执行 CUDA 推理。
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "moss_audio_4b_thinking"
TEST_RUNTIME = tempfile.TemporaryDirectory(prefix="moss-audio-4b-thinking-tests-")
TEST_ROOT = Path(TEST_RUNTIME.name)
MODEL_DIR = TEST_ROOT / "model"
DEPENDENCY_DIR = TEST_ROOT / "moss-audio"

os.environ.update(
    {
        "HF_MIRROR_DIR": str(TEST_ROOT / "hf-mirror"),
        "MOSS_AUDIO_4B_THINKING_MODEL_DIR": str(MODEL_DIR),
        "MOSS_AUDIO_4B_THINKING_DEPENDENCY_PATH": str(DEPENDENCY_DIR),
        "STORAGE_DIR": str(TEST_ROOT / "storage"),
        "RUNTIME_CACHE_DIR": str(TEST_ROOT / "cache"),
        "GPU_LOCK_FILE": str(TEST_ROOT / "cache" / "gpu.lock"),
        "LOCAL_FILES_ONLY": "1",
        "CUDA_RELEASE_DELAY": "0",
        "MOSS_AUDIO_4B_THINKING_REQUEST_TIMEOUT": "5",
        "MOSS_AUDIO_4B_THINKING_PORT": "8341",
    }
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(DEPENDENCY_DIR / "src").mkdir(parents=True, exist_ok=True)
for filename in ("audio_io.py", "modeling_moss_audio.py", "processing_moss_audio.py"):
    (DEPENDENCY_DIR / "src" / filename).touch()

sys.path.insert(0, str(SERVICE_DIR))
spec = importlib.util.spec_from_file_location(
    "moss_audio_4b_thinking_service_main",
    SERVICE_DIR / "main.py",
)
assert spec and spec.loader
main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = main
spec.loader.exec_module(main)


class MossAudio4BThinkingMigrationTests(unittest.TestCase):
    """验证 HTTP 契约、临时文件回收和 worker 启动边界。"""

    def test_instruct_variant_uses_instruct_model_and_upstream_defaults(self) -> None:
        """验证同一项目以 Instruct 变体启动时不会回退到 Thinking 权重。"""
        instruct_root = TEST_ROOT / "instruct-variant"
        instruct_model_dir = instruct_root / "model"
        instruct_dependency_dir = instruct_root / "moss-audio"
        instruct_model_dir.mkdir(parents=True, exist_ok=True)
        source_dir = instruct_dependency_dir / "src"
        source_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("audio_io.py", "modeling_moss_audio.py", "processing_moss_audio.py"):
            (source_dir / filename).touch()

        environment = {
            "MOSS_AUDIO_4B_VARIANT": "instruct",
            "MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR": str(instruct_model_dir),
            "MOSS_AUDIO_4B_INSTRUCT_DEPENDENCY_PATH": str(instruct_dependency_dir),
            "MOSS_AUDIO_4B_INSTRUCT_PORT": "8342",
            "RUNTIME_CACHE_DIR": str(instruct_root / "cache"),
            "MOSS_AUDIO_4B_INSTRUCT_WORKER_TMP_DIR": str(instruct_root / "cache" / "worker"),
            "CUDA_RELEASE_DELAY": "0",
        }
        module_name = "moss_audio_4b_instruct_service_for_test"
        with patch.dict(os.environ, environment, clear=False):
            instruct_spec = importlib.util.spec_from_file_location(
                module_name,
                SERVICE_DIR / "main.py",
            )
            assert instruct_spec and instruct_spec.loader
            instruct = importlib.util.module_from_spec(instruct_spec)
            sys.modules[module_name] = instruct
            try:
                instruct_spec.loader.exec_module(instruct)
                request = instruct.MossAudioThinkingRequest()
                audio_path = instruct_root / "input.wav"
                audio_path.write_bytes(b"RIFF" + b"\0" * 40)
                payload = instruct.manager.build_worker_payload(request, audio_path)

                self.assertEqual(instruct.API_PORT, 8342)
                self.assertEqual(instruct.MODEL_VARIANT, "instruct")
                self.assertEqual(instruct.MODEL_SERVICE_NAME, "MOSS-Audio-4B-Instruct")
                self.assertEqual(request.prompt, "Describe this audio.")
                self.assertTrue(request.do_sample)
                self.assertEqual(payload["model_path"], str(instruct_model_dir))
                self.assertEqual(payload["dependency_path"], str(instruct_dependency_dir))
            finally:
                sys.modules.pop(module_name, None)

    def test_start_script_exposes_instruct_service_on_8342(self) -> None:
        """验证 start.sh 传入 Instruct 变体和独立 8342 端口。"""
        source = (REPOSITORY_DIR / "start.sh").read_text(encoding="utf-8")

        self.assertIn(
            'MOSS_AUDIO_4B_INSTRUCT_PORT="${MOSS_AUDIO_4B_INSTRUCT_PORT:-8342}"',
            source,
        )
        self.assertIn(
            'MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR="${MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-4B-Instruct}"',
            source,
        )
        self.assertIn('MOSS_AUDIO_4B_VARIANT="instruct"', source)
        self.assertIn("moss_audio_4b_instruct_pid", source)

    def test_route_and_health_contract(self) -> None:
        from fastapi.testclient import TestClient

        expected_routes = {
            ("GET", "/v1/health"),
            ("POST", "/v1/mossAudioThinking/understand"),
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
        self.assertTrue(payload["available"]["moss_audio_4b_thinking_model_dir"])
        self.assertTrue(payload["available"]["moss_audio_source"])
        self.assertEqual(payload["runtime"]["worker_runtime"], "uv")
        self.assertEqual(payload["runtime"]["worker_python"], sys.executable)

    def test_request_defaults_and_worker_payload(self) -> None:
        audio_path = TEST_ROOT / "uploaded.wav"
        audio_path.write_bytes(b"RIFF" + b"\0" * 40)
        request = main.MossAudioThinkingRequest()
        payload = main.manager.build_worker_payload(request, audio_path)

        self.assertEqual(request.prompt, main.DEFAULT_PROMPT)
        self.assertEqual(request.max_new_tokens, 1024)
        self.assertEqual(payload["model_path"], str(MODEL_DIR))
        self.assertEqual(payload["dependency_path"], str(DEPENDENCY_DIR))
        self.assertEqual(payload["audio_path"], str(audio_path))
        self.assertTrue(payload["local_files_only"])
        self.assertFalse(payload["do_sample"])

    def test_understand_route_streams_upload_and_returns_text_without_model_load(self) -> None:
        from fastapi.testclient import TestClient

        with patch.object(
            main.manager,
            "run_worker",
            return_value={"text": "遇到我们的时候你才是挑战者。", "elapsed_seconds": 2.72},
        ) as run_worker:
            response = TestClient(main.app).post(
                "/v1/mossAudioThinking/understand",
                files={"audio": ("sample.wav", b"RIFF" + b"\0" * 40, "audio/wav")},
                data={"prompt": "请准确转写这段音频，仅输出转写文本。", "strip_thinking": "true"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["code"], 200)
        self.assertEqual(response.json()["text"], "遇到我们的时候你才是挑战者。")
        self.assertEqual(response.json()["audio"]["suffix"], ".wav")
        self.assertEqual(response.json()["elapsed_seconds"], 2.72)
        run_worker.assert_called_once()
        self.assertTrue(run_worker.call_args.args[0]["strip_thinking"])
        self.assertEqual(list((Path(main.RUNTIME_CACHE_DIR) / "uploads").iterdir()), [])

    def test_understand_route_rejects_non_audio_upload(self) -> None:
        from fastapi.testclient import TestClient

        with patch.object(main.manager, "run_worker") as run_worker:
            response = TestClient(main.app).post(
                "/v1/mossAudioThinking/understand",
                files={"audio": ("notes.txt", b"not an audio file", "text/plain")},
            )

        self.assertEqual(response.status_code, 422)
        run_worker.assert_not_called()

    def test_worker_uses_uv_interpreter_and_cleans_temporary_files(self) -> None:
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0
            pid = None

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                output_path = Path(captured["command"][-1])
                output_path.write_text(
                    json.dumps({"text": "mock result", "elapsed_seconds": 0.1}),
                    encoding="utf-8",
                )
                return "worker ok", ""

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return FakeProcess()

        with patch.object(main.subprocess, "Popen", side_effect=fake_popen):
            result = main.manager.run_worker({"prompt": "mock"})

        command = captured["command"]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], main.WORKER_SCRIPT)
        self.assertNotIn("conda", command)
        self.assertEqual(result["text"], "mock result")
        self.assertEqual(list(Path(main.WORKER_TMP_DIR).iterdir()), [])

    def test_worker_keeps_source_generation_defaults_and_strips_thinking(self) -> None:
        worker_spec = importlib.util.spec_from_file_location(
            "moss_audio_4b_thinking_worker_for_test",
            SERVICE_DIR / "worker.py",
        )
        assert worker_spec and worker_spec.loader
        worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(worker)

        self.assertEqual(worker.strip_thinking("<think>内部推理</think>\n最终答案"), "最终答案")
        self.assertEqual(worker.strip_thinking("没有推理标签"), "没有推理标签")
        self.assertEqual(
            worker.build_generation_kwargs({"max_new_tokens": 128, "do_sample": False}),
            {
                "max_new_tokens": 128,
                "num_beams": 1,
                "use_cache": True,
                "do_sample": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
