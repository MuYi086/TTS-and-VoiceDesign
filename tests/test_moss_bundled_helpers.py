import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import moss_api
from moss_tts_worker import generation_frame_budget


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = 32001
        self.returncode = None

    def communicate(self, timeout=None):
        output_index = self.command.index("--output-wav") + 1
        Path(self.command[output_index]).write_bytes(b"RIFF-moss-wave")
        self.returncode = 0
        return "worker completed", ""


class RetryableCudaProcess:
    calls = 0
    payloads = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = 32002 + type(self).calls
        self.returncode = None
        request_index = self.command.index("--input-json") + 1
        with open(self.command[request_index], "r", encoding="utf-8") as f:
            type(self).payloads.append(json.load(f))
        type(self).calls += 1

    def communicate(self, timeout=None):
        if type(self).calls == 1:
            self.returncode = 1
            return "", "RuntimeError: CUDA driver error: device not ready"

        output_index = self.command.index("--output-wav") + 1
        Path(self.command[output_index]).write_bytes(b"RIFF-retried-wave")
        self.returncode = 0
        return "worker completed after retry", ""


class MossBundledHelpersTests(unittest.TestCase):
    def test_worker_payload_has_no_external_helper_dependency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_name = "reference.wav"
            audio_path = Path(tmp_dir) / moss_api.hash_filename(audio_name)
            audio_path.write_bytes(b"fake-wave")
            request = moss_api.MossSynthesizeRequest(
                text="测试。",
                audio_path=audio_name,
            )

            with mock.patch.object(moss_api, "PROMPTS_DIR", tmp_dir):
                payload = moss_api.MossWorkerManager().build_worker_payload(request)

        self.assertNotIn("moss_helper_script", payload)
        self.assertTrue(payload["auto_limit_max_new_tokens"])
        self.assertEqual(payload["sdpa_backend"], "math")

    def test_explicit_max_new_tokens_disables_automatic_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_name = "reference.wav"
            audio_path = Path(tmp_dir) / moss_api.hash_filename(audio_name)
            audio_path.write_bytes(b"fake-wave")
            request = moss_api.MossSynthesizeRequest(
                text="测试。",
                audio_path=audio_name,
                max_new_tokens=777,
            )

            with mock.patch.object(moss_api, "PROMPTS_DIR", tmp_dir):
                payload = moss_api.MossWorkerManager().build_worker_payload(request)

        self.assertEqual(payload["max_new_tokens"], 777)
        self.assertFalse(payload["auto_limit_max_new_tokens"])

    def test_generation_budget_bounds_missing_eos(self):
        self.assertEqual(
            generation_frame_budget(
                "测试文本",
                4096,
                auto_limit=True,
                min_new_tokens=256,
                new_tokens_per_char=10,
            ),
            256,
        )
        self.assertEqual(
            generation_frame_budget(
                "字" * 38,
                4096,
                auto_limit=True,
                min_new_tokens=256,
                new_tokens_per_char=10,
            ),
            380,
        )
        self.assertEqual(
            generation_frame_budget(
                "测试",
                4096,
                auto_limit=False,
                min_new_tokens=256,
                new_tokens_per_char=10,
            ),
            4096,
        )

    def test_worker_starts_without_checking_an_external_helper_file(self):
        manager = moss_api.MossWorkerManager()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(moss_api, "MOSS_WORKER_SCRIPT", __file__),
                mock.patch.object(moss_api, "MOSS_MODEL_DIR", tmp_dir),
                mock.patch.object(moss_api, "MOSS_WORKER_TMP_DIR", tmp_dir),
                mock.patch.object(
                    moss_api,
                    "resolve_conda_executable",
                    return_value="/fake/conda",
                ),
                mock.patch.object(moss_api.subprocess, "Popen", FakeProcess),
            ):
                audio = manager.run_worker({"text": "测试。"})

        self.assertEqual(audio, b"RIFF-moss-wave")
        self.assertIsNone(manager.last_error)

    def test_cuda_driver_error_retries_once_with_safe_backend(self):
        manager = moss_api.MossWorkerManager()
        RetryableCudaProcess.calls = 0
        RetryableCudaProcess.payloads = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(moss_api, "MOSS_WORKER_SCRIPT", __file__),
                mock.patch.object(moss_api, "MOSS_MODEL_DIR", tmp_dir),
                mock.patch.object(moss_api, "MOSS_WORKER_TMP_DIR", tmp_dir),
                mock.patch.object(moss_api, "MOSS_CUDA_RETRY_COUNT", 1),
                mock.patch.object(moss_api, "MOSS_CUDA_RETRY_MAX_NEW_TOKENS", 384),
                mock.patch.object(moss_api, "CUDA_RELEASE_DELAY", 0),
                mock.patch.object(
                    moss_api,
                    "resolve_conda_executable",
                    return_value="/fake/conda",
                ),
                mock.patch.object(moss_api.subprocess, "Popen", RetryableCudaProcess),
            ):
                audio = manager.run_worker(
                    {
                        "text": "测试。",
                        "max_new_tokens": 4096,
                        "attn_implementation": "sdpa",
                    }
                )

        self.assertEqual(audio, b"RIFF-retried-wave")
        self.assertEqual(RetryableCudaProcess.calls, 2)
        self.assertEqual(RetryableCudaProcess.payloads[1]["attn_implementation"], "eager")
        self.assertEqual(RetryableCudaProcess.payloads[1]["sdpa_backend"], "math")
        self.assertEqual(RetryableCudaProcess.payloads[1]["max_new_tokens"], 384)
        self.assertTrue(RetryableCudaProcess.payloads[1]["auto_limit_max_new_tokens"])
        self.assertIsNone(manager.last_error)


if __name__ == "__main__":
    unittest.main()
