import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import omnivoice_api
from omnivoice_tts_worker import split_text


class RetryableCudaProcess:
    calls = 0
    payloads = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = 33001 + type(self).calls
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
        Path(self.command[output_index]).write_bytes(b"RIFF-omnivoice-retried")
        self.returncode = 0
        return "worker completed after retry", ""


class OmniVoiceCudaRecoveryTests(unittest.TestCase):
    def test_default_payload_uses_stable_attention_and_short_chunks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_name = "reference.wav"
            audio_path = Path(tmp_dir) / omnivoice_api.hash_filename(audio_name)
            audio_path.write_bytes(b"fake-wave")
            request = omnivoice_api.OmniVoiceSynthesizeRequest(
                text="测试。",
                audio_path=audio_name,
            )

            with (
                mock.patch.object(omnivoice_api, "PROMPTS_DIR", tmp_dir),
                mock.patch.object(omnivoice_api, "OMNIVOICE_ASR_MODEL_DIR", tmp_dir),
                mock.patch.object(omnivoice_api, "OMNIVOICE_MAX_CHARS_PER_CHUNK", 60),
                mock.patch.object(omnivoice_api, "OMNIVOICE_ATTN_IMPLEMENTATION", "sdpa"),
                mock.patch.object(omnivoice_api, "OMNIVOICE_SDPA_BACKEND", "math"),
            ):
                payload = omnivoice_api.OmniVoiceWorkerManager().build_worker_payload(request)

        self.assertEqual(payload["max_chars_per_chunk"], 60)
        self.assertEqual(payload["attn_implementation"], "sdpa")
        self.assertEqual(payload["sdpa_backend"], "math")
        self.assertEqual(payload["asr_model_path"], tmp_dir)

    def test_missing_asr_model_fails_before_starting_worker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_name = "reference.wav"
            audio_path = Path(tmp_dir) / omnivoice_api.hash_filename(audio_name)
            audio_path.write_bytes(b"fake-wave")
            missing_asr_dir = str(Path(tmp_dir) / "missing-whisper")
            request = omnivoice_api.OmniVoiceSynthesizeRequest(
                text="测试。",
                audio_path=audio_name,
            )

            with (
                mock.patch.object(omnivoice_api, "PROMPTS_DIR", tmp_dir),
                mock.patch.object(
                    omnivoice_api,
                    "OMNIVOICE_ASR_MODEL_DIR",
                    missing_asr_dir,
                ),
            ):
                with self.assertRaises(omnivoice_api.HTTPException) as context:
                    omnivoice_api.OmniVoiceWorkerManager().build_worker_payload(request)

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("OMNIVOICE_ASR_MODEL_DIR", context.exception.detail)

    def test_prompt_text_does_not_require_asr_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_name = "reference.wav"
            audio_path = Path(tmp_dir) / omnivoice_api.hash_filename(audio_name)
            audio_path.write_bytes(b"fake-wave")
            request = omnivoice_api.OmniVoiceSynthesizeRequest(
                text="测试。",
                audio_path=audio_name,
                prompt_text="参考文本。",
            )

            with (
                mock.patch.object(omnivoice_api, "PROMPTS_DIR", tmp_dir),
                mock.patch.object(
                    omnivoice_api,
                    "OMNIVOICE_ASR_MODEL_DIR",
                    str(Path(tmp_dir) / "missing-whisper"),
                ),
            ):
                payload = omnivoice_api.OmniVoiceWorkerManager().build_worker_payload(request)

        self.assertEqual(payload["ref_text"], "参考文本。")

    def test_long_text_is_split_before_generation(self):
        chunks = split_text("字" * 102, 60)

        self.assertEqual([len(chunk) for chunk in chunks], [60, 42])

    def test_cuda_driver_error_retries_with_eager_and_smaller_chunks(self):
        manager = omnivoice_api.OmniVoiceWorkerManager()
        RetryableCudaProcess.calls = 0
        RetryableCudaProcess.payloads = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(omnivoice_api, "OMNIVOICE_WORKER_SCRIPT", __file__),
                mock.patch.object(omnivoice_api, "OMNIVOICE_MODEL_DIR", tmp_dir),
                mock.patch.object(omnivoice_api, "OMNIVOICE_WORKER_TMP_DIR", tmp_dir),
                mock.patch.object(omnivoice_api, "OMNIVOICE_CUDA_RETRY_COUNT", 1),
                mock.patch.object(omnivoice_api, "OMNIVOICE_CUDA_RETRY_MAX_CHARS", 48),
                mock.patch.object(omnivoice_api, "CUDA_RELEASE_DELAY", 0),
                mock.patch.object(
                    omnivoice_api,
                    "resolve_conda_executable",
                    return_value="/fake/conda",
                ),
                mock.patch.object(
                    omnivoice_api.subprocess,
                    "Popen",
                    RetryableCudaProcess,
                ),
            ):
                audio = manager.run_worker(
                    {
                        "text": "测试。",
                        "attn_implementation": "sdpa",
                        "sdpa_backend": "auto",
                        "max_chars_per_chunk": 120,
                    }
                )

        self.assertEqual(audio, b"RIFF-omnivoice-retried")
        self.assertEqual(RetryableCudaProcess.calls, 2)
        self.assertEqual(
            RetryableCudaProcess.payloads[1]["attn_implementation"],
            "eager",
        )
        self.assertEqual(RetryableCudaProcess.payloads[1]["sdpa_backend"], "math")
        self.assertEqual(RetryableCudaProcess.payloads[1]["max_chars_per_chunk"], 48)
        self.assertIsNone(manager.last_error)


if __name__ == "__main__":
    unittest.main()
