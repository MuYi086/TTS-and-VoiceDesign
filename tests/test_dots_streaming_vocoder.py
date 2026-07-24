import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import dots_api
import dots_tts_worker


class StreamingRuntime:
    def __init__(self):
        self.stream_kwargs = None
        self.generate_called = False

    def generate_stream(self, **kwargs):
        self.stream_kwargs = kwargs
        yield np.array([[[0.1, 0.2]]], dtype=np.float32)
        yield np.array([[[0.3, 0.4, 0.5]]], dtype=np.float32)

    def generate(self, **kwargs):
        self.generate_called = True
        raise AssertionError("流式模式不应调用整段 generate()")


class NonStreamingRuntime:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return {"audio": np.array([[[0.6, 0.7]]], dtype=np.float32)}


class RecoveringStreamingRuntime:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = []

    def generate_stream(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.failures:
            raise RuntimeError(self.failures.pop(0))
        yield np.array([[[0.8, 0.9]]], dtype=np.float32)


class DotsStreamingVocoderTests(unittest.TestCase):
    def test_streaming_mode_concatenates_official_runtime_chunks(self):
        runtime = StreamingRuntime()
        kwargs = {"text": "测试流式解码"}

        waveform = dots_tts_worker.generate_chunk_audio(
            runtime,
            kwargs,
            use_streaming_vocoder=True,
            np=np,
        )

        np.testing.assert_allclose(waveform, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(runtime.stream_kwargs, kwargs)
        self.assertFalse(runtime.generate_called)

    def test_non_streaming_override_preserves_legacy_generate_path(self):
        runtime = NonStreamingRuntime()
        kwargs = {"text": "兼容旧版运行时"}

        waveform = dots_tts_worker.generate_chunk_audio(
            runtime,
            kwargs,
            use_streaming_vocoder=False,
            np=np,
        )

        np.testing.assert_allclose(waveform, [0.6, 0.7])
        self.assertEqual(runtime.generate_kwargs, kwargs)

    def test_api_enables_streaming_vocoder_in_worker_payload_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_name = "reference.wav"
            audio_path = Path(tmp_dir) / dots_api.hash_filename(audio_name)
            audio_path.write_bytes(b"fake-wave")
            request = dots_api.DotsSynthesizeRequest(
                text="测试。",
                audio_path=audio_name,
            )

            with (
                mock.patch.object(dots_api, "PROMPTS_DIR", tmp_dir),
                mock.patch.object(dots_api, "DOTS_USE_STREAMING_VOCODER", True),
            ):
                payload = dots_api.DotsWorkerManager().build_worker_payload(request)

        self.assertTrue(payload["use_streaming_vocoder"])

    def test_no_payload_error_retries_then_falls_back_to_audio_only(self):
        error = dots_tts_worker.PROMPT_CONTINUATION_NO_PAYLOAD_ERROR
        runtime = RecoveringStreamingRuntime([error, error])
        seed_everything = mock.Mock()
        torch = mock.Mock()
        torch.cuda.is_available.return_value = False

        waveform, used_audio_only_fallback = (
            dots_tts_worker.generate_chunk_audio_with_recovery(
                runtime,
                {
                    "text": "测试恢复。",
                    "prompt_audio_path": "/tmp/reference.wav",
                    "prompt_text": "参考音频转写。",
                },
                use_streaming_vocoder=True,
                np=np,
                seed_everything=seed_everything,
                seed=42,
                torch=torch,
            )
        )

        np.testing.assert_allclose(waveform, [0.8, 0.9])
        self.assertTrue(used_audio_only_fallback)
        self.assertEqual(len(runtime.calls), 3)
        self.assertEqual(runtime.calls[0]["prompt_text"], "参考音频转写。")
        self.assertEqual(runtime.calls[1]["prompt_text"], "参考音频转写。")
        self.assertIsNone(runtime.calls[2]["prompt_text"])
        self.assertEqual(seed_everything.call_args_list, [mock.call(43), mock.call(44)])

    def test_unrelated_generation_error_is_not_retried(self):
        runtime = RecoveringStreamingRuntime(["CUDA out of memory"])
        seed_everything = mock.Mock()
        torch = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            dots_tts_worker.generate_chunk_audio_with_recovery(
                runtime,
                {
                    "text": "测试错误。",
                    "prompt_audio_path": "/tmp/reference.wav",
                    "prompt_text": "参考音频转写。",
                },
                use_streaming_vocoder=True,
                np=np,
                seed_everything=seed_everything,
                seed=42,
                torch=torch,
            )

        self.assertEqual(len(runtime.calls), 1)
        seed_everything.assert_not_called()


if __name__ == "__main__":
    unittest.main()
