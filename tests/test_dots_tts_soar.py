"""Regression coverage for the dots.tts-soar adapter without loading weights."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import dots_tts_soar_api
import dots_tts_soar_worker


class DotsTtsSoarTests(unittest.TestCase):
    def test_clone_payload_uses_reference_audio_and_prompt_text(self):
        with TemporaryDirectory() as prompts_dir, patch.object(
            dots_tts_soar_api, "PROMPTS_DIR", prompts_dir
        ):
            audio_path = "roles/narrator.wav"
            stored_path = Path(prompts_dir) / dots_tts_soar_api.hash_filename(audio_path)
            stored_path.write_bytes(b"reference-audio")
            request = dots_tts_soar_api.DotsTtsSoarSynthesizeRequest(
                text="今天晴暖转阴雨。",
                audio_path=audio_path,
                prompt_text="这是一句参考音频转写。",
            )

            payload = dots_tts_soar_api.DotsTtsSoarWorkerManager().build_worker_payload(request)

        self.assertEqual(payload["operation"], "clone")
        self.assertEqual(payload["ref_audio_path"], str(stored_path))
        self.assertEqual(payload["prompt_text"], "这是一句参考音频转写。")
        self.assertEqual(payload["num_steps"], 10)
        self.assertEqual(payload["guidance_scale"], 1.2)
        self.assertEqual(payload["speaker_scale"], 1.5)

    def test_clone_payload_supports_x_vector_only_and_sidecar_fallback(self):
        with TemporaryDirectory() as prompts_dir, patch.object(
            dots_tts_soar_api, "PROMPTS_DIR", prompts_dir
        ):
            audio_path = "reference.wav"
            (Path(prompts_dir) / dots_tts_soar_api.hash_filename(audio_path)).write_bytes(
                b"reference-audio"
            )
            request = dots_tts_soar_api.DotsTtsSoarSynthesizeRequest(
                text="目标台词。", audio_path=audio_path
            )
            payload_without_text = dots_tts_soar_api.DotsTtsSoarWorkerManager().build_worker_payload(request)
            dots_tts_soar_api.save_prompt_text_sidecar(audio_path, "sidecar 中的准确转写")
            payload_with_sidecar = dots_tts_soar_api.DotsTtsSoarWorkerManager().build_worker_payload(request)

        self.assertIsNone(payload_without_text["prompt_text"])
        self.assertEqual(payload_with_sidecar["prompt_text"], "sidecar 中的准确转写")

    def test_manager_persists_successful_worker_audio(self):
        payload = b"RIFF-dots-wave"
        with TemporaryDirectory() as output_dir, patch.object(
            dots_tts_soar_api, "DOTS_TTS_SOAR_OUTPUT_DIR", output_dir
        ), patch.object(dots_tts_soar_api, "run_local_worker", return_value=payload):
            audio = dots_tts_soar_api.DotsTtsSoarWorkerManager().run_worker({})
            saved_audio = list(Path(output_dir).glob("dots_tts_soar_*.wav"))
            saved_bytes = saved_audio[0].read_bytes() if saved_audio else None

        self.assertEqual(audio, payload)
        self.assertEqual(len(saved_audio), 1)
        self.assertEqual(saved_bytes, payload)

    def test_worker_explicitly_releases_cuda_allocator(self):
        calls = []

        class FakeCuda:
            def is_available(self):
                return True

            def synchronize(self):
                calls.append("synchronize")

            def empty_cache(self):
                calls.append("empty_cache")

            def ipc_collect(self):
                calls.append("ipc_collect")

        class FakeTorch:
            cuda = FakeCuda()

        dots_tts_soar_worker.clear_cuda_cache(FakeTorch())
        self.assertEqual(calls, ["synchronize", "empty_cache", "ipc_collect"])


if __name__ == "__main__":
    unittest.main()
