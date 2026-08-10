"""Regression coverage for the LongCat-AudioDiT-3.5B adapter."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import longcat_audiodit_api
import longcat_audiodit_worker


class LongCatAudioDitTests(unittest.TestCase):
    def test_clone_payload_uses_prompt_audio_and_accurate_prompt_text(self):
        with TemporaryDirectory() as prompts_dir, patch.object(
            longcat_audiodit_api, "PROMPTS_DIR", prompts_dir
        ):
            audio_path = "roles/narrator.wav"
            stored_path = Path(prompts_dir) / longcat_audiodit_api.hash_filename(audio_path)
            stored_path.write_bytes(b"reference-audio")
            request = longcat_audiodit_api.LongCatAudioDitSynthesizeRequest.model_validate(
                {
                    "text": "今天晴暖转阴雨。",
                    "audio_path": audio_path,
                    "prompt_text": "这是一句参考音频转写。",
                }
            )

            payload = longcat_audiodit_api.LongCatAudioDitWorkerManager().build_worker_payload(
                request
            )

        self.assertEqual(payload["operation"], "clone")
        self.assertEqual(payload["ref_audio_path"], str(stored_path))
        self.assertEqual(payload["prompt_text"], "这是一句参考音频转写。")
        self.assertEqual(payload["nfe"], 16)
        self.assertEqual(payload["guidance_method"], "apg")
        self.assertEqual(payload["guidance_strength"], 4.0)
        self.assertEqual(payload["vae_dtype"], "float16")

    def test_clone_payload_can_recover_prompt_text_from_upload_sidecar(self):
        with TemporaryDirectory() as prompts_dir, patch.object(
            longcat_audiodit_api, "PROMPTS_DIR", prompts_dir
        ):
            audio_path = "reference.wav"
            stored_path = Path(prompts_dir) / longcat_audiodit_api.hash_filename(audio_path)
            stored_path.write_bytes(b"reference-audio")
            longcat_audiodit_api.save_prompt_text_sidecar(audio_path, "侧车中的准确转写")
            request = longcat_audiodit_api.LongCatAudioDitSynthesizeRequest(
                text="目标台词。", audio_path=audio_path
            )

            payload = longcat_audiodit_api.LongCatAudioDitWorkerManager().build_worker_payload(
                request
            )

        self.assertEqual(payload["prompt_text"], "侧车中的准确转写")

    def test_clone_requires_prompt_text_when_no_sidecar_exists(self):
        with TemporaryDirectory() as prompts_dir, patch.object(
            longcat_audiodit_api, "PROMPTS_DIR", prompts_dir
        ):
            audio_path = "reference.wav"
            (Path(prompts_dir) / longcat_audiodit_api.hash_filename(audio_path)).write_bytes(
                b"reference-audio"
            )
            request = longcat_audiodit_api.LongCatAudioDitSynthesizeRequest(
                text="目标台词。", audio_path=audio_path
            )

            with self.assertRaises(HTTPException) as raised:
                longcat_audiodit_api.LongCatAudioDitWorkerManager().build_worker_payload(request)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("prompt_text", str(raised.exception.detail))

    def test_worker_text_chunking_matches_official_punctuation_limits(self):
        chunks = longcat_audiodit_worker.split_text("第一句。第二句，内容很长。", 5)

        self.assertEqual(chunks, ["第一句。", "第二句，", "内容很长。"])

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

        fake_torch = SimpleNamespace(cuda=FakeCuda())
        with patch.object(longcat_audiodit_worker.gc, "collect") as collect:
            longcat_audiodit_worker.release_cuda_memory(fake_torch)

        collect.assert_called_once_with()
        self.assertEqual(calls, ["synchronize", "empty_cache", "ipc_collect"])


if __name__ == "__main__":
    unittest.main()
