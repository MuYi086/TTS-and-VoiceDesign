"""Regression coverage for top-level clone debugging defaults."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import dots_tts_soar_api
import longcat_audiodit_api
import qwen3_tts_api
import voxcpm2_api


class TopLevelCloneDefaultsTests(unittest.TestCase):
    def test_longcat_top_level_defaults_reach_worker_payload(self):
        with TemporaryDirectory() as prompts_dir, patch.object(
            longcat_audiodit_api, "PROMPTS_DIR", prompts_dir
        ), patch.object(
            longcat_audiodit_api, "LONGCAT_AUDIODIT_NFE", 7
        ), patch.object(
            longcat_audiodit_api, "LONGCAT_AUDIODIT_GUIDANCE_STRENGTH", 2.25
        ), patch.object(
            longcat_audiodit_api, "LONGCAT_AUDIODIT_DURATION_SCALE", 0.85
        ):
            audio_path = "reference.wav"
            stored_path = Path(prompts_dir) / longcat_audiodit_api.hash_filename(audio_path)
            stored_path.write_bytes(b"reference-audio")
            request = longcat_audiodit_api.LongCatAudioDitSynthesizeRequest(
                text="目标台词。",
                audio_path=audio_path,
                prompt_text="参考音频转写。",
            )

            payload = longcat_audiodit_api.LongCatAudioDitWorkerManager().build_worker_payload(
                request
            )

        self.assertEqual(payload["nfe"], 7)
        self.assertEqual(payload["guidance_strength"], 2.25)
        self.assertEqual(payload["duration_scale"], 0.85)

    def test_qwen_top_level_defaults_reach_worker_payload(self):
        with TemporaryDirectory() as prompts_dir, patch.object(
            qwen3_tts_api, "PROMPTS_DIR", prompts_dir
        ), patch.object(
            qwen3_tts_api, "QWEN3_TTS_MAX_NEW_TOKENS", 777
        ), patch.object(
            qwen3_tts_api, "QWEN3_TTS_TEMPERATURE", 0.65
        ), patch.object(
            qwen3_tts_api, "QWEN3_TTS_X_VECTOR_ONLY", True
        ):
            audio_path = "reference.wav"
            stored_path = Path(prompts_dir) / qwen3_tts_api.hash_filename(audio_path)
            stored_path.write_bytes(b"reference-audio")
            request = qwen3_tts_api.Qwen3TtsSynthesizeRequest(
                text="目标台词。",
                audio_path=audio_path,
                prompt_text="参考音频转写。",
            )

            payload = qwen3_tts_api.Qwen3TtsWorkerManager().build_worker_payload(request)

        self.assertEqual(payload["max_new_tokens"], 777)
        self.assertEqual(payload["temperature"], 0.65)
        self.assertTrue(payload["x_vector_only"])

    def test_dots_top_level_defaults_reach_worker_payload(self):
        with TemporaryDirectory() as prompts_dir, patch.object(
            dots_tts_soar_api, "PROMPTS_DIR", prompts_dir
        ), patch.object(
            dots_tts_soar_api, "DOTS_TTS_SOAR_NUM_STEPS", 13
        ), patch.object(
            dots_tts_soar_api, "DOTS_TTS_SOAR_GUIDANCE_SCALE", 1.75
        ), patch.object(
            dots_tts_soar_api, "DOTS_TTS_SOAR_SPEAKER_SCALE", 1.1
        ):
            audio_path = "reference.wav"
            stored_path = Path(prompts_dir) / dots_tts_soar_api.hash_filename(audio_path)
            stored_path.write_bytes(b"reference-audio")
            request = dots_tts_soar_api.DotsTtsSoarSynthesizeRequest(
                text="目标台词。",
                audio_path=audio_path,
                prompt_text="参考音频转写。",
            )

            payload = dots_tts_soar_api.DotsTtsSoarWorkerManager().build_worker_payload(request)

        self.assertEqual(payload["num_steps"], 13)
        self.assertEqual(payload["guidance_scale"], 1.75)
        self.assertEqual(payload["speaker_scale"], 1.1)

    def test_ming_top_level_defaults_reach_shared_worker_payload(self):
        with TemporaryDirectory() as prompts_dir, patch.object(
            voxcpm2_api, "PROMPTS_DIR", prompts_dir
        ), patch.object(
            voxcpm2_api, "MING_OMNI_TTS_MAX_DECODE_STEPS", 321
        ), patch.object(
            voxcpm2_api, "MING_OMNI_TTS_CFG", 1.4
        ), patch.object(
            voxcpm2_api, "MING_OMNI_TTS_SIGMA", 0.15
        ), patch.object(
            voxcpm2_api, "MING_OMNI_TTS_TEMPERATURE", 0.2
        ):
            audio_path = "reference.wav"
            stored_path = Path(prompts_dir) / voxcpm2_api.hash_filename(audio_path)
            stored_path.write_bytes(b"reference-audio")
            request = voxcpm2_api.MingSynthesizeRequest(
                text="目标台词。",
                audio_path=audio_path,
                prompt_text="参考音频转写。",
            )

            payload = voxcpm2_api.MingWorkerManager().build_worker_payload(request)

        self.assertEqual(payload["max_decode_steps"], 321)
        self.assertEqual(payload["cfg"], 1.4)
        self.assertEqual(payload["sigma"], 0.15)
        self.assertEqual(payload["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
