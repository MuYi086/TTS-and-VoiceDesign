"""Regression coverage for the shared voice-cloning synthesis contract."""

import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

API_DIR = Path(__file__).resolve().parents[1] / "api"
sys.path.insert(0, str(API_DIR))

import api
import dots_api
import longcat_api
import moss_api
import omnivoice_api
import qwen3_tts_api
import voxcpm2_api
import voxcpm2_worker


SYNTHESIS_REQUEST_MODELS = {
    "8300 IndexTTS2": api.TextToSpeechRequest,
    "8301 dots.tts": dots_api.DotsSynthesizeRequest,
    "8302 LongCat": longcat_api.LongCatSynthesizeRequest,
    "8303 MOSS": moss_api.MossSynthesizeRequest,
    "8304 OmniVoice": omnivoice_api.OmniVoiceSynthesizeRequest,
    "8305 Qwen3-TTS": qwen3_tts_api.Qwen3TtsSynthesizeRequest,
    "8306 VoxCPM2": voxcpm2_api.VoxCpm2SynthesizeRequest,
}

REFERENCE_TEXT_REQUEST_MODELS = {
    "8301 dots.tts": dots_api.DotsSynthesizeRequest,
    "8302 LongCat": longcat_api.LongCatSynthesizeRequest,
    "8303 MOSS": moss_api.MossSynthesizeRequest,
    "8304 OmniVoice": omnivoice_api.OmniVoiceSynthesizeRequest,
    "8305 Qwen3-TTS": qwen3_tts_api.Qwen3TtsSynthesizeRequest,
    "8306 VoxCPM2": voxcpm2_api.VoxCpm2SynthesizeRequest,
}

REFERENCE_TEXT_MANAGER_CASES = (
    (
        "8301 dots.tts",
        dots_api,
        dots_api.DotsSynthesizeRequest,
        dots_api.DotsWorkerManager,
        "prompt_text",
    ),
    (
        "8302 LongCat",
        longcat_api,
        longcat_api.LongCatSynthesizeRequest,
        longcat_api.LongCatWorkerManager,
        "prompt_text",
    ),
    (
        "8303 MOSS",
        moss_api,
        moss_api.MossSynthesizeRequest,
        moss_api.MossWorkerManager,
        "prompt_text",
    ),
    (
        "8304 OmniVoice",
        omnivoice_api,
        omnivoice_api.OmniVoiceSynthesizeRequest,
        omnivoice_api.OmniVoiceWorkerManager,
        "ref_text",
    ),
    (
        "8305 Qwen3-TTS",
        qwen3_tts_api,
        qwen3_tts_api.Qwen3TtsSynthesizeRequest,
        qwen3_tts_api.Qwen3TtsWorkerManager,
        "ref_text",
    ),
    (
        "8306 VoxCPM2",
        voxcpm2_api,
        voxcpm2_api.VoxCpm2SynthesizeRequest,
        voxcpm2_api.VoxCpm2WorkerManager,
        "prompt_text",
    ),
)


class SynthesizeStylePromptContractTests(unittest.TestCase):
    def test_all_synthesis_request_models_reject_style_prompt(self):
        for style_prompt in ("年轻男性，声线偏细，略带沙哑。", None):
            for name, request_model in SYNTHESIS_REQUEST_MODELS.items():
                with self.subTest(name=name, style_prompt=style_prompt):
                    with self.assertRaisesRegex(ValidationError, "style_prompt"):
                        request_model.model_validate(
                            {
                                "text": "公主殿下，火山口真的有烈火仙莲吗？",
                                "audio_path": "missing-reference.wav",
                                "style_prompt": style_prompt,
                                # Existing WebUI compatibility data must remain harmless.
                                "emo_vector": [0, 0, 0, 0.5, 0, 0, 0, 0],
                            }
                        )

    def test_compatibility_fields_do_not_make_the_contract_strict(self):
        for name, request_model in SYNTHESIS_REQUEST_MODELS.items():
            with self.subTest(name=name):
                request = request_model.model_validate(
                    {
                        "text": "测试。",
                        "audio_path": "missing-reference.wav",
                        "emo_vector": [0, 0, 0, 0.5, 0, 0, 0, 0],
                    }
                )
                self.assertEqual(request.text, "测试。")

    def test_reference_text_models_declare_reference_transcript_explicitly(self):
        for name, request_model in REFERENCE_TEXT_REQUEST_MODELS.items():
            with self.subTest(name=name):
                request = request_model.model_validate(
                    {
                        "text": "这是待合成的台词。",
                        "audio_path": "reference.wav",
                        "prompt_text": "这是参考音频的准确转写。",
                    }
                )
                self.assertIn("prompt_text", request_model.model_fields)
                self.assertEqual(request.prompt_text, "这是参考音频的准确转写。")

    def test_indextts2_does_not_declare_an_unsupported_reference_transcript(self):
        request = api.TextToSpeechRequest.model_validate(
            {
                "text": "这是待合成的台词。",
                "audio_path": "reference.wav",
                "prompt_text": "IndexTTS2 官方克隆签名不使用该字段。",
            }
        )

        self.assertNotIn("prompt_text", api.TextToSpeechRequest.model_fields)
        self.assertFalse(hasattr(request, "prompt_text"))

    def test_reference_transcript_reaches_each_supported_worker_payload(self):
        for case in REFERENCE_TEXT_MANAGER_CASES:
            name, module, request_model, manager_type, payload_key = case
            with self.subTest(name=name), TemporaryDirectory() as prompts_dir, patch.object(
                module, "PROMPTS_DIR", prompts_dir
            ):
                audio_path = "reference.wav"
                stored_path = Path(prompts_dir) / module.hash_filename(audio_path)
                stored_path.write_bytes(b"reference-audio")
                request = request_model.model_validate(
                    {
                        "text": "这是待合成的台词。",
                        "audio_path": audio_path,
                        "prompt_text": "这是参考音频的准确转写。",
                    }
                )

                payload = manager_type().build_worker_payload(request)

                self.assertEqual(payload[payload_key], "这是参考音频的准确转写。")

    def test_voxcpm_worker_never_prepends_a_style_prompt(self):
        helper_args = voxcpm2_worker.build_helper_args(
            {"style_prompt": "这段文本绝不能混入待朗读文案。"}
        )
        self.assertEqual(helper_args.style_prompt, "")


if __name__ == "__main__":
    unittest.main()
