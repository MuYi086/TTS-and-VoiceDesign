"""Regression coverage for the shared voice-cloning synthesis contract."""

import unittest
from pathlib import Path
import sys

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

    def test_voxcpm_worker_never_prepends_a_style_prompt(self):
        helper_args = voxcpm2_worker.build_helper_args(
            {"style_prompt": "这段文本绝不能混入待朗读文案。"}
        )
        self.assertEqual(helper_args.style_prompt, "")


if __name__ == "__main__":
    unittest.main()
