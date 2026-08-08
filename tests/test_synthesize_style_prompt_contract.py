"""Regression coverage for the shared voice-cloning synthesis contract."""

import asyncio
import contextlib
import hashlib
import io
import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

API_DIR = Path(__file__).resolve().parents[1] / "api"
sys.path.insert(0, str(API_DIR))

import api
import qwen3_tts_api
import voxcpm2_api
import voxcpm2_worker


SYNTHESIS_REQUEST_MODELS = {
    "8300 IndexTTS2": api.TextToSpeechRequest,
    "8305 Qwen3-TTS": qwen3_tts_api.Qwen3TtsSynthesizeRequest,
    "8306 VoxCPM2": voxcpm2_api.VoxCpm2SynthesizeRequest,
}

REFERENCE_TEXT_REQUEST_MODELS = {
    "8305 Qwen3-TTS": qwen3_tts_api.Qwen3TtsSynthesizeRequest,
    "8306 VoxCPM2": voxcpm2_api.VoxCpm2SynthesizeRequest,
}

REFERENCE_TEXT_MANAGER_CASES = (
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
            {
                "cfg_value": voxcpm2_api.VOXCPM2_CFG_VALUE,
                "control_instruction": "克制紧张，略慢，关键处停顿，吐字清晰",
            }
        )
        self.assertEqual(helper_args.control_instruction, "克制紧张，略慢，关键处停顿，吐字清晰")

    def test_voxcpm2_controllable_clone_requires_exclusive_control_instruction(self):
        request = voxcpm2_api.VoxCpm2SynthesizeRequest.model_validate(
            {
                "text": "门后有人。",
                "audio_path": "reference.wav",
                "clone_mode": "controllable",
                "control_instruction": "克制紧张，略慢，关键处停顿，吐字清晰",
            }
        )
        self.assertEqual(request.clone_mode, "controllable")
        self.assertIsNone(request.prompt_text)

        invalid_cases = (
            {"clone_mode": "controllable"},
            {
                "clone_mode": "controllable",
                "control_instruction": "克制紧张",
                "prompt_text": "参考音频准确转写",
            },
            {
                "clone_mode": "ultimate",
                "control_instruction": "克制紧张",
            },
            {"control_instruction": "克制紧张"},
        )
        for override in invalid_cases:
            with self.subTest(override=override), self.assertRaisesRegex(ValidationError, "VoxCPM2|control_instruction"):
                voxcpm2_api.VoxCpm2SynthesizeRequest.model_validate(
                    {"text": "门后有人。", "audio_path": "reference.wav", **override}
                )

    def test_voxcpm2_controllable_payload_omits_prompt_text_and_sidecar(self):
        with TemporaryDirectory() as prompts_dir, patch.object(voxcpm2_api, "PROMPTS_DIR", prompts_dir):
            audio_path = "reference.wav"
            (Path(prompts_dir) / voxcpm2_api.hash_filename(audio_path)).write_bytes(b"reference-audio")
            voxcpm2_api.save_prompt_text_sidecar(audio_path, "不应进入可控克隆的转写")
            request = voxcpm2_api.VoxCpm2SynthesizeRequest.model_validate(
                {
                    "text": "门后有人。",
                    "audio_path": audio_path,
                    "clone_mode": "controllable",
                    "control_instruction": "克制紧张，略慢，关键处停顿，吐字清晰",
                }
            )

            payload = voxcpm2_api.VoxCpm2WorkerManager().build_worker_payload(request)

        self.assertEqual(payload["clone_mode"], "controllable")
        self.assertIsNone(payload["prompt_text"])
        self.assertEqual(payload["control_instruction"], "克制紧张，略慢，关键处停顿，吐字清晰")

    def test_voxcpm2_clone_payloads_use_global_cfg_value(self):
        with TemporaryDirectory() as prompts_dir, patch.object(voxcpm2_api, "PROMPTS_DIR", prompts_dir):
            audio_path = "reference.wav"
            (Path(prompts_dir) / voxcpm2_api.hash_filename(audio_path)).write_bytes(b"reference-audio")
            manager = voxcpm2_api.VoxCpm2WorkerManager()
            with patch.object(voxcpm2_api, "VOXCPM2_CFG_VALUE", 1.75):
                controllable_request = voxcpm2_api.VoxCpm2SynthesizeRequest.model_validate(
                    {
                        "text": "门后有人。",
                        "audio_path": audio_path,
                        "clone_mode": "controllable",
                        "control_instruction": "克制紧张",
                    }
                )
                ultimate_request = voxcpm2_api.VoxCpm2SynthesizeRequest.model_validate(
                    {
                        "text": "门后有人。",
                        "audio_path": audio_path,
                        "clone_mode": "ultimate",
                        "prompt_text": "门后有人。",
                    }
                )

                controllable_payload = manager.build_worker_payload(controllable_request)
                ultimate_payload = manager.build_worker_payload(ultimate_request)

        self.assertEqual(controllable_payload["cfg_value"], 1.75)
        self.assertEqual(ultimate_payload["cfg_value"], 1.75)
        self.assertEqual(controllable_payload["seed"], -1)
        self.assertEqual(ultimate_payload["seed"], -1)

    def test_voxcpm2_project_defaults_match_official_demo_behavior(self):
        self.assertEqual(voxcpm2_api.VOXCPM2_CFG_DEFAULT, 2.0)
        self.assertEqual(voxcpm2_api.VOXCPM2_SEED_DEFAULT, -1)

    def test_voxcpm2_audio_check_reports_content_hash(self):
        with TemporaryDirectory() as prompts_dir, patch.object(voxcpm2_api, "PROMPTS_DIR", prompts_dir):
            audio_path = "same-name.wav"
            content = b"current-reference-audio"
            (Path(prompts_dir) / voxcpm2_api.hash_filename(audio_path)).write_bytes(content)

            response = asyncio.run(voxcpm2_api.check_audio_exists(audio_path))

        self.assertTrue(response["exists"])
        self.assertEqual(response["size_bytes"], len(content))
        self.assertEqual(response["sha256"], hashlib.sha256(content).hexdigest())

    def test_voxcpm2_nonverbal_tag_is_validated_and_reaches_worker_payload(self):
        with TemporaryDirectory() as prompts_dir, patch.object(voxcpm2_api, "PROMPTS_DIR", prompts_dir):
            audio_path = "reference.wav"
            (Path(prompts_dir) / voxcpm2_api.hash_filename(audio_path)).write_bytes(b"reference-audio")
            request = voxcpm2_api.VoxCpm2SynthesizeRequest.model_validate(
                {
                    "text": "唉，还是晚了一步。",
                    "audio_path": audio_path,
                    "clone_mode": "controllable",
                    "control_instruction": "自然、清晰地表达，保留必要的非语言反应，吐字清晰",
                    "nonverbal_tags": ["sigh"],
                }
            )
            payload = voxcpm2_api.VoxCpm2WorkerManager().build_worker_payload(request)

        self.assertEqual(payload["nonverbal_tags"], ["sigh"])
        self.assertIsNone(payload["prompt_text"])

        invalid_cases = (
            {"nonverbal_tags": ["unknown"]},
            {"nonverbal_tags": ["sigh", "laughing"]},
            {"clone_mode": "ultimate", "control_instruction": None, "nonverbal_tags": ["sigh"]},
            {"text": "[sigh]测试。"},
            {"prompt_text": "[sigh]参考转写"},
        )
        for override in invalid_cases:
            with self.subTest(override=override), self.assertRaisesRegex(ValidationError, "nonverbal_tags|非语言标签"):
                voxcpm2_api.VoxCpm2SynthesizeRequest.model_validate(
                    {
                        "text": "测试。",
                        "audio_path": "reference.wav",
                        "clone_mode": "controllable",
                        "control_instruction": "自然表达",
                        **override,
                    }
                )

    def test_voxcpm2_denoise_automatically_loads_denoiser_and_preserves_generate_parameters(self):
        with TemporaryDirectory() as prompts_dir, patch.object(voxcpm2_api, "PROMPTS_DIR", prompts_dir):
            audio_path = "reference.wav"
            (Path(prompts_dir) / voxcpm2_api.hash_filename(audio_path)).write_bytes(b"reference-audio")
            request = voxcpm2_api.VoxCpm2SynthesizeRequest.model_validate(
                {
                    "text": "测试。",
                    "audio_path": audio_path,
                    "normalize": True,
                    "denoise": True,
                    "retry_badcase": False,
                    "load_denoiser": False,
                }
            )
            payload = voxcpm2_api.VoxCpm2WorkerManager().build_worker_payload(request)

        self.assertTrue(payload["normalize"])
        self.assertTrue(payload["denoise"])
        self.assertFalse(payload["retry_badcase"])
        self.assertTrue(payload["load_denoiser"])

    def test_voxcpm_worker_logs_final_model_text_with_control_and_tag(self):
        class FakeModel:
            tts_model = type("TtsModel", (), {"sample_rate": 24000})()

            def generate(self, **kwargs):
                return [0.0]

        class FakeVoxCPM:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return FakeModel()

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def synchronize():
                return None

            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def ipc_collect():
                return None

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def inference_mode():
                return contextlib.nullcontext()

        class FakeSoundFile:
            @staticmethod
            def write(path, waveform, sample_rate):
                Path(path).write_bytes(b"RIFF")

        with TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            ref_audio = temp_path / "reference.wav"
            output_wav = temp_path / "output.wav"
            ref_audio.write_bytes(b"reference-audio")
            request = {
                "text": "门后有人。",
                "ref_audio_path": str(ref_audio),
                "model_path": tmp_dir,
                "voxcpm2_helper_script": "unused-by-mock.py",
                "device": "cuda",
                "clone_mode": "controllable",
                "control_instruction": "克制紧张",
                "nonverbal_tags": ["sigh"],
                "cfg_value": 2.0,
                "inference_timesteps": 10,
                "normalize": False,
                "denoise": False,
                "retry_badcase": True,
                "load_denoiser": False,
                "optimize": False,
                "local_files_only": True,
            }
            output = io.StringIO()
            with (
                patch.object(voxcpm2_worker, "load_voxcpm2_helpers", return_value=__import__("voxcpm2_helpers")),
                patch("voxcpm2_helpers.import_runtime", return_value=(FakeVoxCPM, object(), FakeSoundFile, FakeTorch)),
                patch("voxcpm2_helpers.set_seed"),
                patch("voxcpm2_helpers.join_waveforms", return_value=[0.0]),
                patch.object(voxcpm2_worker, "trim_leading_silence", return_value=([0.0], 0)),
                contextlib.redirect_stdout(output),
            ):
                voxcpm2_worker.synthesize(request, output_wav)

        self.assertIn(
            "最终模型文本 chunk 1/1 clone_mode=controllable: (克制紧张)[sigh]门后有人。",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
