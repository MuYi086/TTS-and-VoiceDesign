"""Regression coverage for Task 6 model/environment routing."""

import asyncio
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import api
import voxcpm2_api


class Task6ModelRoutingTests(unittest.TestCase):
    def test_voice_design_provider_registry_exposes_all_local_models(self):
        response = asyncio.run(api.voice_design_providers())
        providers = {provider["id"]: provider for provider in response["providers"]}

        self.assertEqual(
            {"qwen", "moss", "ming", "mimo", "voxcpm2"},
            set(providers),
        )
        self.assertEqual(providers["qwen"]["route"], "/v1/qwen/design")
        self.assertEqual(providers["moss"]["route"], "/v1/moss/design")
        self.assertEqual(providers["ming"]["route"], "/v1/Ming/design")
        self.assertEqual(providers["qwen"]["environment"], "qwen3-voiceDesign")
        self.assertEqual(providers["moss"]["environment"], "moss-voiceGenerator")
        self.assertEqual(providers["ming"]["environment"], "Ming-omni-tts-0.5B")

    def test_voice_design_requests_use_the_dedicated_worker_configs(self):
        cases = (
            (
                api.QwenDesignRequest(voice_description="清晰的成年女声。"),
                api.run_qwen_voice_design,
                api.QWEN_VOICEDESIGN_WORKER,
            ),
            (
                api.MossDesignRequest(voice_description="温柔的成年女声。"),
                api.run_moss_voice_design,
                api.MOSS_VOICEGENERATOR_WORKER,
            ),
            (
                api.MingDesignRequest(voice_description="沉稳的成年女声。"),
                api.run_ming_voice_design,
                api.MING_OMNI_TTS_WORKER,
            ),
        )

        for request, runner, worker_config in cases:
            with self.subTest(environment=worker_config.conda_env), patch.object(
                api, "run_local_worker", return_value=b"wav"
            ) as run_worker:
                self.assertEqual(runner(request), b"wav")

            payload, config = run_worker.call_args.args
            self.assertIs(config, worker_config)
            self.assertEqual(payload["voice_description"], request.voice_description)
            self.assertEqual(payload["local_files_only"], api.LOCAL_FILES_ONLY)

    def test_shared_8306_route_selects_ming_only_when_requested(self):
        self.assertEqual(voxcpm2_api.resolve_synthesis_backend({}), "voxcpm2")
        self.assertEqual(voxcpm2_api.resolve_synthesis_backend({"backend": "voxcpm2"}), "voxcpm2")
        self.assertEqual(voxcpm2_api.resolve_synthesis_backend({"backend": "ming"}), "ming")
        self.assertEqual(
            voxcpm2_api.resolve_synthesis_backend({"model": "Ming-omni-tts-0.5B"}),
            "ming",
        )

    def test_ming_clone_payload_keeps_reference_audio_and_transcript(self):
        with TemporaryDirectory() as prompts_dir, patch.object(voxcpm2_api, "PROMPTS_DIR", prompts_dir):
            audio_path = "reference.wav"
            stored_path = Path(prompts_dir) / voxcpm2_api.hash_filename(audio_path)
            stored_path.write_bytes(b"reference-audio")
            request = voxcpm2_api.MingSynthesizeRequest.model_validate(
                {
                    "text": "这是目标台词。",
                    "audio_path": audio_path,
                    "prompt_text": "这是参考音频转写。",
                    "style": "自然、清晰",
                }
            )

            payload = voxcpm2_api.MingWorkerManager().build_worker_payload(request)

        self.assertEqual(payload["operation"], "clone")
        self.assertEqual(payload["ref_audio_path"], str(stored_path))
        self.assertEqual(payload["prompt_text"], "这是参考音频转写。")
        self.assertEqual(payload["ref_text"], "这是参考音频转写。")
        self.assertEqual(payload["style"], "自然、清晰")
        self.assertEqual(payload["model_path"], voxcpm2_api.MING_OMNI_TTS_MODEL_DIR)


if __name__ == "__main__":
    unittest.main()
