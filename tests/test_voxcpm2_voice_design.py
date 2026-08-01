"""Regression coverage for the main API's dedicated VoxCPM2 VoiceDesign dispatch."""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import api
import voxcpm2_voice_design


class VoxCpm2VoiceDesignDispatchTests(unittest.TestCase):
    def test_voxcpm2_route_dispatches_to_dedicated_worker(self):
        request = api.VoxCpm2VoiceDesignRequest(
            voice_description="年轻女性，声音温柔甜美。",
            text="你好。",
        )
        with mock.patch.object(api, "run_voxcpm2_voice_design", return_value=b"RIFF-wav") as design:
            response = asyncio.run(api.voxcpm2_design(request))

        self.assertEqual(response.media_type, "audio/wav")
        self.assertEqual(response.body, b"RIFF-wav")
        design.assert_called_once_with(request)

    def test_provider_advertises_the_voxcpm2_route(self):
        providers = asyncio.run(api.voice_design_providers())["providers"]
        provider = next(item for item in providers if item["id"] == "voxcpm2")
        self.assertEqual(provider["route"], "/v1/voxcpm2/design")
        self.assertIn(
            "/v1/voxcpm2/design",
            {route.path for route in api.app.routes},
        )

    def test_voice_design_payload_uses_official_format_without_reference_audio(self):
        request = voxcpm2_voice_design.VoxCpm2VoiceDesignRequest(
            voice_description="年轻女性，声音温柔甜美。",
            text="你好。",
        )

        payload = voxcpm2_voice_design.build_voice_design_worker_payload(request)

        self.assertEqual(payload["operation"], "voice_design")
        self.assertEqual(payload["voice_description"], "年轻女性，声音温柔甜美。")
        self.assertEqual(payload["text"], "你好。")
        self.assertNotIn("ref_audio_path", payload)
        self.assertEqual(payload["cfg_value"], 2.0)
        self.assertEqual(payload["inference_timesteps"], 10)
        self.assertEqual(payload["seed"], 20260614)

    def test_voice_design_uses_its_dedicated_worker_script(self):
        request = voxcpm2_voice_design.VoxCpm2VoiceDesignRequest(
            voice_description="沉稳的成年男性。",
            text="你好。",
        )
        with mock.patch.object(
            voxcpm2_voice_design.manager,
            "run_worker",
            return_value=b"RIFF-wav",
        ) as run_worker:
            audio = voxcpm2_voice_design.run_voxcpm2_voice_design(request)

        self.assertEqual(audio, b"RIFF-wav")
        self.assertEqual(
            run_worker.call_args.kwargs["worker_script"],
            voxcpm2_voice_design.VOXCPM2_VOICE_DESIGN_WORKER_SCRIPT,
        )

    def test_qwen_request_has_no_voxcpm_provider_switch(self):
        self.assertNotIn("provider", api.QwenDesignRequest.model_fields)


if __name__ == "__main__":
    unittest.main()
