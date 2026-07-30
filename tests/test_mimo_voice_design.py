"""Regression coverage for the MiMo VoiceDesign cloud API adapter."""

import asyncio
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import api


class MiMoVoiceDesignTests(unittest.TestCase):
    def test_build_messages_uses_user_for_design_and_assistant_for_spoken_text(self):
        messages = api.mimo_build_messages("沉稳的中年女声", "那些文字需要传承。")

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "沉稳的中年女声"},
                {"role": "assistant", "content": "那些文字需要传承。"},
            ],
        )

    def test_transport_failure_is_retryable(self):
        error = api.MiMoTransportError("MiMo 网络请求失败: timed out")
        with mock.patch.object(api, "mimo_post_json", side_effect=[error, {"ok": True}]), mock.patch.object(
            api.time, "sleep"
        ) as sleep:
            response = api.mimo_post_json_with_retry(
                url="https://api.xiaomimimo.com/v1/chat/completions",
                payload={},
                headers={},
                timeout=10,
                min_request_interval_seconds=0,
                max_retries=1,
                retry_base_seconds=0.5,
                retry_max_seconds=1,
                chunk_label="MiMo chunk 1/1",
            )

        self.assertEqual(response, {"ok": True})
        sleep.assert_called_once_with(0.5)

    def test_url_error_is_reported_as_transport_failure(self):
        with mock.patch.object(
            api.urllib.request,
            "urlopen",
            side_effect=urllib.error.URLError("timed out"),
        ):
            with self.assertRaisesRegex(api.MiMoTransportError, "网络请求失败: timed out"):
                api.mimo_post_json("https://api.xiaomimimo.com/v1/chat/completions", {}, {}, 10)

    def test_endpoint_returns_503_for_an_unreachable_upstream(self):
        request = api.MimoDesignRequest(
            voice_description="中年女性，醇厚沉稳。",
            text="需要谨慎地传承。",
        )
        with mock.patch.object(
            api,
            "run_mimo_voice_design",
            side_effect=api.MiMoTransportError("MiMo 网络请求失败: timed out"),
        ):
            with self.assertRaises(api.HTTPException) as raised:
                asyncio.run(api.mimo_design(request))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("HTTPS_PROXY", raised.exception.detail)

    def test_voice_design_request_matches_official_message_contract(self):
        captured = []

        def fake_post(url, payload, headers, **_kwargs):
            captured.append((url, payload, headers))
            return {"choices": [{"message": {"audio": {"data": ""}}}]}

        with mock.patch.object(api, "mimo_post_json_with_retry", side_effect=fake_post), mock.patch.object(
            api, "join_wav_bytes", return_value=b"wav"
        ):
            audio = api.run_mimo_voice_design(
                {
                    "voice_description": "中年女性，醇厚沉稳。",
                    "text": "需要谨慎地传承。",
                    "api_key": "test-key",
                }
            )

        self.assertEqual(audio, b"wav")
        self.assertEqual(len(captured), 1)
        url, payload, headers = captured[0]
        self.assertEqual(url, "https://api.xiaomimimo.com/v1/chat/completions")
        self.assertEqual(headers["api-key"], "test-key")
        self.assertEqual(payload["model"], "mimo-v2.5-tts-voicedesign")
        self.assertEqual(payload["audio"], {"format": "wav"})
        self.assertEqual(
            payload["messages"],
            [
                {"role": "user", "content": "中年女性，醇厚沉稳。"},
                {"role": "assistant", "content": "需要谨慎地传承。"},
            ],
        )

if __name__ == "__main__":
    unittest.main()
