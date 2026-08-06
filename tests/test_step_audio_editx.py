"""Step-Audio-EditX 编辑 API 的无模型回归测试。"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import api
import step_audio_editx


class StepAudioEditXTests(unittest.TestCase):
    def test_emotion_edit_requires_the_prompt_contract(self):
        request = step_audio_editx.StepAudioEditXEditRequest.model_validate(
            {
                "prompt_audio": "step-audio-editx/line-1.wav",
                "prompt_text": "这是一条台词。",
                "edit_type": "emotion",
                "edit_info": "coldness",
            }
        )

        self.assertEqual(request.generated_text, "这是一条台词。")
        self.assertEqual(request.edit_type, "emotion")
        self.assertEqual(request.edit_info, "coldness")

        with self.assertRaisesRegex(ValueError, "edit_info"):
            step_audio_editx.StepAudioEditXEditRequest.model_validate(
                {
                    "prompt_audio": "step-audio-editx/line-1.wav",
                    "prompt_text": "这是一条台词。",
                    "edit_type": "emotion",
                }
            )

    def test_worker_payload_preserves_edit_type_and_edit_info(self):
        request = step_audio_editx.StepAudioEditXEditRequest.model_validate(
            {
                "prompt_audio": "step-audio-editx/line-1.wav",
                "prompt_text": "这是一条台词。",
                "generated_text": "这是一条台词。",
                "edit_type": "emotion",
                "edit_info": "coldness",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            prompt_audio = Path(directory) / "line-1.wav"
            prompt_audio.write_bytes(b"RIFF-prompt")
            payload = step_audio_editx.StepAudioEditXWorkerManager().build_worker_payload(
                request,
                prompt_audio,
                local_files_only=True,
            )

        self.assertEqual(payload["prompt_wav_path"], str(prompt_audio))
        self.assertEqual(payload["prompt_text"], "这是一条台词。")
        self.assertEqual(payload["generated_text"], "这是一条台词。")
        self.assertEqual(payload["edit_type"], "emotion")
        self.assertEqual(payload["edit_info"], "coldness")
        self.assertTrue(payload["local_files_only"])

    def test_route_resolves_uploaded_audio_and_returns_wav(self):
        request = step_audio_editx.StepAudioEditXEditRequest.model_validate(
            {
                "prompt_audio": "step-audio-editx/line-1.wav",
                "prompt_text": "这是一条台词。",
                "edit_type": "emotion",
                "edit_info": "coldness",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            prompts_dir = Path(directory) / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / api.hash_filename(request.prompt_audio)).write_bytes(b"RIFF-prompt")
            lock_path = Path(directory) / "gpu.lock"
            with (
                mock.patch.object(api, "PROMPTS_DIR", str(prompts_dir)),
                mock.patch.object(api, "GPU_LOCK_FILE", str(lock_path)),
                mock.patch.object(api, "CUDA_RELEASE_DELAY", 0),
                mock.patch.object(
                    api.step_audio_editx_manager,
                    "build_worker_payload",
                    return_value={"edit_type": "emotion", "edit_info": "coldness"},
                ) as build_payload,
                mock.patch.object(
                    api.step_audio_editx_manager,
                    "run_worker",
                    return_value=b"RIFF-edited-wave",
                ) as run_worker,
            ):
                response = asyncio.run(api.step_audio_editx_edit(request))

        self.assertEqual(response.media_type, "audio/wav")
        self.assertEqual(response.body, b"RIFF-edited-wave")
        self.assertEqual(build_payload.call_args.args[0], request)
        self.assertEqual(build_payload.call_args.args[1].name, api.hash_filename(request.prompt_audio))
        run_worker.assert_called_once_with({"edit_type": "emotion", "edit_info": "coldness"})


if __name__ == "__main__":
    unittest.main()
