"""Regression coverage for successful TTS output persistence."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import audio_output
import longcat_audiodit_api
import qwen3_tts_api
import voxcpm2_api


class AudioOutputTests(unittest.TestCase):
    def test_persist_audio_bytes_uses_model_prefix_and_wav_extension(self):
        with TemporaryDirectory() as output_dir:
            output_path = audio_output.persist_audio_bytes(
                b"RIFF-test-wave", "qwen3_tts", output_dir
            )

            self.assertTrue(output_path.is_file())
            self.assertTrue(output_path.name.startswith("qwen3_tts_"))
            self.assertEqual(output_path.suffix, ".wav")
            self.assertEqual(output_path.read_bytes(), b"RIFF-test-wave")

    def test_persist_audio_bytes_rejects_empty_audio_without_creating_a_file(self):
        with TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "empty audio"):
                audio_output.persist_audio_bytes(b"", "qwen3_tts", output_dir)

            self.assertEqual(list(Path(output_dir).iterdir()), [])

    def test_qwen_manager_persists_successful_worker_audio(self):
        payload = b"RIFF-qwen-wave"

        class FakeProcess:
            returncode = 0

            def __init__(self, command, **_kwargs):
                self.command = command

            def communicate(self, timeout):
                output_path = Path(self.command[self.command.index("--output-wav") + 1])
                output_path.write_bytes(payload)
                return "", ""

        with TemporaryDirectory() as worker_tmp_dir, TemporaryDirectory() as output_dir, patch.object(
            qwen3_tts_api, "QWEN3_TTS_WORKER_TMP_DIR", worker_tmp_dir
        ), patch.object(
            qwen3_tts_api, "QWEN3_TTS_WORKER_SCRIPT", __file__
        ), patch.object(
            qwen3_tts_api, "QWEN3_TTS_MODEL_DIR", worker_tmp_dir
        ), patch.object(
            qwen3_tts_api, "QWEN3_TTS_OUTPUT_DIR", output_dir
        ), patch.object(
            qwen3_tts_api, "resolve_conda_executable", return_value="/usr/bin/conda"
        ), patch.object(
            qwen3_tts_api.subprocess, "Popen", FakeProcess
        ), patch.object(
            qwen3_tts_api, "terminate_process_group"
        ):
            audio = qwen3_tts_api.Qwen3TtsWorkerManager().run_worker({})
            saved_audio = list(Path(output_dir).glob("qwen3_tts_*.wav"))
            saved_bytes = saved_audio[0].read_bytes() if saved_audio else None

        self.assertEqual(audio, payload)
        self.assertEqual(len(saved_audio), 1)
        self.assertEqual(saved_bytes, payload)

    def test_ming_manager_persists_successful_worker_audio(self):
        payload = b"RIFF-ming-wave"
        with TemporaryDirectory() as output_dir, patch.object(
            voxcpm2_api, "MING_OMNI_TTS_OUTPUT_DIR", output_dir
        ), patch.object(voxcpm2_api, "run_local_worker", return_value=payload):
            audio = voxcpm2_api.MingWorkerManager().run_worker({})
            saved_audio = list(Path(output_dir).glob("ming_tts_*.wav"))
            saved_bytes = saved_audio[0].read_bytes() if saved_audio else None

        self.assertEqual(audio, payload)
        self.assertEqual(len(saved_audio), 1)
        self.assertEqual(saved_bytes, payload)

    def test_longcat_manager_persists_successful_worker_audio(self):
        payload = b"RIFF-longcat-wave"
        with TemporaryDirectory() as output_dir, patch.object(
            longcat_audiodit_api, "LONGCAT_AUDIODIT_OUTPUT_DIR", output_dir
        ), patch.object(longcat_audiodit_api, "run_local_worker", return_value=payload):
            audio = longcat_audiodit_api.LongCatAudioDitWorkerManager().run_worker({})
            saved_audio = list(Path(output_dir).glob("longcat_audiodit_*.wav"))
            saved_bytes = saved_audio[0].read_bytes() if saved_audio else None

        self.assertEqual(audio, payload)
        self.assertEqual(len(saved_audio), 1)
        self.assertEqual(saved_bytes, payload)


if __name__ == "__main__":
    unittest.main()
