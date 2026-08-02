"""Regression tests for the MOSS VoiceGenerator compatibility fixes."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from moss_voice_design_compat import (  # noqa: E402
    install_moss_decode_compatibility,
    is_moss_codec_path_ready,
    split_sizes_from_break_positions,
    validate_moss_codec_path,
    validate_moss_codec_compatibility,
)


class MossVoiceDesignCompatibilityTests(unittest.TestCase):
    def test_break_positions_become_complete_split_sizes(self):
        self.assertEqual(split_sizes_from_break_positions(106, [80]), [80, 26])
        self.assertEqual(
            split_sizes_from_break_positions(10, [2, 5, 7]),
            [2, 3, 2, 3],
        )

    def test_processor_patch_is_idempotent(self):
        class FakeProcessor:
            def _parse_audio_codes(self, start_length, audio_codes):
                return "old parser"

        processor = FakeProcessor()
        install_moss_decode_compatibility(processor)
        patched_parser = processor._parse_audio_codes
        install_moss_decode_compatibility(processor)

        self.assertTrue(processor._unitale_fixed_audio_parser)
        self.assertIs(processor._parse_audio_codes, patched_parser)

    def test_rejects_the_v2_codec_for_voicegenerator(self):
        class ModelConfig:
            sampling_rate = 24000

        class CodecConfig:
            sampling_rate = 48000
            number_channels = 2

        class FakeProcessor:
            model_config = ModelConfig()
            audio_tokenizer = type("Codec", (), {"config": CodecConfig()})()

        with self.assertRaisesRegex(RuntimeError, "MOSS-Audio-Tokenizer-v2"):
            validate_moss_codec_compatibility(FakeProcessor())

    def test_accepts_the_24khz_mono_codec(self):
        class ModelConfig:
            sampling_rate = 24000

        class Codec:
            sampling_rate = 24000
            number_channels = 1

        class FakeProcessor:
            model_config = ModelConfig()
            audio_tokenizer = Codec()

        validate_moss_codec_compatibility(FakeProcessor())

    def test_rejects_an_empty_codec_directory_before_transformers(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "缺少 config.json"):
                validate_moss_codec_path(directory)
            self.assertFalse(is_moss_codec_path_ready(directory))

    def test_rejects_a_codec_config_without_model_type(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps({"sampling_rate": 24000}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "model_type"):
                validate_moss_codec_path(directory)

    def test_rejects_metadata_only_codec_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_type": "moss-audio-tokenizer",
                        "sampling_rate": 24000,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "未找到模型权重"):
                validate_moss_codec_path(directory)

    def test_accepts_complete_v1_codec_metadata_and_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_type": "moss-audio-tokenizer",
                        "sampling_rate": 24000,
                        "number_channels": 1,
                    }
                ),
                encoding="utf-8",
            )
            (Path(directory) / "model.safetensors").write_bytes(b"test")
            validate_moss_codec_path(directory)
            self.assertTrue(is_moss_codec_path_ready(directory))


if __name__ == "__main__":
    unittest.main()
