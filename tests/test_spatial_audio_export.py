"""空间音频导出的无模型回归测试。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPOSITORY_DIR / "main"
sys.path.insert(0, str(SERVICE_DIR))

spec = importlib.util.spec_from_file_location(
    "spatial_audio_for_test",
    SERVICE_DIR / "spatial_audio.py",
)
assert spec and spec.loader
spatial_audio = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = spatial_audio
spec.loader.exec_module(spatial_audio)


class SpatialAudioProcessorTests(unittest.TestCase):
    def test_balanced_mp3_export_uses_soxr_spatial_mix_and_two_pass_loudnorm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spatial-audio-tests-") as temporary_dir:
            root = Path(temporary_dir)
            staged_path = root / "upload.wav.part"
            staged_path.write_bytes(b"RIFF-input")
            staged = spatial_audio.StagedUpload(
                path=staged_path,
                sha256="a" * 64,
                size_bytes=10,
                suffix=".wav",
            )
            processor = spatial_audio.SpatialAudioProcessor(
                cache_dir=root / "cache",
                ffmpeg_bin="ffmpeg-test",
                timeout_seconds=30,
            )
            commands: list[list[str]] = []

            def fake_run(command: list[str]) -> str:
                commands.append(command)
                if "-f" in command and "null" in command:
                    return """
                    {
                        "input_i": "-16.98",
                        "input_tp": "-0.97",
                        "input_lra": "9.00",
                        "input_thresh": "-27.98",
                        "target_offset": "-0.67"
                    }
                    """
                Path(command[-1]).write_bytes(b"processed-audio")
                return ""

            with patch.object(processor, "_run_ffmpeg", side_effect=fake_run):
                result = processor.export(staged, profile="balanced", output_format="mp3")

            self.assertEqual(len(commands), 3)
            pre_master_filter = commands[0][commands[0].index("-filter_complex") + 1]
            self.assertIn("aresample=48000:resampler=soxr:precision=28", pre_master_filter)
            self.assertIn("haas=", pre_master_filter)
            self.assertIn("volume=0.38", pre_master_filter)
            analyze_filter = commands[1][commands[1].index("-af") + 1]
            self.assertIn("loudnorm=I=-18:TP=-2.5:LRA=7", analyze_filter)
            master_filter = commands[2][commands[2].index("-af") + 1]
            self.assertIn("measured_I=-16.98", master_filter)
            self.assertIn("measured_TP=-0.97", master_filter)
            self.assertIn("offset=-0.67", master_filter)
            self.assertIn("libmp3lame", commands[2])
            self.assertIn("192k", commands[2])
            self.assertEqual(result.media_type, "audio/mpeg")
            self.assertEqual(result.path.read_bytes(), b"processed-audio")
            self.assertFalse(staged_path.exists())

            result.cleanup()
            self.assertFalse(result.path.parent.exists())

    def test_standard_wav_export_keeps_stereo_and_writes_24_bit_pcm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spatial-audio-tests-") as temporary_dir:
            root = Path(temporary_dir)
            staged_path = root / "upload.wav.part"
            staged_path.write_bytes(b"RIFF-input")
            staged = spatial_audio.StagedUpload(
                path=staged_path,
                sha256="b" * 64,
                size_bytes=10,
                suffix=".wav",
            )
            processor = spatial_audio.SpatialAudioProcessor(cache_dir=root / "cache")
            commands: list[list[str]] = []

            def fake_run(command: list[str]) -> str:
                commands.append(command)
                if "null" in command:
                    return (
                        '{"input_i":"-18.0","input_tp":"-3.0","input_lra":"7.0",'
                        '"input_thresh":"-28.0","target_offset":"0.0"}'
                    )
                Path(command[-1]).write_bytes(b"wav")
                return ""

            with patch.object(processor, "_run_ffmpeg", side_effect=fake_run):
                result = processor.export(staged, profile="standard", output_format="wav")

            pre_master_filter = commands[0][commands[0].index("-filter_complex") + 1]
            self.assertIn("aresample=48000:resampler=soxr:precision=28", pre_master_filter)
            self.assertNotIn("pan=mono", pre_master_filter)
            self.assertNotIn("haas=", pre_master_filter)
            self.assertIn("pcm_s24le", commands[2])
            self.assertEqual(result.media_type, "audio/wav")
            result.cleanup()

    def test_balanced_profile_uses_audible_side_only_wet_bed(self) -> None:
        spatial_filter = spatial_audio.build_pre_master_filter("balanced")

        side_only_pan = "pan=stereo|c0=0.5*c0-0.5*c1|c1=0.5*c1-0.5*c0"
        self.assertEqual(spatial_filter.count(side_only_pan), 2)
        self.assertIn(
            "haas=left_delay=4:right_delay=8:side_gain=1.20",
            spatial_filter,
        )
        self.assertIn("volume=0.38[wide]", spatial_filter)
        self.assertIn(
            "aecho=0.8:0.70:'19|37|61':'0.24|0.14|0.08'",
            spatial_filter,
        )
        self.assertIn("volume=0.16[room]", spatial_filter)

    def test_immersive_profile_is_stronger_without_changing_standard(self) -> None:
        immersive_filter = spatial_audio.build_pre_master_filter("immersive")
        standard_filter = spatial_audio.build_pre_master_filter("standard")

        self.assertIn("volume=0.78[dry]", immersive_filter)
        self.assertIn(
            "haas=left_delay=7:right_delay=13:side_gain=1.45",
            immersive_filter,
        )
        self.assertIn("volume=0.58[wide]", immersive_filter)
        self.assertIn("volume=0.24[room]", immersive_filter)
        self.assertNotIn("haas=", standard_filter)
        self.assertNotIn("aecho=", standard_filter)

    def test_invalid_profile_and_output_format_are_rejected_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spatial-audio-tests-") as temporary_dir:
            processor = spatial_audio.SpatialAudioProcessor(cache_dir=temporary_dir)
            staged = spatial_audio.StagedUpload(
                path=Path(temporary_dir) / "missing.wav",
                sha256="c" * 64,
                size_bytes=1,
                suffix=".wav",
            )

            with self.assertRaisesRegex(spatial_audio.SpatialAudioExportError, "profile"):
                processor.export(staged, profile="unknown", output_format="wav")
            with self.assertRaisesRegex(spatial_audio.SpatialAudioExportError, "output_format"):
                processor.export(staged, profile="balanced", output_format="flac")

    def test_loudnorm_parser_rejects_incomplete_measurements(self) -> None:
        with self.assertRaisesRegex(spatial_audio.SpatialAudioExportError, "loudnorm"):
            spatial_audio.parse_loudnorm_measurement('{"input_i":"-18.0"}')


if __name__ == "__main__":
    unittest.main()
