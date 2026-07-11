"""Unit tests for generated-audio leading-silence cleanup."""

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

API_DIR = Path(__file__).resolve().parents[1] / "api"
sys.path.insert(0, str(API_DIR))

from audio_trim import trim_leading_silence
from moss_tts_worker import trim_generated_audio


class LeadingSilenceTrimTests(unittest.TestCase):
    sample_rate = 1000

    def setUp(self):
        self.mono = np.concatenate(
            [
                np.zeros(600, dtype=np.float32),
                np.full(200, 0.2, dtype=np.float32),
            ]
        )

    def test_trims_a_substantial_mono_prefix_but_keeps_pre_roll(self):
        trimmed, trimmed_samples = trim_leading_silence(self.mono, self.sample_rate, np)

        self.assertGreater(trimmed_samples, 500)
        self.assertLess(trimmed_samples, 600)
        self.assertEqual(trimmed.shape, (self.mono.size - trimmed_samples,))

    def test_preserves_both_common_multichannel_layouts(self):
        frame_major = np.column_stack([self.mono, self.mono * 0.5])
        channel_major = frame_major.T

        trimmed_frame_major, frame_major_samples = trim_leading_silence(
            frame_major, self.sample_rate, np
        )
        trimmed_channel_major, channel_major_samples = trim_leading_silence(
            channel_major, self.sample_rate, np
        )

        self.assertEqual(frame_major_samples, channel_major_samples)
        self.assertEqual(trimmed_frame_major.shape[1], 2)
        self.assertEqual(trimmed_channel_major.shape[0], 2)
        self.assertEqual(trimmed_frame_major.shape[0], self.mono.size - frame_major_samples)
        self.assertEqual(trimmed_channel_major.shape[1], self.mono.size - channel_major_samples)

    def test_keeps_a_natural_short_onset(self):
        short_prefix = np.concatenate(
            [
                np.zeros(100, dtype=np.float32),
                np.full(200, 0.2, dtype=np.float32),
            ]
        )

        trimmed, trimmed_samples = trim_leading_silence(short_prefix, self.sample_rate, np)

        self.assertEqual(trimmed_samples, 0)
        np.testing.assert_array_equal(trimmed, short_prefix)

    def test_moss_wrapper_keeps_channel_first_layout(self):
        channel_first = torch.from_numpy(self.mono).unsqueeze(0)

        trimmed, trimmed_samples = trim_generated_audio(
            channel_first, self.sample_rate, np, torch
        )

        self.assertGreater(trimmed_samples, 500)
        self.assertEqual(trimmed.shape, (1, self.mono.size - trimmed_samples))


if __name__ == "__main__":
    unittest.main()
