"""LongCat-AudioDiT 参考音频与生成时长预算回归测试。"""

import sys
import unittest
from pathlib import Path

import numpy as np


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from longcat_audiodit_worker import (
    estimate_duration_frames,
    prompt_frame_budget,
    truncate_prompt_text,
)


class LongCatDurationBudgetTests(unittest.TestCase):
    sample_rate = 24000
    full_hop = 2048
    max_duration = 30.0

    def test_long_prompt_is_reduced_before_model_duration_is_calculated(self):
        prompt_frames = 383
        prompt_text = "这是一段很长的参考音频转写文本" * 4
        chunk = "需要合成的短句"

        budget = prompt_frame_budget(
            chunks=[chunk],
            prompt_text=prompt_text,
            prompt_frames=prompt_frames,
            sample_rate=self.sample_rate,
            full_hop=self.full_hop,
            max_duration=self.max_duration,
            duration_scale=1.0,
            np=np,
        )
        truncated_text = truncate_prompt_text(prompt_text, budget, prompt_frames)
        duration = estimate_duration_frames(
            gen_text=chunk,
            prompt_text=truncated_text,
            prompt_frames=budget,
            sample_rate=self.sample_rate,
            full_hop=self.full_hop,
            max_duration=self.max_duration,
            duration_scale=1.0,
            np=np,
        )

        self.assertLess(budget, prompt_frames)
        self.assertGreater(duration, budget)
        self.assertLessEqual(duration, 351)
        self.assertLess(len(truncated_text), len(prompt_text))

    def test_normal_prompt_stays_within_available_budget(self):
        prompt_frames = 220
        budget = prompt_frame_budget(
            chunks=["测试短句"],
            prompt_text="正常长度的参考音频文本",
            prompt_frames=prompt_frames,
            sample_rate=self.sample_rate,
            full_hop=self.full_hop,
            max_duration=self.max_duration,
            duration_scale=1.0,
            np=np,
        )

        self.assertGreaterEqual(budget, prompt_frames)

    def test_longest_chunk_controls_shared_prompt_budget(self):
        prompt_frames = 383
        prompt_text = "参考文本" * 20
        short_only_budget = prompt_frame_budget(
            chunks=["短句"],
            prompt_text=prompt_text,
            prompt_frames=prompt_frames,
            sample_rate=self.sample_rate,
            full_hop=self.full_hop,
            max_duration=self.max_duration,
            duration_scale=1.0,
            np=np,
        )
        mixed_budget = prompt_frame_budget(
            chunks=["短句", "较长的生成文本" * 8],
            prompt_text=prompt_text,
            prompt_frames=prompt_frames,
            sample_rate=self.sample_rate,
            full_hop=self.full_hop,
            max_duration=self.max_duration,
            duration_scale=1.0,
            np=np,
        )

        self.assertLess(mixed_budget, short_only_budget)


if __name__ == "__main__":
    unittest.main()
