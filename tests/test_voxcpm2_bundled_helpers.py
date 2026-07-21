import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import voxcpm2_api
import voxcpm2_helpers


class FakeProcess:
    payloads = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = 33001
        self.returncode = None
        request_index = command.index("--input-json") + 1
        with open(command[request_index], "r", encoding="utf-8") as f:
            type(self).payloads.append(json.load(f))

    def communicate(self, timeout=None):
        output_index = self.command.index("--output-wav") + 1
        Path(self.command[output_index]).write_bytes(b"RIFF-voxcpm2-wave")
        self.returncode = 0
        return "worker completed", ""


class FakeGenerateModel:
    def generate(
        self,
        text,
        prompt_wav_path=None,
        prompt_text=None,
        reference_wav_path=None,
        cfg_value=2.0,
        inference_timesteps=10,
    ):
        raise AssertionError("测试不应执行推理")


class VoxCpm2BundledHelpersTests(unittest.TestCase):
    def test_default_helper_is_bundled_in_repository(self):
        self.assertEqual(
            Path(voxcpm2_api.VOXCPM2_HELPER_DEFAULT),
            API_DIR / "voxcpm2_helpers.py",
        )
        self.assertTrue(Path(voxcpm2_api.VOXCPM2_HELPER_DEFAULT).is_file())
        self.assertEqual(
            voxcpm2_api.resolve_voxcpm2_helper_script(None),
            voxcpm2_api.VOXCPM2_HELPER_DEFAULT,
        )

    def test_missing_legacy_helper_falls_back_but_custom_override_is_preserved(self):
        legacy_path = next(iter(voxcpm2_api.VOXCPM2_EXTERNAL_HELPER_PATHS))
        custom_path = "/opt/unitale/custom_voxcpm2_helper.py"
        with mock.patch.object(voxcpm2_api.os.path, "isfile", return_value=False):
            self.assertEqual(
                voxcpm2_api.resolve_voxcpm2_helper_script(legacy_path),
                voxcpm2_api.VOXCPM2_HELPER_DEFAULT,
            )
            self.assertEqual(
                voxcpm2_api.resolve_voxcpm2_helper_script(custom_path),
                custom_path,
            )

    def test_worker_starts_with_bundled_helper(self):
        manager = voxcpm2_api.VoxCpm2WorkerManager()
        FakeProcess.payloads = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(voxcpm2_api, "VOXCPM2_WORKER_SCRIPT", __file__),
                mock.patch.object(voxcpm2_api, "VOXCPM2_MODEL_DIR", tmp_dir),
                mock.patch.object(voxcpm2_api, "VOXCPM2_WORKER_TMP_DIR", tmp_dir),
                mock.patch.object(
                    voxcpm2_api,
                    "VOXCPM2_HELPER_SCRIPT",
                    voxcpm2_api.VOXCPM2_HELPER_DEFAULT,
                ),
                mock.patch.object(
                    voxcpm2_api,
                    "resolve_conda_executable",
                    return_value="/fake/conda",
                ),
                mock.patch.object(voxcpm2_api.subprocess, "Popen", FakeProcess),
            ):
                audio = manager.run_worker(
                    {"voxcpm2_helper_script": voxcpm2_api.VOXCPM2_HELPER_DEFAULT}
                )

        self.assertEqual(audio, b"RIFF-voxcpm2-wave")
        self.assertEqual(
            FakeProcess.payloads[0]["voxcpm2_helper_script"],
            str(API_DIR / "voxcpm2_helpers.py"),
        )
        self.assertIsNone(manager.last_error)

    def test_clone_arguments_support_reference_only_and_transcript_modes(self):
        args = SimpleNamespace(cfg_value=2.5, inference_timesteps=12)
        ref_audio = Path("reference.wav")

        reference_only = voxcpm2_helpers.generate_kwargs(
            FakeGenerateModel(), args, "测试文本", ref_audio, None
        )
        self.assertEqual(reference_only["reference_wav_path"], "reference.wav")
        self.assertNotIn("prompt_text", reference_only)
        self.assertNotIn("prompt_wav_path", reference_only)

        with_transcript = voxcpm2_helpers.generate_kwargs(
            FakeGenerateModel(), args, "测试文本", ref_audio, "参考音频文本"
        )
        self.assertEqual(with_transcript["prompt_text"], "参考音频文本")
        self.assertEqual(with_transcript["prompt_wav_path"], "reference.wav")
        self.assertEqual(with_transcript["reference_wav_path"], "reference.wav")


if __name__ == "__main__":
    unittest.main()
