import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import api
import indextts_worker


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    cuda = FakeCuda()


class FakeIndexTTS2:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.device = "cpu"
        self.use_fp16 = kwargs["use_fp16"]
        self.use_cuda_kernel = kwargs["use_cuda_kernel"]
        self.infer_kwargs = None
        self.__class__.instances.append(self)

    def infer(self, **kwargs):
        self.infer_kwargs = kwargs
        waveform = np.full(1600, 0.25, dtype=np.float32)
        sf.write(kwargs["output_path"], waveform, 16000, format="WAV")


class FakeProcess:
    next_pid = 30000

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = self.__class__.next_pid
        self.__class__.next_pid += 1
        self.returncode = None

    def communicate(self, timeout=None):
        output_index = self.command.index("--output-wav") + 1
        Path(self.command[output_index]).write_bytes(b"RIFF-fake-wave")
        self.returncode = 0
        return "worker completed", ""

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode


class IndexTTSWorkerTests(unittest.TestCase):
    def setUp(self):
        FakeIndexTTS2.instances.clear()

    def test_worker_preserves_inference_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_dir = root / "model"
            code_dir = root / "code"
            runtime_dir = root / "runtime"
            model_dir.mkdir()
            code_dir.mkdir()
            cfg_path = model_dir / "config.yaml"
            ref_audio_path = root / "reference.wav"
            output_path = root / "output.wav"
            cfg_path.write_text("model: fake\n", encoding="utf-8")
            sf.write(ref_audio_path, np.full(800, 0.1, dtype=np.float32), 16000)

            request = {
                "text": "测试文本",
                "ref_audio_path": str(ref_audio_path),
                "emo_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
                "emo_text": None,
                "model_dir": str(model_dir),
                "cfg_path": str(cfg_path),
                "code_dir": str(code_dir),
                "aux_paths": {"bigvgan": str(model_dir / "bigvgan")},
                "device": "cpu",
                "use_fp16": False,
                "use_cuda_kernel": False,
                "num_beams": 1,
                "local_files_only": True,
                "runtime_cache_dir": str(runtime_dir),
                "hf_mirror_dir": str(root / "hf-mirror"),
            }

            with mock.patch.object(
                indextts_worker,
                "import_runtime",
                return_value=(FakeIndexTTS2, np, sf, FakeTorch),
            ):
                indextts_worker.synthesize(request, output_path)

            self.assertTrue(output_path.is_file())
            waveform, sample_rate = sf.read(output_path, dtype="float32")
            self.assertEqual(sample_rate, 16000)
            self.assertGreater(waveform.size, 0)
            instance = FakeIndexTTS2.instances[0]
            self.assertEqual(instance.init_kwargs["device"], "cpu")
            self.assertEqual(instance.infer_kwargs["text"], "测试文本")
            self.assertEqual(instance.infer_kwargs["num_beams"], 1)
            self.assertFalse(instance.infer_kwargs["use_emo_text"])
            self.assertEqual(instance.infer_kwargs["emo_alpha"], 0.6)

    def test_each_request_starts_a_fresh_process_and_cleans_temp_files(self):
        manager = api.ModelManager()
        processes = []

        def start_process(command, **kwargs):
            process = FakeProcess(command, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(api, "INDEXTTS_WORKER_SCRIPT", __file__),
                mock.patch.object(api, "INDEXTTS_WORKER_TMP_DIR", tmp_dir),
                mock.patch.object(api, "CUDA_RELEASE_DELAY", 0),
                mock.patch.object(api.subprocess, "Popen", side_effect=start_process),
            ):
                first = manager.run_indextts_worker({"request": 1})
                second = manager.run_indextts_worker({"request": 2})

            self.assertEqual(first, b"RIFF-fake-wave")
            self.assertEqual(second, b"RIFF-fake-wave")
            self.assertEqual(len(processes), 2)
            self.assertNotEqual(processes[0].pid, processes[1].pid)
            self.assertTrue(all(proc.kwargs["start_new_session"] for proc in processes))
            self.assertIsNone(manager.indextts_process)
            self.assertEqual(list(Path(tmp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
