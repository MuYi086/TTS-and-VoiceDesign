import ast
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_DIR / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import api
import dots_api
import gpu_runtime
import longcat_api
import moss_api
import omnivoice_api
import qwen3_tts_api
import soundeffect_api
import voxcpm2_api


API_MODULES = (
    api,
    dots_api,
    longcat_api,
    moss_api,
    omnivoice_api,
    qwen3_tts_api,
    soundeffect_api,
    voxcpm2_api,
)


class RunningProcess:
    def __init__(self, timeout_once: bool = False):
        self.pid = 43210
        self.returncode = None
        self.timeout_once = timeout_once
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.timeout_once and self.wait_calls == 1:
            raise subprocess.TimeoutExpired("worker", timeout)
        self.returncode = -signal.SIGTERM
        return self.returncode


class GpuRuntimeLifecycleTests(unittest.TestCase):
    def test_api_wrappers_do_not_import_torch_at_module_scope(self):
        for module in API_MODULES:
            source = Path(module.__file__).read_text(encoding="utf-8")
            tree = ast.parse(source)
            top_level_imports = {
                alias.name
                for node in tree.body
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertNotIn("torch", top_level_imports, module.__file__)

    def test_all_local_services_share_one_gpu_lock_file(self):
        lock_files = {str(Path(module.GPU_LOCK_FILE).resolve()) for module in API_MODULES}
        self.assertEqual(len(lock_files), 1)

    def test_cuda_status_uses_nvidia_smi_without_torch(self):
        output = "0, NVIDIA Test GPU, 4096, 16384, 12288\n"
        with (
            mock.patch.object(gpu_runtime.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(
                gpu_runtime.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=output),
            ) as run,
        ):
            status = gpu_runtime.cuda_status()

        self.assertTrue(status["available"])
        self.assertEqual(status["device_name"], "NVIDIA Test GPU")
        self.assertEqual(status["memory"]["free_mib"], 4096.0)
        self.assertEqual(status["memory"]["used_mib"], 12288.0)
        self.assertIsNone(status["memory"]["allocated_mib"])
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/nvidia-smi")

    def test_worker_process_group_is_terminated_and_waited(self):
        process = RunningProcess()
        with mock.patch.object(gpu_runtime.os, "killpg") as killpg:
            gpu_runtime.terminate_process_group(process, "test")

        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertEqual(process.wait_calls, 1)
        self.assertIsNotNone(process.returncode)

    def test_stuck_worker_process_group_escalates_to_sigkill(self):
        process = RunningProcess(timeout_once=True)
        with mock.patch.object(gpu_runtime.os, "killpg") as killpg:
            gpu_runtime.terminate_process_group(process, "test")

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(process.pid, signal.SIGTERM),
                mock.call(process.pid, signal.SIGKILL),
            ],
        )
        self.assertEqual(process.wait_calls, 2)
        self.assertIsNotNone(process.returncode)


if __name__ == "__main__":
    unittest.main()
