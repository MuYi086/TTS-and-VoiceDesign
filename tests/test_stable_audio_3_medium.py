"""Regression coverage for the Stable Audio 3 Medium adapter without loading weights."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError


API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import stable_audio_3_medium_api
import stable_audio_3_medium_worker


class StableAudio3MediumTests(unittest.TestCase):
    def create_model_dir(self, root: Path) -> Path:
        model_dir = root / "model"
        for name in stable_audio_3_medium_api.REQUIRED_MODEL_FILES:
            path = model_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")
        return model_dir

    def test_request_enforces_official_380_second_limit(self):
        request = stable_audio_3_medium_api.StableAudio3MediumGenerateRequest(
            prompt="A short wooden knock. TrackType: SFX", seconds=380
        )
        self.assertEqual(request.seconds, 380)

        official_request = stable_audio_3_medium_api.StableAudio3MediumGenerateRequest(
            prompt="A short wooden knock. TrackType: SFX", duration=12
        )
        self.assertEqual(official_request.seconds, 12)

        with self.assertRaises(ValidationError):
            stable_audio_3_medium_api.StableAudio3MediumGenerateRequest(
                prompt="A short wooden knock. TrackType: SFX", seconds=380.1
            )
        with self.assertRaises(ValidationError):
            stable_audio_3_medium_api.StableAudio3MediumGenerateRequest(
                prompt="A short wooden knock. TrackType: SFX", seconds=3, duration=4
            )

    def test_payload_keeps_moss_compatible_seconds_and_medium_defaults(self):
        request = stable_audio_3_medium_api.StableAudio3MediumGenerateRequest(
            prompt="A dry metal latch clicks shut. TrackType: SFX", seconds=1.5
        )

        payload = stable_audio_3_medium_api.StableAudio3MediumWorkerManager().build_worker_payload(
            request
        )

        self.assertEqual(payload["prompt"], "A dry metal latch clicks shut. TrackType: SFX")
        self.assertEqual(payload["seconds"], 1.5)
        self.assertEqual(payload["steps"], 8)
        self.assertEqual(payload["cfg_scale"], 1.0)
        self.assertEqual(payload["seed"], -1)
        self.assertEqual(payload["device"], "cuda")
        self.assertEqual(payload["dtype"], "float16")

    def test_generate_endpoint_returns_worker_wav_for_moss_compatible_request(self):
        with patch.object(
            stable_audio_3_medium_api,
            "CUDA_RELEASE_DELAY",
            0,
        ), patch.object(
            stable_audio_3_medium_api.manager,
            "run_worker",
            return_value=b"RIFF-stable-audio-medium-wave",
        ) as worker:
            with TestClient(stable_audio_3_medium_api.app) as client:
                response = client.post(
                    "/v1/generate",
                    json={
                        "prompt": "A short wooden knock. TrackType: SFX",
                        "seconds": 1,
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.headers["access-control-allow-origin"], "*")
        self.assertEqual(response.content, b"RIFF-stable-audio-medium-wave")
        self.assertEqual(worker.call_args.args[0]["seconds"], 1)

    def test_manager_persists_successful_worker_audio_to_temp_audio(self):
        audio = b"RIFF-stable-audio-medium-wave"
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_dir = self.create_model_dir(root)
            upstream_path = root / "stable-audio-3"
            (upstream_path / "stable_audio_3").mkdir(parents=True)
            output_dir = root / "tempAudio"
            with patch.object(
                stable_audio_3_medium_api,
                "STABLE_AUDIO_3_MEDIUM_MODEL_DIR",
                model_dir,
            ), patch.object(
                stable_audio_3_medium_api,
                "STABLE_AUDIO_3_REPO_PATH",
                upstream_path,
            ), patch.object(
                stable_audio_3_medium_api,
                "STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR",
                output_dir,
            ), patch.object(
                stable_audio_3_medium_api,
                "run_local_worker",
                return_value=audio,
            ):
                result = stable_audio_3_medium_api.StableAudio3MediumWorkerManager().run_worker(
                    {"prompt": "test"}
                )
                saved_audio = list(output_dir.glob("stable_audio_3_medium_*.wav"))
                self.assertEqual(result, audio)
                self.assertEqual(len(saved_audio), 1)
                self.assertEqual(saved_audio[0].read_bytes(), audio)

    def test_incomplete_model_is_reported_without_starting_worker(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_dir = root / "incomplete-model"
            model_dir.mkdir()
            upstream_path = root / "stable-audio-3"
            (upstream_path / "stable_audio_3").mkdir(parents=True)
            with patch.object(
                stable_audio_3_medium_api,
                "STABLE_AUDIO_3_MEDIUM_MODEL_DIR",
                model_dir,
            ), patch.object(
                stable_audio_3_medium_api,
                "STABLE_AUDIO_3_REPO_PATH",
                upstream_path,
            ), patch.object(stable_audio_3_medium_api, "run_local_worker") as worker:
                with self.assertRaisesRegex(RuntimeError, "本地权重不完整"):
                    stable_audio_3_medium_api.StableAudio3MediumWorkerManager().run_worker({})

                worker.assert_not_called()

    def test_worker_patches_text_encoder_to_the_local_checkpoint(self):
        with TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory) / "model"
            local_text_encoder = model_dir / "t5gemma-b-b-ul2"
            local_text_encoder.mkdir(parents=True)
            original = {
                "model": {
                    "conditioning": {
                        "configs": [
                            {
                                "id": "prompt",
                                "config": {
                                    "repo_id": "stabilityai/stable-audio-3-medium",
                                    "subfolder": "t5gemma-b-b-ul2",
                                },
                            }
                        ]
                    }
                }
            }

            patched = stable_audio_3_medium_worker.patch_local_text_encoder_path(
                original, model_dir
            )

        config = patched["model"]["conditioning"]["configs"][0]["config"]
        self.assertEqual(config["model_path"], str(local_text_encoder.resolve()))
        self.assertNotIn("repo_id", config)
        self.assertNotIn("subfolder", config)
        self.assertEqual(
            original["model"]["conditioning"]["configs"][0]["config"]["repo_id"],
            "stabilityai/stable-audio-3-medium",
        )

    def test_worker_reads_medium_sample_size_from_model_config(self):
        class FakeModel:
            model_config = {"sample_size": 16_777_216}

        self.assertEqual(stable_audio_3_medium_worker.model_sample_size(FakeModel()), 16_777_216)

    def test_worker_explicitly_releases_cuda_allocator(self):
        calls = []

        class FakeCuda:
            def is_available(self):
                return True

            def synchronize(self):
                calls.append("synchronize")

            def empty_cache(self):
                calls.append("empty_cache")

            def ipc_collect(self):
                calls.append("ipc_collect")

        class FakeTorch:
            cuda = FakeCuda()

        stable_audio_3_medium_worker.clear_cuda_cache(FakeTorch())
        self.assertEqual(calls, ["synchronize", "empty_cache", "ipc_collect"])

    def test_medium_worker_rejects_non_official_device_and_dtype(self):
        with self.assertRaisesRegex(ValueError, "device must be cuda"):
            stable_audio_3_medium_worker.resolve_device("cpu")
        with self.assertRaisesRegex(ValueError, "dtype must be float16"):
            stable_audio_3_medium_worker.resolve_model_half("float32")
        self.assertTrue(stable_audio_3_medium_worker.resolve_model_half("float16"))


if __name__ == "__main__":
    unittest.main()
