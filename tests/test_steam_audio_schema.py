"""Steam Audio Render Manifest v1 的无模型校验测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIR / "main"))

from spatial_schema import SpatialRenderManifest  # noqa: E402


def manifest_payload() -> dict[str, object]:
    """返回可按测试场景修改的最小合法对象级 Manifest。"""
    return {
        "version": "1.0",
        "sample_rate": 48000,
        "timeline_duration_ms": 4000,
        "scene": {
            "room": "narrow_corridor",
            "listener_pose": "center_forward",
            "acoustic_quality": "balanced",
        },
        "sources": [
            {
                "id": "dialogue_001",
                "kind": "dialogue",
                "asset_id": "line_audio_001",
                "asset_filename": "line_audio_001.wav",
                "start_ms": 500,
                "trim_start_ms": 100,
                "duration_ms": 2000,
                "playback_rate": 1.0,
                "gain_db": -2.0,
                "loop": False,
                "spatial": {
                    "mode": "point",
                    "location": "rear_right",
                    "distance": "near",
                    "movement": "approaching",
                    "occlusion": "none",
                    "spatial_blend": "strong",
                    "source": "rule",
                },
            }
        ],
    }


class SteamAudioSchemaTests(unittest.TestCase):
    def test_valid_manifest_exposes_only_referenced_upload_names(self) -> None:
        manifest = SpatialRenderManifest.model_validate(manifest_payload())

        self.assertEqual(manifest.version, "1.0")
        self.assertEqual(manifest.referenced_filenames(), {"line_audio_001.wav"})

    def test_rejects_path_traversal_url_windows_path_and_unknown_fields(self) -> None:
        for unsafe in ("../voice.wav", "folder/voice.wav", r"C:\voice.wav", "https://x/a.wav"):
            payload = manifest_payload()
            payload["sources"][0]["asset_filename"] = unsafe  # type: ignore[index]
            with self.subTest(unsafe=unsafe), self.assertRaises(ValidationError):
                SpatialRenderManifest.model_validate(payload)

        payload = manifest_payload()
        payload["sources"][0]["spatial"]["filter_hz"] = 1200  # type: ignore[index]
        with self.assertRaises(ValidationError):
            SpatialRenderManifest.model_validate(payload)

    def test_rejects_duplicate_ids_conflicting_assets_and_timeline_overflow(self) -> None:
        duplicate = manifest_payload()
        duplicate["sources"].append(dict(duplicate["sources"][0]))  # type: ignore[union-attr,index]
        with self.assertRaisesRegex(ValidationError, "source id 重复"):
            SpatialRenderManifest.model_validate(duplicate)

        conflict = manifest_payload()
        second = dict(conflict["sources"][0])  # type: ignore[index]
        second.update({"id": "dialogue_002", "asset_filename": "different.wav"})
        conflict["sources"].append(second)  # type: ignore[union-attr]
        with self.assertRaisesRegex(ValidationError, "映射到多个"):
            SpatialRenderManifest.model_validate(conflict)

        overflow = manifest_payload()
        overflow["sources"][0]["duration_ms"] = 3900  # type: ignore[index]
        with self.assertRaisesRegex(ValidationError, "超出 timeline"):
            SpatialRenderManifest.model_validate(overflow)

    def test_bgm_must_preserve_stereo_and_only_beds_may_loop(self) -> None:
        bgm = manifest_payload()
        source = bgm["sources"][0]  # type: ignore[index]
        source["kind"] = "bgm"
        source["loop"] = True
        with self.assertRaisesRegex(ValidationError, "preserve_stereo"):
            SpatialRenderManifest.model_validate(bgm)

        dialogue = manifest_payload()
        dialogue["sources"][0]["loop"] = True  # type: ignore[index]
        with self.assertRaisesRegex(ValidationError, "只有 bgm 或 ambience"):
            SpatialRenderManifest.model_validate(dialogue)

    def test_rejects_renderer_v1_capabilities_that_are_not_implemented(self) -> None:
        for field, value, message in (
            ("mode", "diffuse", "尚未实现 diffuse"),
            ("occlusion", "wooden_door", "尚未实现遮挡"),
        ):
            payload = manifest_payload()
            payload["sources"][0]["spatial"][field] = value  # type: ignore[index]
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, message):
                SpatialRenderManifest.model_validate(payload)

    def test_rejects_timeline_longer_than_two_hours(self) -> None:
        payload = manifest_payload()
        payload["timeline_duration_ms"] = 2 * 60 * 60 * 1000 + 1
        with self.assertRaises(ValidationError):
            SpatialRenderManifest.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
