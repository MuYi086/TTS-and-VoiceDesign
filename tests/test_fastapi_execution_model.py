"""Regression checks for blocking FastAPI handlers.

Every model request starts a one-shot worker, takes the shared GPU file lock,
or performs a blocking upstream HTTP call.  FastAPI must execute those
handlers in its worker thread pool so they cannot stall an application's event
loop and prevent health checks or uploads from being served.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def route_handler_kinds(source_path: Path) -> dict[str, str]:
    """Return whether each decorated application route is sync or async."""
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    handlers: dict[str, str] = {}

    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            if decorator.func.attr not in {"get", "post"}:
                continue
            route = decorator.args[0]
            if isinstance(route, ast.Constant) and isinstance(route.value, str):
                handlers[route.value] = type(node).__name__
    return handlers


class FastApiExecutionModelTests(unittest.TestCase):
    """Keep blocking work off FastAPI event loops without changing API contracts."""

    def test_blocking_routes_use_sync_handlers(self) -> None:
        expected_routes = {
            "main/main.py": {"/v1/health", "/v1/check/audio"},
            "mimo_tts/main.py": {
                "/v1/health",
                "/v1/voice-design/providers",
                "/v1/mimo/timbre",
            },
            "qwen3_tts/main.py": {"/v1/health", "/internal/unload_all", "/v1/check/audio", "/v1/qwen/clone"},
            "voxcpm2/main.py": {"/v1/health", "/internal/unload_all", "/v1/check/audio"},
            "LongCat_AudioDiT_3.5B_bf16/main.py": {
                "/v1/health",
                "/internal/unload_all",
                "/v1/check/audio",
                "/v1/longCat/clone",
            },
            "dots_tts_soar/main.py": {
                "/v1/health",
                "/internal/unload_all",
                "/v1/check/audio",
                "/v2/dotsTTS/clone",
            },
            "moss_soundEffect/main.py": {
                "/v1/health",
                "/internal/unload_all",
                "/v1/moss/soundEffect",
            },
            "stable_audio_3_medium/main.py": {
                "/v1/health",
                "/internal/unload_all",
                "/v1/stableAudio/soundEffect",
            },
            "ace_step_1_5/main.py": {
                "/v1/health",
                "/internal/unload_all",
                "/v1/aceStep/bgm",
            },
            "qwen3_voiceDesign/main.py": {"/v1/health", "/internal/unload_all"},
            "moss_voiceGenerator/main.py": {"/v1/health", "/internal/unload_all"},
            "Step_Audio_EditX/main.py": {"/v1/health", "/internal/unload_all", "/v1/check/audio"},
        }

        for relative_path, routes in expected_routes.items():
            with self.subTest(source=relative_path):
                handler_kinds = route_handler_kinds(REPOSITORY_DIR / relative_path)
                self.assertEqual(
                    {route: handler_kinds.get(route) for route in routes},
                    {route: "FunctionDef" for route in routes},
                )

        voxcpm2_source = (REPOSITORY_DIR / "voxcpm2/main.py").read_text(encoding="utf-8")
        self.assertIn("return await run_in_threadpool(synthesize_voxcpm2_payload, data)", voxcpm2_source)

    def test_async_upload_routes_offload_filesystem_work(self) -> None:
        upload_sources = {
            "main/main.py",
            "qwen3_tts/main.py",
            "voxcpm2/main.py",
            "LongCat_AudioDiT_3.5B_bf16/main.py",
            "dots_tts_soar/main.py",
            "Step_Audio_EditX/main.py",
        }

        for relative_path in upload_sources:
            with self.subTest(source=relative_path):
                source = (REPOSITORY_DIR / relative_path).read_text(encoding="utf-8")
                self.assertIn(
                    "return await run_in_threadpool(store_uploaded_audio,",
                    source,
                )


if __name__ == "__main__":
    unittest.main()
