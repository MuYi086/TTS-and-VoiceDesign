# Repository Guidelines

## Project Structure & Module Organization

This repository provides Unitale's local TTS and voice-design backend. Keep service-facing code in `api/`: `api.py` is the main API, `*_api.py` files expose model-specific HTTP services, and `*_worker.py` files run inference in their respective Conda environments. Shared request models and audio utilities live in `api/synthesis_request.py` and `api/audio_trim.py`. Runtime uploads, prompts, caches, and vendored upstream code belong under `api/prompts/`, `api/.cache/`, and `api/vendor/`; do not add generated audio or model weights to Git. Unit tests live in `tests/`; the standalone MOSS sound-effect example is in `soundEffect/`.

## Build, Test, and Development Commands

Activate the main environment before local work:

```bash
conda activate unitale-tts-local
bash start.sh                         # starts ports 8300–8306
conda run -n unitale-tts-local python -m unittest discover -s tests  # runs repository regression tests
curl http://127.0.0.1:8300/v1/health # checks the main service
```

`start.sh` exports the model paths, ports, cache paths, and worker Conda environments. Override settings through environment variables (for example, `PORT=8400 bash start.sh`) instead of editing machine-specific defaults. Use `soundEffect/run_moss_soundeffect_v2.sh` only for its GPU-backed sound-effect smoke test.

## Coding Style & Naming Conventions

Write Python with four-space indentation, `snake_case` functions and variables, and `PascalCase` Pydantic models. Follow the existing module split: keep HTTP validation and response handling in `*_api.py`, and heavyweight model loading/inference in `*_worker.py`. Preserve request compatibility fields and document any changed API contract in `README.md`. No formatter or linter is configured; match surrounding imports, type hints, docstrings, and line wrapping, and avoid unrelated reformatting.

## Testing Guidelines

Tests use the standard-library `unittest` runner and follow `test_*.py` / `test_*` naming. Add focused regression tests under `tests/` for shared API models, validation behavior, and audio utilities. Avoid tests that download models, require CUDA, or call external services; mock or isolate those boundaries. Run the full discovery command above before submitting changes.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit-style subjects, primarily `feat: <summary>` (Chinese summaries are common). Use a clear type such as `feat:`, `fix:`, or `docs:` and keep each commit scoped. Pull requests should explain affected endpoints or workers, list test commands and results, link relevant issues, and include request/response examples or screenshots when behavior visible to the WebUI changes.

## Security & Configuration

Never commit API keys, local model paths, uploaded reference audio, caches, or generated WAV files. Supply `MIMO_API_KEY` and deployment-specific paths through the environment.
