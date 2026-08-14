# Repository Guidelines

## Project Structure & Module Organization

This repository is Unitale's local TTS, voice-design, and sound-effect backend.

- `api/` contains the HTTP entry points (`api.py` and model-specific `*_api.py` files), heavyweight inference workers (`*_worker.py`), shared request/audio helpers, and vendored upstream code.
- The main API listens on `8300`; dedicated services use `8305` (Qwen3-TTS), `8306` (VoxCPM2), `8307` (LongCat), `8308` (dots.tts-soar), `8311` (MOSS sound effects), and `8313` (Stable Audio 3 Medium).
- `tests/` contains standard-library `unittest` regression tests. `soundEffect/` contains the GPU-backed MOSS example and smoke test; `README.md` documents API contracts and model setup.
- `api/prompts/`, `api/.cache/`, `api/tempAudio/`, and `api/vendor/` may contain runtime files, caches, generated WAVs, or local dependencies. Do not commit model weights, uploaded/reference audio, generated audio, or machine-specific paths.

## Build, Test, and Development Commands

Ensure Conda and uv are available, then run:

```bash
bash start.sh
uv run --project qwen3_tts python -m unittest discover -s tests -v
curl http://127.0.0.1:8300/v1/health
```

`start.sh` launches all seven HTTP wrappers and their dedicated workers; Qwen3-TTS 8305 runs from `qwen3_tts/.venv`, while the remaining lightweight wrappers use the shared `moss-soundEffect` Conda environment by default. Override ports, model locations, environments, and caches with environment variables (for example, `PORT=8400 bash start.sh`) rather than editing host-specific defaults. Use `soundEffect/run_moss_soundeffect_v2.sh` only for its CUDA/model smoke test.

## Coding Style & Architecture

Use four-space Python indentation, `snake_case` for functions/variables, and `PascalCase` for Pydantic models. Keep request validation, HTTP responses, and compatibility behavior in `*_api.py`; keep model loading and inference in `*_worker.py`. Heavy workers run per request and coordinate through `GPU_LOCK_FILE`, so preserve cleanup and locking behavior. Match existing imports, type hints, docstrings, and line wrapping; no formatter or linter is configured.

Preserve existing routes and compatibility fields. Add or update focused tests and document any API contract change in `README.md`. Model-specific defaults generally live at the top of the relevant API module; `start.sh` should primarily provide routing, paths, environments, and runtime configuration.

## Testing Guidelines

Name files `test_*.py` and methods `test_*`. Tests must avoid downloading models, requiring CUDA, or calling external services; mock workers, subprocesses, and filesystem boundaries instead. Run the full discovery command before submitting changes.

## Commits, Pull Requests & Security

Use concise Conventional Commit-style subjects such as `feat:`, `fix:`, or `docs:`; recent commits commonly use Chinese summaries. PRs should describe affected endpoints/workers, list test commands and results, link relevant issues, and include request/response examples or screenshots for WebUI-visible changes.

Keep `MIMO_API_KEY` and deployment-specific paths in environment variables. Never commit secrets, local model directories, caches, uploaded audio, or generated WAV files.
