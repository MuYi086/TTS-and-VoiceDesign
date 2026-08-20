# MiMo TTS VoiceDesign uv 服务

该目录提供独立的 MiMo 云端音色设计服务，不依赖 `main/`，也不加载本地模型。服务默认监听 `8303`。

```bash
uv sync --project mimo_tts --locked
MIMO_API_KEY=... uv run --project mimo_tts python mimo_tts/main.py
```

接口：

- `GET /v1/health`
- `POST /v1/mimo/timbre`

成功生成的 WAV 默认缓存到 `storage/timbre/`。可通过 `MIMO_TTS_HOST`、`MIMO_TTS_PORT`、
`MIMO_BASE_URL`、`MIMO_MODEL`、`MIMO_TIMEOUT`、`MIMO_MAX_CHARS_PER_CHUNK`、
`MIMO_PAUSE_MS`、`MIMO_MAX_RETRIES` 和 `TIMBRE_STORAGE_DIR` 覆盖配置。
`MIMO_API_KEY` 只从服务进程环境读取，不能通过请求体覆盖。
