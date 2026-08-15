# MOSS-SoundEffect v2 uv 服务

这个目录提供 MOSS-SoundEffect v2 的独立 uv HTTP 控制面和一次性模型 worker，
共用 `moss_soundEffect/.venv`，worker 退出后释放显存。迁移已完成，8311 只由
本目录的 uv 服务提供。

## 运行

先准备 Python 3.12.13、依赖、上游源码和本地模型：

```bash
uv sync --project moss_soundEffect --locked
export MOSS_SOUNDEFFECT_CODE_PATH="$HOME/tts-depency/MOSS-TTS"
export MOSS_SOUNDEFFECT_MODEL_DIR="$HOME/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0"
```

启动 8311：

```bash
HOST=127.0.0.1 PORT=8311 \
  uv run --no-sync --project moss_soundEffect \
  python moss_soundEffect/main.py
```

健康检查和生成：

```bash
curl http://127.0.0.1:8311/v1/health
curl -X POST http://127.0.0.1:8311/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"门吱吱作响的声音，刺耳急促","seconds":1}' \
  -o moss-sfx.wav
```

接口保留 `/v1/generate`、兼容别名 `/v2/synthesize` 和本机内部路由
`/internal/unload_all`。成功响应是 48 kHz 单声道 `audio/wav`；`seconds` 范围
为 `(0, 30]`。模型不会在 API 进程导入，真实推理通过同一 uv Python 启动
`worker.py`。

## 环境变量

- `MOSS_SOUNDEFFECT_MODEL_DIR`：默认 `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0`
- `MOSS_SOUNDEFFECT_CODE_PATH`：默认 `$HOME/tts-depency/MOSS-TTS`
- `MOSS_SOUNDEFFECT_DEVICE`、`MOSS_SOUNDEFFECT_DTYPE`
- `MOSS_SOUNDEFFECT_DEFAULT_SECONDS`、`MOSS_SOUNDEFFECT_DEFAULT_STEPS`
- `MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE`、`MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT`
- `MOSS_SOUNDEFFECT_DEFAULT_SEED`、`MOSS_SOUNDEFFECT_REQUEST_TIMEOUT`
- `RUNTIME_CACHE_DIR`、`GPU_LOCK_FILE`、`LOCAL_FILES_ONLY`、`CUDA_RELEASE_DELAY`

上游 MOSS-TTS 源码和模型权重不提交到本仓库。FlashAttention、SageAttention
均为可选；当前代码在未安装时回退 PyTorch SDPA，首轮迁移不依赖本地
`/home/muyi086/tts-depency/flash-attention`。
