# Stable Audio 3 Medium uv service

这是独立的 Stable Audio 3 Medium uv 服务，默认监听 `8311`。
HTTP 进程只负责校验请求、持有共享 GPU 锁和返回 WAV；每个请求由当前项目的
Python 3.12.13 启动一次性 `worker.py`，worker 退出后释放模型显存。

## 启动

先在本项目内完成依赖同步，并准备外置的官方 Stable Audio 3 源码和
`hf-mirror` 权重：

```bash
cd stable_audio_3_medium
uv sync --locked
export STABLE_AUDIO_3_REPO_PATH="$HOME/tts-depency/stable-audio-3"
uv run python main.py
```

默认读取：

```text
$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium
```

可通过 `STABLE_AUDIO_3_MEDIUM_MODEL_DIR`、`STABLE_AUDIO_3_REPO_PATH`、
`HF_MIRROR_DIR`、`RUNTIME_CACHE_DIR`、`GPU_LOCK_FILE` 和
`STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR` 覆盖路径，默认保存到根项目的
`storage/soundEffect/`。旧的全局 `TTS_OUTPUT_DIR` 不参与 Stable Audio 的目录解析。
`LOCAL_FILES_ONLY=1` 时不会
从 Hugging Face 联网下载权重。

## FlashAttention 结论

上游 `stable-audio-3` 当前源码在没有 `flash_attn` 时会回退到
`flex_attention`/分块 SDPA，因此 FlashAttention 不是该源码的硬性 import
依赖；本服务默认允许这个回退，并在 `/v1/health` 的
`runtime.flash_attention` 中报告实际模式。

FlashAttention 仍是官方 Medium 的推荐高性能路径，能够降低长音频的显存和
运行时间。若要严格要求它：

```bash
export STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN=1
```

迁移前 Conda 环境里的 `cp310` wheel 不能安装到 Python 3.12。目标环境需要
匹配 `cp312`、Torch 2.7、CUDA 12.6 的 wheel，或在有 `nvcc`、`ninja` 和
CUDA devel toolkit 的环境中从固定版本的 `/home/muyi086/tts-depency/flash-attention`
源码构建。当前本机 checkout 缺少 `csrc/cutlass` 子模块且没有 `nvcc`，不能
把它当作已经可用的 cp312 二进制依赖。

## API 兼容性

保留以下路由和字段：

- `GET /v1/health`
- `POST /v1/stableAudio/soundEffect`
- `prompt`、`seconds`、`duration`、`steps`、`cfg_scale`、`seed`、`device`、`dtype`

生成成功返回 `audio/wav`。`seconds` 是 WebUI 兼容字段，`duration` 是官方
Stable Audio 别名；默认值和上限与旧 API 保持一致。

## 测试

测试不加载模型、不调用 CUDA 推理，也不访问外部服务：

```bash
uv run python -m unittest discover -s tests -v
```

真实 GPU canary 需要完整权重、CUDA 和兼容的 Stable Audio 源码：

```bash
curl -fsS http://127.0.0.1:8311/v1/health
curl -fsS -X POST http://127.0.0.1:8311/v1/stableAudio/soundEffect \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A short glass shattering sound effect, crisp and dramatic", "seconds":1}' \
  -o /tmp/stable_audio_3_medium.wav
```
