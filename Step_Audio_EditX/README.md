# Step-Audio-EditX uv 服务

该目录提供 Step-Audio-EditX 的轻量 HTTP 控制面和一次性推理 worker。模型权重、Tokenizer
和官方源码继续使用外部目录，不复制到项目中。

## 安装与启动

依赖已锁定在 `uv.lock`，普通包使用阿里云 PyPI，Torch CUDA 12.8 wheel 使用阿里云显式
cu128 索引。手动准备好 wheel 后执行：

```bash
uv sync --project Step_Audio_EditX --locked
uv pip check --python Step_Audio_EditX/.venv/bin/python
```

启动服务：

```bash
STEP_AUDIO_EDITX_HOST=127.0.0.1 \
STEP_AUDIO_EDITX_PORT=8316 \
uv run --project Step_Audio_EditX python Step_Audio_EditX/main.py
```

`start.sh` 默认以 `uv` 模式启动 8316；该服务完整提供
`/v1/upload_audio`、`/v1/check/audio` 和 `/v1/step-audio-editx/edit`，不再经过主 API
代理，也不再保留 Conda 回退路径。

## 外部路径

可通过环境变量覆盖：

- `STEP_AUDIO_EDITX_MODEL_DIR`：Step-Audio-EditX 模型目录
- `STEP_AUDIO_TOKENIZER_PATH`：Step-Audio-Tokenizer 目录
- `STEP_AUDIO_EDITX_CODE_PATH`：官方 Step-Audio-EditX 源码目录
- `PROMPTS_DIR`、`GPU_LOCK_FILE`：共享 prompt 资产和 GPU 锁

worker 只在请求期间加载模型，结束后退出释放显存。`LOCAL_FILES_ONLY=1` 时会设置
Hugging Face/Transformers 离线变量。当前推理使用 `VLLM_ATTENTION_BACKEND=TRITON_ATTN`，
不要求 `flash_attn`；`funasr==1.4.0`、`sox==1.5.0` 和 `ffmpeg-python==0.2.0` 是
Tokenizer 导入/音频处理链的必要 Python 依赖，`torchcodec==0.9.1` 是当前
`torchaudio==2.9.1` 读取输入音频所需的依赖；系统还需要 `sox` 和 `ffmpeg` 命令。
项目也声明了 `hdbscan==0.8.41` 与 `rotary-embedding-torch==0.8.9`，用于 FunASR 的
说话人分离和 MossFormer 可选路径。
