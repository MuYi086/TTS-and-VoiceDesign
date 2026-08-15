# dots.tts-soar → `dots_tts_soar` 迁移计划

> **task30 完成状态（2026-08-15）**：独立 uv 服务已确认接管 8308，旧
> `api/dots_tts_soar_api.py`、旧 worker 及 `dots_tts_soar` Conda 环境已删除。
> 本文保留迁移过程中的基线、验证记录和历史回滚方案；当前启动链路不再提供
> Conda 回退入口。

> 原始模型/环境：`dots.tts-soar` / `dots_tts_soar`
>
> 目标 uv 项目：`TTS-and-VoiceDesign/dots_tts_soar/`
>
> 目标 Python：`3.12.13`
>
> 评估日期：2026-08-15
>
> 评估依据：`task28.md`、`项目升级评估.md`、当前 `api/dots_tts_soar_api.py`、
> `api/dots_tts_soar_worker.py`、`start.sh`、`scoring-for-TTS` 中的 dots.tts
> 安装指南和脚本、实际 `dots_tts_soar` Conda 环境，以及
> `/home/muyi086/tts-depency` 的当前目录。

## 1. 结论

### 1.1 可以迁移，但“复刻 Conda 环境”需要准确理解

可以按模型创建 `dots_tts_soar/`，使用 `uv init` 建立 Python 3.12.13 项目，
把 `api/dots_tts_soar_api.py` 的 HTTP 控制面和
`api/dots_tts_soar_worker.py` 的一次性推理 worker 迁入该项目，最后让
`start.sh` 通过新项目启动 8308 服务。

这次迁移的可行性较高，原因是：

1. 官方 `dots.tts` 包声明支持 `>=3.10,<3.13`，Python 3.12.13 在其支持范围内。
2. 当前 worker 的模型加载只发生在子进程中，HTTP API 本身不需要在导入时加载
   模型权重；这适合 uv 项目把 API 和 worker 放在同一个 `.venv` 中。
3. 当前环境使用的是标准 PyTorch CUDA wheel、PyPI 音频库和官方 `dots.tts`
   源码包，没有发现必须从 `/home/muyi086/tts-depency` 引入的模型源码。
4. 模型权重已经是外部 Hugging Face 镜像目录，迁移不需要复制或提交模型资产。

但是 uv 不能逐项替代 Conda 的所有内容。下面这些仍属于宿主机或系统层，不能
仅靠 `pyproject.toml` 解决：

- NVIDIA 驱动、GPU 架构和可用显存；
- PyTorch wheel 携带的 CUDA 运行时与驱动兼容性；
- Linux 的 `fcntl`、进程组和 `nvidia-smi` 能力；
- `soundfile` 使用的 libsndfile/native 音频运行库；
- `/home/muyi086/hf-mirror/rednote-hilab/dots.tts-soar` 模型权重；
- 官方 Git 仓库的源码下载能力，除非提前放置一个本地源码副本。

因此本计划中的“完整复刻”定义为：使用 uv 锁定 Python、HTTP 依赖、模型运行
依赖和官方源码 commit；继续使用宿主机 CUDA 能力和外置模型权重；通过无模型
导入、健康检查、mock 契约测试和真实 GPU canary 证明迁移前后行为一致。

### 1.2 推荐迁移边界

```text
TTS-and-VoiceDesign/
├── dots_tts_soar/
│   ├── pyproject.toml       # API + worker 依赖、镜像和 Git 源配置
│   ├── uv.lock              # 解析成功后提交
│   ├── .python-version      # 3.12.13
│   ├── main.py              # 原 dots_tts_soar_api.py 的 HTTP 控制面
│   ├── worker.py            # 一请求一进程的 dots.tts 推理
│   ├── runtime.py           # worker 启动、超时、进程组和临时文件边界
│   ├── audio_trim.py        # 现有 dots worker 使用的前导静音处理
│   └── README.md            # 项目运行、资产和 GPU 验证说明
└── start.sh                     # 固定通过 uv 项目启动 8308
```

建议新项目自包含 `main.py`、`worker.py` 所需的轻量共享逻辑，或者将其整理到
`runtime.py`；不要让新项目运行时依赖当前仓库根目录的偶然 `sys.path` 顺序。
迁移完成后旧 `api/dots_tts_soar_*` 文件和独立 Conda 环境不再属于运行闭包，
已按 task30 删除。

## 2. 当前运行链路与必须保留的功能

### 2.1 当前链路

迁移后的 `start.sh` dots 服务是：

```text
start.sh
  └─ uv run --no-sync --project "$DOTS_TTS_SOAR_PROJECT_DIR" \
       python "$DOTS_TTS_SOAR_PROJECT_DIR/main.py"
       └─ sys.executable dots_tts_soar/worker.py
```

HTTP 控制面和模型推理 worker 共用目标 uv 项目的 Python，不再依赖共享
Conda 环境中恰好存在的 API 包。

### 2.2 端口、路由和响应契约

这些内容不能因目录和环境迁移改变：

| 方法 | 路由 | 作用 | 响应约束 |
| --- | --- | --- | --- |
| `GET` | `/v1/health` | 检查 worker、模型、路径、Conda/uv、CUDA | JSON；保留现有关键字段，并增加新 uv 状态也可以 |
| `POST` | `/v1/upload_audio` | 上传参考音频和可选逐字稿 | JSON；保留逻辑文件名、大小、SHA-256、`has_prompt_text` |
| `GET` | `/v1/check/audio` | 检查参考音频是否存在 | JSON；保留 `exists`、`size_bytes`、`sha256`、`has_prompt_text` |
| `POST` | `/v2/synthesize` | 参考音频声音克隆 | 成功响应仍为 `audio/wav` 原始字节 |
| `POST` | `/internal/unload_all` | 本机内部兼容路由 | 仅接受本机请求，返回 JSON；worker 是一次性进程 |

默认端口继续为 `8308`，继续支持 `HOST`、`PORT`、`HF_MIRROR_DIR`、
`PROMPTS_DIR`、`RUNTIME_CACHE_DIR`、`GPU_LOCK_FILE`、`LOCAL_FILES_ONLY`、
`DOTS_TTS_SOAR_MODEL_DIR` 以及现有 dots 参数环境变量。

### 2.3 `/v2/synthesize` 的请求和模型约束

以下行为属于 WebUI 可见契约，迁移时必须逐项回归：

- `text` 和 `audio_path` 仍为必填字段；请求中的多余兼容字段继续忽略。
- `prompt_text` 仍可直接传入；未传时从上传接口保存的 sidecar 读取。
- `prompt_text` 必须是参考音频的准确逐字稿，推荐使用 continuation voice
  cloning；未提供时保留官方支持的 x-vector-only cloning。
- `language`、`template_name`、`precision`、`seed`、`ode_method`、`num_steps`、
  `guidance_scale`、`speaker_scale`、`max_generate_length`、
  `max_chars_per_chunk`、`pause_ms`、`normalize_text`、`profile_inference`
  等参数继续支持当前 API 的覆盖方式。
- 文本标题符号、Markdown 列表符号和空文本校验继续保留。
- 文本分片、每片前导静音裁剪、片段间停顿和最终前导静音兜底处理继续保留。
- 输出为 `48000 Hz`、单声道 WAV；不能变成 JSON、立体声或其他采样率。
- 请求仍由共享 `GPU_LOCK_FILE` 串行化；一请求一个 worker；成功、异常、超时
  和取消后都要结束进程组并释放 CUDA 上下文。

当前 API 入口中的默认值为：

| 参数 | 默认值 |
| --- | ---: |
| `precision` | `bfloat16` |
| `language` | `chinese` |
| `ode_method` | `euler` |
| `num_steps` | `10` |
| `guidance_scale` | `1.2` |
| `speaker_scale` | `1.5` |
| `max_generate_length` | `500` |
| `max_chars_per_chunk` | `120`，`0` 表示不分片 |
| `pause_ms` | `250` |
| `seed` | `42` |
| `normalize_text` | `false` |
| `profile_inference` | `false` |
| `request_timeout` | `900` 秒 |

## 3. 本机环境和资产审计

### 3.1 迁移前 Conda 基线（已删除）

本机实际检查结果如下，不能把当前版本直接当成 Python 3.12 的已验证结果：

```text
environment: dots_tts_soar
prefix:      /home/muyi086/miniconda3/envs/dots_tts_soar
Python:      3.10.20
dots.tts:    0.2.1
Torch:       2.8.0+cu128
Torch CUDA:  12.8
CUDA usable: torch.cuda.is_available() == True
pip check:   No broken requirements found.
```

当前 `dots-tts` 的 `direct_url.json` 证明其来源是：

```text
https://github.com/rednote-hilab/dots.tts.git
commit 4947d364baa5afb2daf1feb00a247b5f23f97878
```

当前官方包元数据为 `dots.tts==0.2.1`，并声明 `Requires-Python: >=3.10,<3.13`。
因此目标环境必须锁为 `3.12.13`，不能使用 Python 3.13；同时必须在新环境
完成真正的 import 和推理验证，不能只因为版本范围允许就宣称迁移成功。

### 3.2 模型资产

当前模型目录为：

```text
$HF_MIRROR_DIR/rednote-hilab/dots.tts-soar
默认展开路径：/home/muyi086/hf-mirror/rednote-hilab/dots.tts-soar
目录大小：约 4.9G
```

已检查到的关键文件包括：

```text
config.json
model.safetensors
speaker_encoder.safetensors
vocoder.safetensors
latent_stats.pt
tokenizer.json
tokenizer_config.json
special_tokens_map.json
added_tokens.json
chat_template.jinja
vocab.json
merges.txt
```

`config.json` 的 vocoder 采样率为 `48000`，模型输出因此必须继续写成 48 kHz
单声道 WAV。模型权重、上传参考音频、运行生成音频和 Hugging Face 缓存都不
进入 Git，也不放进 `dots_tts_soar/` 项目目录。

如果部署机尚未准备模型，可以使用 hf-mirror 下载；这一步不由 `uv sync` 负责：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"

uv run --project dots_tts_soar hf download \
  rednote-hilab/dots.tts-soar \
  --local-dir "$HF_MIRROR_DIR/rednote-hilab/dots.tts-soar"
```

生产/离线运行继续设置：

```bash
export HF_HOME="$HF_MIRROR_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 3.3 `/home/muyi086/tts-depency` 审计

当前目录中与模型服务相关的仓库有：

| 目录 | 是否属于 dots.tts-soar 运行闭包 | 结论 |
| --- | --- | --- |
| `LongCat-AudioDiT` | 否 | 只供 LongCat 服务使用 |
| `MOSS-Audio` | 否 | 只供 MOSS 音频/声效相关能力使用 |
| `MOSS-TTS` | 否 | 只供 MOSS TTS/VoiceGenerator 使用 |
| `Step-Audio-EditX` | 否 | 只供 Step-Audio-EditX 使用 |
| `stable-audio-3` | 否 | 只供 Stable Audio 3 Medium 使用 |
| `flash-attention` | 否 | 当前 dots.tts worker 不导入它；不要因目录存在而加入依赖 |
| `vllm` | 否 | 当前 dots.tts worker 不使用 vLLM |

结论是：当前没有需要复制或通过路径注入到 dots 项目的本地运行仓库。`scoring-for-TTS`
中的 `tts_local_dots_tts_soar.py` 和安装指南可以继续作为人工 GPU smoke test
参考，但它们不是 HTTP 服务的运行时依赖。

如果部署环境不能访问 GitHub，才需要准备一个可审计的本地官方源码副本：

```bash
git clone https://github.com/rednote-hilab/dots.tts.git \
  /home/muyi086/tts-depency/dots.tts
git -C /home/muyi086/tts-depency/dots.tts \
  checkout 4947d364baa5afb2daf1feb00a247b5f23f97878
```

这只是离线安装备选路径，不应把该仓库复制进本项目或提交到本项目 Git。

## 4. 依赖闭包审计

### 4.1 API 控制面直接依赖

`api/dots_tts_soar_api.py` 及其共享模块实际需要的第三方包为：

| 分发包 | 当前环境版本 | 用途 | 目标处理 |
| --- | ---: | --- | --- |
| `fastapi` | `0.141.1` | HTTP 路由、请求模型、上传 | 目标项目显式锁定 |
| `starlette` | `1.6.0` | `BaseHTTPMiddleware` | 目标项目显式锁定 |
| `uvicorn` | `0.52.1` | 8308 ASGI 服务 | 目标项目显式锁定 |
| `pydantic` | `2.12.5` | `CloneSynthesisRequest` 和字段校验 | 目标项目显式锁定 |
| `python-multipart` | `0.0.32` | `UploadFile`、`File`、`Form` | 目标项目显式锁定 |

`hashlib`、`fcntl`、`json`、`os`、`pathlib`、`subprocess`、`tempfile`、
`threading`、`traceback`、`typing` 等是 Python/Linux 标准库或系统能力，不应
写成 PyPI 依赖。

### 4.2 worker 直接依赖

`api/dots_tts_soar_worker.py` 的 import 闭包为：

| 分发包 | 当前环境版本 | 直接用途 | 目标处理 |
| --- | ---: | --- | --- |
| `dots.tts` | `0.2.1` | `DotsTtsRuntime`、文本处理和模型推理 | 固定官方 Git commit |
| `torch` | `2.8.0+cu128` | CUDA 张量、模型和推理上下文 | 从 cu128 PyTorch wheel 源显式解析 |
| `torchaudio` | `2.8.0+cu128` | fbank、重采样和音频特征 | 与 torch 同版本/同 CUDA 源 |
| `numpy` | `2.2.6` | worker 音频拼接、前导静音处理 | 显式锁定 |
| `soundfile` | `0.13.1` | WAV 写出 | 显式锁定，并检查 native libsndfile |

worker 通过 `from api.audio_trim import ...` 的当前逻辑只用到 NumPy；迁移到
新目录后应把这一小段逻辑随项目迁入，避免 uv worker 依赖旧 `api` 目录。

### 4.3 `dots.tts==0.2.1` 的官方依赖

当前安装包元数据实际声明了以下运行依赖；这些是模型运行闭包，不是凭经验
猜测的依赖：

| 分发包 | 当前环境版本 | 说明 |
| --- | ---: | --- |
| `einops` | `0.8.2` | 模型张量重排 |
| `huggingface-hub` | `0.36.2` | 本地/镜像模型加载 |
| `loguru` | `0.7.3` | 官方 runtime 日志 |
| `langcodes[data]` | `3.5.1` | 语言标识及语言数据 |
| `gradio` | `6.17.3` | 官方包声明的 `>=6.17,<7` 依赖；`6.17.0` 在当前镜像不可用 |
| `librosa` | `0.11.0` | 官方 runtime 音频读取/重采样 |
| `PyYAML` | `6.0.3` | 官方配置/双流 runtime 依赖 |
| `safetensors` | `0.8.0rc0` | 模型权重加载 |
| `torchdiffeq` | `0.2.5` | flow matching ODE 求解 |
| `tqdm` | `4.70.0` | 官方 runtime 进度/工具依赖 |
| `lingua-language-detector` | `2.1.1` | 文本语言识别 |
| `WeTextProcessing` | `1.2.0` | 中英文文本规范化 |
| `pynini` | `2.1.7` | `WeTextProcessing` 的有限状态文本处理依赖 |

`gradio-client`、`hf-gradio`、`orjson`、`httpx`、`pandas`、`scikit-learn`、
`numba`、`llvmlite`、`soxr`、`audioread` 等会由上述包的依赖解析带入；它们
不需要单独复制当前 Conda 环境的安装命令，最终以 `uv.lock` 为准。

### 4.4 PyTorch 和系统层依赖

当前环境中的 `nvidia-cublas-cu12`、`nvidia-cuda-runtime-cu12`、
`nvidia-cudnn-cu12`、`nvidia-cufft-cu12`、`nvidia-curand-cu12`、
`nvidia-cusolver-cu12`、`nvidia-cusparse-cu12`、`nvidia-nccl-cu12`、
`nvidia-nvjitlink-cu12`、`nvidia-nvtx-cu12`、`triton` 等属于 PyTorch CUDA
wheel 的解析/安装结果，不建议手工逐个写入 `pyproject.toml`。让与 `torch==2.8.0`
和 `torchaudio==2.8.0` 匹配的 cu128 wheel 自动带入，并在锁文件中固定结果。

部署机还必须满足：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Linux native 层需确认 `libsndfile` 可被 `soundfile` 使用；如果系统缺少它，
按发行版安装 `libsndfile1`，不要把 apt/conda 系统包伪装成 Python 依赖。

## 5. 推荐 `pyproject.toml`

下面是根据当前环境实际版本、官方 `dots.tts` 元数据和目标 Python 3.12.13
整理的首轮配置。它让普通包走清华镜像，让 PyTorch CUDA wheel 走阿里云
cu128 源；不把当前 Conda 中与该服务无关的评测、数据处理和其他模型包复制
进来。

```toml
[project]
name = "dots-tts-soar-service"
version = "0.1.0"
description = "Unitale dots.tts-soar voice cloning HTTP service"
requires-python = "==3.12.13"
dependencies = [
    "fastapi==0.141.1",
    "starlette==1.6.0",
    "uvicorn==0.52.1",
    "pydantic==2.12.5",
    "python-multipart==0.0.32",
    "dots-tts",
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "numpy==2.2.6",
    "soundfile==0.13.1",
    "einops==0.8.2",
    "huggingface-hub==0.36.2",
    "loguru==0.7.3",
    "langcodes[data]==3.5.1",
    "gradio==6.17.3",
    "librosa==0.11.0",
    "PyYAML==6.0.3",
    "safetensors==0.8.0rc0",
    "torchdiffeq==0.2.5",
    "tqdm==4.70.0",
    "lingua-language-detector==2.1.1",
    "WeTextProcessing==1.2.0",
    "pynini==2.1.7",
]

[tool.uv]
package = false
index-strategy = "first-index"
concurrent-downloads = 20
concurrent-installs = 8

[[tool.uv.index]]
name = "pypi-tuna"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

[[tool.uv.index]]
name = "pytorch-cu128-aliyun"
url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
format = "flat"
explicit = true

[tool.uv.sources]
dots-tts = { git = "https://github.com/rednote-hilab/dots.tts.git", rev = "4947d364baa5afb2daf1feb00a247b5f23f97878" }
torch = { index = "pytorch-cu128-aliyun" }
torchaudio = { index = "pytorch-cu128-aliyun" }
```

说明：

- `dots-tts` 的 Git 源是官方仓库，不是 `/home/muyi086/tts-depency` 仓库；commit
  必须保留，避免“最新源码”改变推理签名。
- `torch` 和 `torchaudio` 必须来自同一 CUDA 系列；不要让清华普通 PyPI 源
  随意选出 CPU wheel 或其他 CUDA 版本。
- 如果阿里 cu128 flat index 在目标机不可用，可把两个 PyTorch 源替换成已经
  验证可访问的官方 cu128 wheel index，再重新生成锁文件；不要在同一个项目
  中无约束混用 cu128、cu130 和 Conda torch。
- `uv.lock` 必须由目标机解析成功后提交；`.venv/`、模型、缓存、WAV 不提交。

## 6. 创建、安装和验证命令

### 6.1 创建项目

先在仓库根目录执行。`--no-workspace` 用于避免未来根目录 `pyproject.toml`
存在时意外加入根 workspace；`--bare` 后手工写入本计划的配置：

```bash
cd /home/muyi086/github/TTS-and-VoiceDesign
uv python install 3.12.13
uv init --bare --no-workspace --name dots_tts_soar \
  --python 3.12.13 dots_tts_soar
printf '3.12.13\n' > dots_tts_soar/.python-version
```

将第 5 节配置写入 `dots_tts_soar/pyproject.toml` 后执行：

```bash
uv lock --project dots_tts_soar
uv sync --project dots_tts_soar --locked
uv run --project dots_tts_soar python --version
```

期望输出 `Python 3.12.13`。如果 `pynini` 或 PyTorch wheel 在 Python 3.12
解析失败，应先解决对应 wheel/镜像问题，不能退回把旧 Python 3.10 环境目录
直接当成 uv 环境。

### 6.2 无模型 import smoke test

API 进程不能在导入时加载模型；worker 也应延迟到 `synthesize()` 才导入
`dots_tts.runtime`。迁移后执行：

```bash
cd /home/muyi086/github/TTS-and-VoiceDesign
uv run --project dots_tts_soar python -c \
  "import fastapi, pydantic, uvicorn, soundfile; print('api imports ok')"

uv run --project dots_tts_soar python -c \
  "from dots_tts.runtime import DotsTtsRuntime; print('dots runtime import ok')"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --project dots_tts_soar python -c \
  "import torch; from dots_tts.runtime import DotsTtsRuntime; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

这些命令只检查包导入，不加载 4.9G 权重；模型初始化必须放到后面的 GPU canary。

### 6.3 模型初始化和真实合成 canary

准备一份获得授权的单说话人参考音频及其准确逐字稿，执行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --project dots_tts_soar python -c \
  "from dots_tts.runtime import DotsTtsRuntime; r=DotsTtsRuntime.from_pretrained('$HOME/hf-mirror/rednote-hilab/dots.tts-soar', precision='bfloat16', max_generate_length=500); print('model_init ok', r.sample_rate)"
```

期望 `sample_rate` 为 `48000`。随后通过新 8308 API 发送与旧 API 相同的请求，
检查：

```bash
curl -X POST http://127.0.0.1:8308/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"今天晴暖转阴雨。","audio_path":"reference.wav","prompt_text":"这里填写参考音频逐字稿。"}' \
  -o dots_tts_soar_canary.wav

uv run --project dots_tts_soar python -c \
  "import soundfile as sf; i=sf.info('dots_tts_soar_canary.wav'); print(i.samplerate, i.channels, i.duration)"
```

验收条件是采样率 48000、通道数 1、文件非空，且日志没有 worker 残留进程或
CUDA 清理异常。应至少用一个短请求和一个触发文本分片的长请求分别验证。

## 7. `main.py`、`worker.py` 和 `start.sh` 的迁移顺序

### 7.1 复制/改造 API 和 worker

以当前 `api/dots_tts_soar_api.py` 和 `api/dots_tts_soar_worker.py` 为行为基线，
新项目实施以下调整：

1. 将 API 文件改名为 `main.py`，保留所有路由、Pydantic 字段、状态码、响应
   MIME、上传 hash、prompt sidecar、CORS、本机内部路由和健康 JSON。
2. 将 worker 改名为 `worker.py`，保留 JSON 输入、WAV 输出、文本分片、音频拼接、
   前导静音裁剪、`DotsTtsRuntime.from_pretrained()` 参数和 `finally` 中的
   `torch.cuda.empty_cache()` / `ipc_collect()`。
3. 新项目的 API manager 使用 `sys.executable` 启动同一 `.venv` 的 worker，
   不再调用 `conda run`；使用 `start_new_session=True`、超时和进程组清理。
4. 新项目继续使用和其他模型相同的 `GPU_LOCK_FILE`，不可为 dots 单独绕开锁。
5. 保留 `LOCAL_FILES_ONLY` 的离线语义；模型路径只接受环境变量和本地目录，
   不在服务启动时自动下载权重。
6. `/v1/health` 中把 `python`、`uv_project`、`worker_script`、模型关键文件、
   `torch` 和 `cuda` 的状态报告清楚；缺少模型时健康接口应返回可诊断的 false
   字段，而不是 import 崩溃。

推荐的新项目 worker 链路如下：

```text
uv run --project dots_tts_soar python dots_tts_soar/main.py
  └─ sys.executable dots_tts_soar/worker.py --input-json ... --output-wav ...
       └─ import dots_tts.runtime（只在 worker 子进程内）
```

### 7.2 `start.sh` 当前启动方式

```bash
export DOTS_TTS_SOAR_PROJECT_DIR="${DOTS_TTS_SOAR_PROJECT_DIR:-$PROJECT_DIR/dots_tts_soar}"
HOST="$DOTS_TTS_SOAR_HOST" PORT="$DOTS_TTS_SOAR_PORT" \
  setsid uv run --no-sync --project "$DOTS_TTS_SOAR_PROJECT_DIR" \
    python "$DOTS_TTS_SOAR_PROJECT_DIR/main.py" &
```

端口、变量名、PID cleanup 和共享 GPU 锁逻辑均已保持；旧 Conda 分支不再存在。

## 8. 测试计划

新项目完成后，应增加标准库 `unittest` 测试（当前仓库的默认测试命令不依赖
CUDA、模型或外部服务）：

1. `main.py` 可以导入，且不会导入 `dots_tts.runtime` 或初始化 CUDA。
2. `/v1/health` 在模型目录不存在、worker 文件存在/缺失、`nvidia-smi` 不可用
   等条件下仍返回 JSON 和诊断字段。
3. `/v1/upload_audio` 能保存音频、覆盖同名逻辑路径、返回大小和 SHA-256，并
   正确创建/删除 prompt sidecar。
4. `/v1/check/audio` 对存在和不存在文件返回兼容字段。
5. `DotsTtsSoarSynthesizeRequest` 保留字段约束、默认值和 `style_prompt` 拒绝行为。
6. worker runner 的 subprocess、超时、非零退出、空 WAV、临时文件清理、进程组
   终止和 GPU 锁释放均使用 mock 验证。
7. 使用 mock runtime 验证文本分片、片段停顿、mono 转换、WAV 写出和前导静音
   裁剪，不调用真实 CUDA。

建议命令：

```bash
uv run --project dots_tts_soar python -m unittest discover -s tests -v
uv run --project dots_tts_soar python -m compileall -q dots_tts_soar
uv lock --project dots_tts_soar --check
```

如果迁移测试放在仓库根目录的 `tests/`，则按当前仓库指南执行：

```bash
uv run --project dots_tts_soar python -m unittest discover -s tests -v
```

真实 GPU canary 与无模型单元测试分开执行，不能为了让默认测试通过而跳过
CUDA/模型验证。

## 9. 验收门槛和回滚

### 9.1 必须全部满足

- `uv sync --locked` 在无模型机器上成功，不下载权重。
- `.python-version` 和 `pyproject.toml` 都严格要求 `3.12.13`。
- `dots.tts` 来源固定为官方 Git commit `4947d364...`，不是未锁定的 latest。
- `torch`/`torchaudio` 同为 `2.8.0` 并使用同一 cu128 来源，`torch.cuda.is_available()`
  在 GPU 机器上为 true。
- API、worker 和 shared runtime 能在没有模型权重时完成导入/health。
- 8308 的 health、upload、check、synthesize 和 unload 路由及响应契约保持不变。
- 真实 canary 输出非空、48 kHz、单声道 WAV；长文本分片和 `prompt_text` sidecar
  都至少验证一次。
- worker 的成功、异常、超时和取消路径都不会残留进程，也不会永久持有 GPU 锁。
- `bash start.sh` 仍以独立 uv 项目启动 8308；旧 Conda 回退入口已删除。
- `uv.lock`、`pyproject.toml`、代码和测试可提交；`.venv`、权重、音频、缓存、
  本地官方源码副本和机器绝对路径不提交。

### 9.2 回滚方式

1. 当前不保留旧 Conda 回退；如需回滚，应先审阅 Git 历史并恢复完整的旧代码
   和环境方案，不通过启动参数隐式切换。
2. uv API 或 worker 失败时，保留现有日志、health 和测试输出，修复独立 uv
   项目本身；不重新把旧 `api` 入口接回启动链路。

## 10. 当前判断的边界

task29 的迁移验收已完成：目标 uv 项目、锁文件、无模型契约测试、8308 health
和 WebUI 兼容性均已验证；task30 进一步删除了旧 API 文件和 `dots_tts_soar`
Conda 环境。当前项目的 dots.tts-soar 运行闭包只包含 `dots_tts_soar/`、外置
模型权重和宿主机 CUDA，不再包含旧 `api` 入口或该 Conda 环境。
