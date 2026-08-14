# MOSS VoiceGenerator 迁移计划与依赖审计

> 实施状态：已完成独立 `moss_voiceGenerator/` uv 服务、8315 端口、一次性 worker、共享 GPU 锁、本地模型路径和 WebUI 端点迁移。旧 `api/` MOSS 入口暂以注释保留，待实际验证通过后删除。

> 评估对象：`moss-voiceGenerator` Conda 环境、`api/api.py` 中的 MOSS VoiceGenerator 路由和 `api/moss_voice_design_worker.py`
>
> 目标目录：`moss_voiceGenerator/`
>
> 评估日期：2026-08-14
>
> 结论：可以迁移到 Python 3.12.13 的 uv 项目，但“完全复刻”应理解为复刻 Python 依赖、模型加载、推理、WAV 响应、GPU 锁和 worker 生命周期；不能把 Conda 的系统库、NVIDIA 驱动、CUDA 驱动或模型权重复制到项目目录。

## 1. 结论和边界

本模型适合按模型建立独立的 `moss_voiceGenerator/` 目录并用 uv 管理。理由是：

- 原环境已经是 Python 3.12.13，与本次目标版本一致。
- 环境中的业务 Python 包主要来自 PyPI，只有 `moss-tts` 是 MOSS-TTS 源码的 editable 安装。
- 当前 API worker 只在请求进程中导入 `torch`、`transformers`、`torchaudio` 和音频处理依赖，没有依赖其他模型的 Python 包。
- MOSS-VoiceGenerator 使用 Transformers 的本地 remote code，模型权重和 MOSS Audio Tokenizer 可以继续从 `hf-mirror` 的本地目录读取。
- 当前显卡可以使用 PyTorch SDPA；`flash_attn` 在原环境中并未安装，因此它不是迁移前提。

推荐的运行边界如下：

```text
moss_voiceGenerator/.venv（uv，Python 3.12.13）
    ├── main.py                 HTTP 服务和兼容接口
    ├── worker.py               一请求一进程的 MOSS 推理
    ├── moss_voice_design_compat.py
    ├── pyproject.toml
    └── uv.lock

宿主机运行时
    ├── NVIDIA 驱动和可用 CUDA 能力
    ├── $HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator
    ├── $HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer（v1）
    ├── $RUNTIME_CACHE_DIR/
    └── /usr/bin/ffmpeg（仅供相关音频工具使用，当前 worker 直接写 WAV）
```

必须处理的端口约束：现有 `/v1/moss/design` 属于主 API `8300`，而不是独立的 MOSS 端口。新的 `main.py` 不能同时独占 `8300`，否则会破坏主 API 的其他路由。实际迁移采用独立 `8315`，WebUI 直接调用新端口；旧主 API 路由暂以注释保留：

```text
WebUI ── POST /v1/moss/design ──> moss_voiceGenerator :8315
                                  （uv main.py + uv worker.py）
```

新服务直接保留原 `/v1/moss/design` 路径、请求字段、原始 WAV 响应和错误语义；模型加载和推理逻辑全部位于 `moss_voiceGenerator/`。主 API `8300` 的旧 MOSS 路由已注释，避免两个进程同时提供同一模型入口。

## 2. 当前功能和必须保留的契约

### 2.1 路由、生命周期和响应

| 项目 | 当前值 | 迁移要求 |
| --- | --- | --- |
| 对外路由 | `POST /v1/moss/design` | 路径不变、大小写不变 |
| 当前对外端口 | `8300` | 主 API 继续使用；新服务使用内部可配置端口，默认建议 `8315` |
| 成功响应 | 原始 WAV 字节，`audio/wav` | 不改成 JSON，不返回永久文件路径 |
| 失败响应 | HTTP 500，带 worker 错误摘要 | 保留可诊断错误 |
| GPU 并发 | `GPU_LOCK_FILE` 文件锁 | 新服务和主 API 继续共享同一个锁文件 |
| worker 生命周期 | 每个请求启动一个 worker，结束后进程退出 | 不改成常驻模型 |
| 超时 | `MOSS_VOICEGENERATOR_REQUEST_TIMEOUT=900` 秒 | 默认保持 900 秒，可由环境变量覆盖 |
| 本地权重 | `LOCAL_FILES_ONLY=1` 时离线加载 | 保持 `HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE` 语义 |

### 2.2 请求字段

当前 `api/api.py` 的 `MossDesignRequest` 字段必须原样接受：

| 字段 | 类型 | 当前默认值 | 说明 |
| --- | --- | --- | --- |
| `voice_description` | `str` | 必填 | 音色、年龄、情绪、语速等自然语言描述 |
| `text` | `str` | `这是生成的参考音频预览。` | 待合成文本 |
| `save_as` | `str \| null` | `designed_voice.wav` | 兼容字段；当前 worker 不依赖它 |
| `max_chars_per_chunk` | `int \| null` | `0` | `0` 表示不分片 |
| `pause_ms` | `int \| null` | `250` | 分片之间的静音时长 |
| `max_new_tokens` | `int \| null` | `4096` | 单个分片的生成上限 |
| `audio_temperature` | `float \| null` | `1.5` | 官方推荐值 |
| `audio_top_p` | `float \| null` | `0.6` | 官方推荐值 |
| `audio_top_k` | `int \| null` | `50` | 官方推荐值 |
| `audio_repetition_penalty` | `float \| null` | `1.1` | 官方推荐值 |
| `dtype` | `str \| null` | `auto` | `auto` 在 CUDA 上解析为 `bfloat16` |
| `attn_implementation` | `str \| null` | `auto` | `auto` 优先可用的 FlashAttention，否则使用 `sdpa` |

worker 还必须继续执行以下行为：

1. 拒绝空的 `text` 和 `voice_description`，并保留当前文本清理行为。
2. 通过 `processor.build_user_message(text=..., instruction=...)` 构建 MOSS 对话。
3. 按现有标点和 `max_chars_per_chunk` 规则分片，分片后插入 `pause_ms` 静音。
4. 输出 24 kHz 单声道 WAV；多通道解码结果需要在写出前合并为单声道。
5. 保留当前的 MOSS 解码兼容补丁，避免上游 processor 将 break positions 当作 `torch.split` 的 split sizes。
6. 在成功、异常、超时路径都释放模型、processor、Python 垃圾和 CUDA cache。

## 3. 当前机器和模型资产证据

### 3.1 原 Conda 环境

本次核对命令：

```bash
conda run -n moss-voiceGenerator python --version
conda run -n moss-voiceGenerator python -m pip freeze --all | sort
conda run -n moss-voiceGenerator python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

结果：

```text
Python 3.12.13
torch 2.12.0+cu130
torch.version.cuda 13.0
torch.cuda.is_available() True
GPU NVIDIA GeForce RTX 4070 Ti SUPER
compute capability (8, 9)
transformers 5.12.0
torchaudio 2.11.0
flash_attn 未安装
```

Conda 原生层只有 Python 运行时、动态库和基础工具；业务包由 pip 安装。因此不应把 `conda env export` 的绝对路径或完整 Conda 目录当作 uv 输入。

### 3.2 模型和 tokenizer

```text
模型仓库：OpenMOSS-Team/MOSS-VoiceGenerator
本地目录：$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator
目录大小：约 4.0G

Tokenizer 仓库：OpenMOSS-Team/MOSS-Audio-Tokenizer
本地目录：$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer
目录大小：约 6.7G
```

模型配置确认了 `sampling_rate=24000`。原始 VoiceGenerator 需要 Audio Tokenizer **v1，24 kHz、单声道**。不能使用 `MOSS-Audio-Tokenizer-v2`，其配置是 48 kHz 双声道。

模型目录必须至少包含 `config.json`、`model.safetensors`、`processing_moss_tts.py`、`configuration_moss_tts.py`、`modeling_moss_tts.py` 和 tokenizer 文件。Tokenizer 目录必须包含 `config.json`、`configuration_moss_audio_tokenizer.py`、`modeling_moss_audio_tokenizer.py` 及完整 safetensors 分片和 index 文件。

下载仍使用 hf-mirror，不复制权重到 Git 仓库：

```bash
HF_ENDPOINT=https://hf-mirror.com \
hf download OpenMOSS-Team/MOSS-VoiceGenerator \
  --local-dir "$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator"

HF_ENDPOINT=https://hf-mirror.com \
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --local-dir "$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer"
```

## 4. 依赖审计

### 4.1 新服务建议的顶层依赖

以下配置以当前环境为第一版迁移基线，避免 uv 自动升级到另一套 Transformers 或 CUDA 组合：

```toml
[project]
name = "moss-voicegenerator-service"
version = "0.1.0"
description = "Unitale MOSS VoiceGenerator HTTP service"
requires-python = "==3.12.13"
dependencies = [
    "fastapi==0.136.3",
    "starlette==1.3.1",
    "uvicorn==0.49.0",
    "pydantic==2.13.4",
    "python-multipart==0.0.32",
    "torch==2.12.0",
    "torchaudio==2.11.0",
    "transformers==5.12.0",
    "accelerate==1.12.0",
    "numpy==2.1.0",
    "soundfile==0.14.0",
    "moss-tts",
]

[tool.uv]
package = false
index-strategy = "first-index"

[[tool.uv.index]]
name = "pypi-tuna"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
moss-tts = { path = "../../../tts-depency/MOSS-TTS", editable = true }
torch = { index = "pytorch-cu130" }
torchaudio = { index = "pytorch-cu130" }
```

普通 Python 包优先走清华源；PyTorch CUDA wheel 使用显式 cu130 索引。如果本机验证阿里 CUDA 镜像已经同步了相同版本，可以替换该索引 URL；不能在没有确认 wheel 的情况下改动 torch 版本。

手动安装备选命令：

```bash
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  fastapi==0.136.3 starlette==1.3.1 uvicorn==0.49.0 \
  pydantic==2.13.4 python-multipart==0.0.32 \
  transformers==5.12.0 accelerate==1.12.0 \
  numpy==2.1.0 soundfile==0.14.0

uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.12.0 torchaudio==2.11.0

test -f ../../../tts-depency/MOSS-TTS/pyproject.toml
uv sync
```

分步命令只是手动安装备选；提交时应以 `pyproject.toml` 和 `uv.lock` 为准，不提交 `.venv`。

### 4.2 MOSS-TTS 源码依赖

原环境中的 `moss-tts` 是以下本地仓库的 editable 安装：

```text
本地仓库：$TTS_DEPENDENCY_ROOT/MOSS-TTS
远程仓库：https://github.com/OpenMOSS/MOSS-TTS
当前 commit：58b20a0d5fcc6766658d50967a90a9d890009a46
```

该仓库的 `pyproject.toml` 默认依赖包括：

```text
safetensors==0.6.2
numpy==2.1.0
orjson==3.11.4
tqdm==4.67.1
PyYAML==6.0.3
einops==0.8.1
scipy==1.16.2
librosa==0.11.0
tiktoken==0.12.0
psutil
packaging
ninja
setuptools
wheel
gradio
```

虽然新的 HTTP wrapper 不直接使用 Gradio，但“复刻原环境”模式应保留上游默认依赖。MOSS remote code 直接需要 torch、torchaudio、transformers、tqdm；新 worker 还直接使用 numpy、soundfile。模型代码已经在模型目录中，不应复制进项目。

### 4.3 原环境的完整 pip 基线

以下清单来自 moss-voiceGenerator 环境的 pip freeze，用于迁移后的 uv.lock 对账，不建议把所有传递包逐个写成顶层依赖。

~~~text
moss-tts @ git+https://github.com/OpenMOSS/MOSS-TTS@58b20a0d5fcc6766658d50967a90a9d890009a46
Jinja2==3.1.6
MarkupSafe==3.0.3
PyYAML==6.0.3
Pygments==2.20.0
accelerate==1.12.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
audioread==3.1.0
brotli==1.2.0
certifi==2026.5.20
cffi==2.0.0
charset-normalizer==3.4.7
click==8.4.1
cuda-bindings==13.3.1
cuda-pathfinder==1.5.5
cuda-toolkit==13.0.2
decorator==5.3.1
einops==0.8.1
fastapi==0.136.3
filelock==3.29.4
flatbuffers==25.12.19
fsspec==2026.4.0
gradio==6.17.3
gradio_client==2.5.0
groovy==0.1.2
h11==0.16.0
hf-gradio==0.4.1
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.19.0
idna==3.18
joblib==1.5.3
lazy-loader==0.5
librosa==0.11.0
llvmlite==0.47.0
markdown-it-py==4.2.0
mdurl==0.1.2
mpmath==1.3.0
msgpack==1.2.0
narwhals==2.22.1
networkx==3.6.1
ninja==1.13.0
numba==0.65.1
numpy==2.1.0
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
onnxruntime==1.26.0
orjson==3.11.4
packaging==26.0
pandas==3.0.3
pillow==12.2.0
platformdirs==4.10.0
pooch==1.9.0
protobuf==7.35.1
psutil==7.2.2
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
pydub==0.25.1
python-dateutil==2.9.0.post0
python-multipart==0.0.32
pytz==2026.2
regex==2026.5.9
requests==2.34.2
rich==15.0.0
safehttpx==0.1.7
safetensors==0.6.2
scikit-learn==1.9.0
scipy==1.16.2
semantic-version==2.10.0
setuptools==81.0.0
shellingham==1.5.4
six==1.17.0
soundfile==0.14.0
sox==1.5.0
soxr==1.1.0
starlette==1.3.1
sympy==1.14.0
threadpoolctl==3.6.0
tiktoken==0.12.0
tokenizers==0.22.2
tomlkit==0.14.0
torch==2.12.0
torchaudio==2.11.0
tqdm==4.67.1
transformers==5.12.0
triton==3.7.0
typer==0.25.1
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.7.0
uvicorn==0.49.0
wheel==0.46.3
~~~

nvidia-*、cuda-bindings、cuda-toolkit、triton 是 torch 2.12.0 CUDA 轮子带来的运行依赖，通常由 uv 根据 torch wheel 自动解析。它们不是模型源码仓库，也不应手工从 Conda 目录拷贝。

### 4.4 Conda 原生层、系统依赖和不可迁移项

原环境的 Conda 原生层包括：Python 3.12.13、libgcc/libstdcxx、libzlib、openssl、sqlite、xz、bzip2、libffi、ncurses、readline、tk、X11 基础库、pip、setuptools 和 wheel。这些是 Conda 环境实现细节，不应写入新项目的业务依赖列表。

仍需由主机提供或单独确认：

- NVIDIA 驱动，并且驱动能够运行 CUDA 13.0 的 torch wheel。
- /usr/bin/ffmpeg；本机已存在 6.1.1。当前 worker 使用 soundfile 写 WAV，不以 ffmpeg 作为合成前提，但上游 Gradio/音频工具可能使用它。
- 编译工具链。只有安装 FlashAttention 或其他带原生扩展的可选包时才需要；普通迁移不应因为它们提前编译失败。
- 模型权重、Tokenizer 权重和 Hugging Face 缓存目录。

## 5. FlashAttention 评估

结论：**不是迁移必需依赖，第一版不要安装。**

证据如下：

- moss-voiceGenerator 环境中 flash_attn 不存在，原 worker 仍可选择 sdpa。
- NVIDIA GeForce RTX 4070 Ti SUPER 的 compute capability 是 8.9，硬件层面具备使用 FlashAttention 2 的条件。
- $TTS_DEPENDENCY_ROOT/flash-attention 当前 commit 为 c75d019dea9d910312974417bc28f190dfdda6d9，不是已经针对当前项目锁定的 wheel。
- 当前系统 nvcc 是 12.0，而当前 torch wheel 是 2.12.0+cu130；直接从源码编译存在 CUDA/ABI 不匹配风险。
- MOSS 官方示例本身将 FlashAttention 写为可选项，CUDA 无该包时使用 PyTorch SDPA。

实现阶段应保持显式策略：

~~~python
if flash_attn_available and cuda_capability_major >= 8:
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"
~~~

只有完成一次独立 canary 后才考虑安装：

~~~bash
cd "$REPOSITORY_ROOT/moss_voiceGenerator"
MAX_JOBS=4 uv pip install --python .venv/bin/python \
  --no-build-isolation "$TTS_DEPENDENCY_ROOT/flash-attention"
~~~

这条命令不属于默认安装步骤；若失败，必须回退到 sdpa，不能因此阻塞 MOSS VoiceGenerator 的基本迁移。

## 6. tts-depency 本地仓库审计

| 本地仓库 | 本次是否需要 | 用途和处理方式 |
| --- | --- | --- |
| MOSS-TTS | 需要，建议锁定 commit | 当前 Conda 环境的 moss-tts editable 来源；提交配置使用 git URL + commit，开发机可用本地路径验证 |
| flash-attention | 可选，不纳入默认安装 | 只用于后续性能实验；当前没有已验证的 cu130 wheel |
| vllm | 不需要 | 当前 worker 使用 Transformers AutoModel，不使用 vLLM |
| MOSS-Audio | 不需要 | MOSS Audio/4B Thinking 是另一条模型链，不是 VoiceGenerator 的 Audio Tokenizer |
| Step-Audio-EditX | 不需要 | 由主 API 的独立编辑 worker 使用 |
| LongCat-AudioDiT | 不需要 | LongCat 独立服务使用 |
| stable-audio-3 | 不需要 | Stable Audio 独立服务使用 |

不要将 MOSS-TTS、FlashAttention 或任何模型权重整个复制到 moss_voiceGenerator/。项目只保存依赖来源、commit、环境变量和验证命令。

## 7. 初始化和安装步骤

### 7.1 创建项目

~~~bash
cd "$REPOSITORY_ROOT"
uv python install 3.12.13
uv init --app --python 3.12.13 --no-readme moss_voiceGenerator
cd moss_voiceGenerator
uv python pin 3.12.13
~~~

将第 4.1 节配置写入 pyproject.toml 后执行：

~~~bash
uv lock
uv sync
uv run python -c "import fastapi, pydantic, uvicorn, torch, torchaudio, transformers; print(torch.__version__, torch.version.cuda)"
~~~

当前开发机直接使用仓库外的本地 MOSS-TTS 源码，避免 uv 访问 GitHub：

~~~bash
test -f "$TTS_DEPENDENCY_ROOT/MOSS-TTS/pyproject.toml"
uv lock
uv sync
~~~

该路径相对于 moss_voiceGenerator/ 解析为仓库外的 tts-depency/MOSS-TTS；它要求两个目录保持当前相对布局。部署到其他机器时，应设置相同目录布局，或把 tool.uv.sources.moss-tts 改为目标机器上的本地路径。

### 7.2 模型离线预检

在首次 GPU 推理前执行：

~~~bash
MODEL_DIR="$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator"
CODEC_DIR="$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer"

test -s "$MODEL_DIR/config.json"
test -s "$MODEL_DIR/model.safetensors"
test -s "$CODEC_DIR/config.json"
test -s "$CODEC_DIR/model.safetensors.index.json"
test -s "$CODEC_DIR/model-00001-of-00002.safetensors"
test -s "$CODEC_DIR/model-00002-of-00002.safetensors"
~~~

main.py 的 health 应报告模型目录、Tokenizer 配置/权重、worker 脚本、Python 解释器、CUDA 和 FlashAttention 状态；缺失权重时返回可诊断状态，不要在服务导入阶段加载模型。

## 8. 迁移实现步骤

1. 在 moss_voiceGenerator/ 中整理 MOSS 请求模型、环境变量解析、health、CORS 和 WAV 响应逻辑。
2. 将 api/moss_voice_design_worker.py 迁移为 uv worker。worker 必须由 sys.executable 启动，不能再调用 conda run。
3. 将 api/moss_voice_design_compat.py 的 codec 检查和 decode 修补一并迁移。补丁只作用于当前 processor 实例，不修改 hf-mirror 中的模型文件。
4. 用与 api/local_worker.py 等价的临时 JSON/WAV 文件和进程组清理机制，保留超时、非零退出、空 WAV、异常清理和 GPU 锁释放行为。
5. 新服务默认监听内部端口 8315；主 API 的 /v1/moss/design 通过 HTTP 转发到该服务，或在确认所有主 API 路由已完整合并后再改变部署结构。
6. start.sh 先增加显式开关和端口变量，再切换默认入口。旧的 Conda MOSS worker 必须保留到真实 GPU canary 和 WebUI 回归完成之后。
7. 成功迁移后再删除旧 worker 或 Conda 环境；删除动作不属于本计划的第一阶段。

建议环境变量：

~~~bash
export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export MOSS_VOICEGENERATOR_MODEL_DIR="${MOSS_VOICEGENERATOR_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator}"
export MOSS_AUDIO_TOKENIZER_PATH="${MOSS_AUDIO_TOKENIZER_PATH:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer}"
export MOSS_VOICEGENERATOR_PORT=8315
export MOSS_VOICEGENERATOR_REQUEST_TIMEOUT=900
export LOCAL_FILES_ONLY=1
~~~

## 9. 无模型测试和真实验证门槛

### 9.1 默认无模型测试

新增 tests/test_moss_voicegenerator_migration.py，使用标准库 unittest 和 mock，至少覆盖：

- main.py 可以在无模型环境导入，不导入 GPU 模型并不初始化 CUDA。
- health 路由的字段、模型路径状态、CUDA 状态和运行时解释器可诊断。
- /v1/moss/design 接受完整请求字段，mock worker 后返回 audio/wav。
- voice_description 或 text 为空时返回校验错误。
- worker 启动命令的第一个解释器是 moss_voiceGenerator/.venv/bin/python，不再出现 conda run。
- worker 超时、非零退出、空输出和异常时会清理 request JSON、output WAV 和进程组。
- 分片、暂停、单声道合并和 WAV 非空校验保持现有行为。
- GPU 锁在成功和异常路径均释放。

默认命令：

~~~bash
uv run --project moss_voiceGenerator python -m unittest discover -s tests -v
~~~

### 9.2 运行时验证

依次执行：

~~~bash
uv run --project moss_voiceGenerator python -c \
  "import torch; import transformers; print(torch.cuda.is_available(), torch.__version__, transformers.__version__)"

uv run --project moss_voiceGenerator python moss_voiceGenerator/main.py

curl http://127.0.0.1:8315/v1/health
~~~

模型 canary 必须使用本地权重和一个短文本，确认：

- AutoProcessor 能加载 MOSS VoiceGenerator 与 v1 codec。
- AutoModel 使用 bfloat16 和 sdpa 正常生成。
- 生成 WAV 采样率为 24000 Hz、单声道且文件非空。
- worker 退出后显存被释放，第二次请求可以再次启动。
- 8300 主 API 的历史路由和 /v1/moss/design 转发均仍可用。

## 10. 切换、回滚和完成标准

切换顺序：

~~~text
旧主 API 8300 + moss-voiceGenerator Conda worker
        │
        ├── 无模型契约测试
        ├── 8315 uv health
        ├── 本地模型 GPU canary
        └── WebUI /v1/moss/design WAV 回归
        ▼
8300 兼容转发 + 8315 uv MOSS 服务
        ▼
确认稳定后，才评估删除旧 Conda worker
~~~

完成迁移至少需要同时满足：

- uv sync 和 uv run 在 Python 3.12.13 下成功。
- uv.lock 固定了 Python 包、MOSS-TTS commit 和 CUDA torch 解析结果。
- 无模型 unittest 全部通过。
- health 不加载模型也能报告缺失资产和 CUDA 状态。
- GPU canary 生成 24 kHz 单声道有效 WAV。
- /v1/moss/design 的输入字段、端口、路径、WAV MIME、GPU 锁、超时和异常行为与旧实现兼容。
- start.sh 能在同一套端口约束下启动主 API 和新的 MOSS 服务。
- 未提交 .venv、Conda 环境、模型权重、缓存、上传音频、生成 WAV 或机器绝对路径。

出现模型加载、CUDA、FlashAttention 或 uv 解析问题时，回退到旧 Conda worker；不要为了让 uv 安装通过而降低 Transformers、torch、Tokenizer 或 API 契约版本。
