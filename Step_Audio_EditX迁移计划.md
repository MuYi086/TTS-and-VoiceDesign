# Step-Audio-EditX → Step_Audio_EditX 迁移计划与依赖审计

> 原始模型环境：`Step-Audio-EditX`
>
> 目标 uv 项目：`Step_Audio_EditX/`
>
> 评估日期：2026-08-14
>
> 评估依据：`项目升级评估.md`、当前 `api/step_audio_editx.py` 与
> `api/step_audio_editx_worker.py`、`start.sh`、`scoring-for-TTS` 中的
> `tts_local_Step_Audio_EditX.py` 和安装指南、实际 Conda 环境、Step-Audio-EditX
> 上游仓库。

## 1. 结论

可以按模型创建 `Step_Audio_EditX/`，用 Python `3.12.13` 的 uv 项目承载
Step-Audio-EditX 的 HTTP 控制面和一次性推理 worker。当前模型环境已经证明了这条
路线的技术基础：

- Conda 环境是 Python `3.12.13`，满足上游项目 `requires-python = ">=3.12,<3.14"`。
- 当前核心组合为 `torch 2.9.1+cu128`、`torchaudio 2.9.1+cu128`、
  `transformers 4.57.3` 和定制 `vllm 0.14.0rc2.dev125+gc826c72a9`。
- `pip check` 通过，CUDA 可用，ONNX Runtime 同时发现 CUDA、TensorRT 和 CPU
  provider。
- Step-Audio-EditX 上游仓库已经有 `pyproject.toml` 和 `uv.lock`，且当前锁文件可
  作为依赖解析参考。
- 当前推理真正需要的是一组明确的音频、FunASR、CosyVoice、Torch 和 vLLM 运行依赖，
  不需要把完整训练/Gradio/量化环境塞入新服务。

但“完全复刻 Conda 环境”不能理解为把 `conda env export` 原样转换成 uv：

1. NVIDIA 驱动、宿主机 CUDA 能力、系统动态库和模型权重不属于 uv 项目。
2. vLLM 是一个带原生扩展的定制预编译 wheel，不能改成普通 PyPI 的同名版本，
   也不能默认从 `/home/muyi086/tts-depency/vllm` 源码构建替代。
3. 上游 `pyproject.toml` 声明了不少训练和 Web Demo 依赖，但当前已验证的 Conda
   环境没有安装 `flash_attn`、`deepspeed`、`gradio`、`bitsandbytes` 或
   `llmcompressor`，推理仍能正常工作。
4. 上游现有 `uv.lock` 与 `pyproject.toml` 已发生漂移：锁文件根包仍是旧依赖集合，
   且锁文件要求 `transformers >=4.57.5`，当前已验证环境是 `4.57.3`。因此不能
   直接把上游锁文件复制后宣称迁移完成，必须根据实际版本重新生成并检查。

推荐结论是：**先用 uv 精确复刻已验证的推理栈，再按 import 和真实推理结果瘦身；
HTTP 服务和 worker 都迁移到新项目，模型、Tokenizer 和上游源码仍放在外部路径。**

## 2. 当前运行边界和迁移目标

### 2.1 当前 API 功能

当前主 API `8300` 中的 Step-Audio-EditX 路由为：

```text
POST /v1/upload_audio
GET  /v1/check/audio?file_name=...
POST /v1/step-audio-editx/edit
GET  /v1/health
```

编辑请求的稳定字段为：

| 字段 | 类型 | 约束和用途 |
| --- | --- | --- |
| `prompt_audio` | `string` | 必填；先通过 `/v1/upload_audio` 上传的音频逻辑路径 |
| `prompt_text` | `string \| null` | `emotion`、`style`、`paralinguistic`、`speed` 必须提供，且应与参考音频逐字对应 |
| `generated_text` | `string \| null` | 目标文本；非 `denoise`/`vad` 时省略则回退为 `prompt_text` |
| `edit_type` | `emotion \| style \| paralinguistic \| denoise \| vad \| speed` | 必填；保持现有 Pydantic Literal 范围 |
| `edit_info` | `string` | `emotion`、`style`、`speed` 必填；`paralinguistic` 使用目标文本中的官方标签 |

成功响应必须继续是原始 WAV 字节：

```text
HTTP 200
Content-Type: audio/wav
```

当前 worker 的运行约束也必须保留：

- 请求期间才加载模型；一请求一个 worker；结束后销毁进程释放显存。
- 所有模型共享 `GPU_LOCK_FILE`，不能因为拆成 uv 项目而取消串行化。
- `LOCAL_FILES_ONLY=1` 时设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。
- 保留 `STEP_AUDIO_EDITX_DTYPE`、`MAX_MODEL_LEN`、`GPU_MEMORY_UTILIZATION`、
  `MAX_NUM_SEQS`、`ENFORCE_EAGER`、`COSYVOICE_DTYPE` 和 `COSYVOICE_CUDA_GRAPH`
  等现有环境变量。
- `STEP_AUDIO_EDITX_CODE_PATH`、模型目录和 Tokenizer 目录继续可由环境变量覆盖。

### 2.2 推荐目标边界

```text
Step_Audio_EditX/.venv（uv，Python 3.12.13）
    ├── main.py       HTTP 服务、请求校验、health、WAV 响应
    ├── worker.py     一次性模型加载、编辑、WAV 输出
    ├── runtime.py    锁、超时、子进程和临时文件管理
    ├── pyproject.toml
    ├── uv.lock
    └── .python-version

宿主机外部资产
    ├── $HF_MIRROR_DIR/stepfun-ai/Step-Audio-EditX
    ├── $HF_MIRROR_DIR/stepfun-ai/Step-Audio-Tokenizer
    ├── $STEP_AUDIO_EDITX_CODE_PATH
    ├── NVIDIA 驱动和 CUDA 能力
    └── $RUNTIME_CACHE_DIR、$PROMPTS_DIR、$GPU_LOCK_FILE
```

`main.py` 不应在导入时加载 Torch、vLLM 或模型权重；只有 worker 子进程才导入
重型模型依赖。这样无模型机器可以运行 health 和契约测试，也能让主 API 继续负责
其他模型。

### 2.3 端口和 WebUI 兼容方案

WebUI 当前固定把 Step-Audio-EditX 请求发到主 API `8300`：

```text
POST http://127.0.0.1:8300/v1/step-audio-editx/edit
```

因此不能直接让新 `main.py` 抢占 `8300`。推荐采用两阶段切换：

1. 新服务先使用内部可配置端口，默认 `8316`，仅用于 health、mock 契约和本地真实
   canary。
2. `api/api.py` 的 `8300/v1/step-audio-editx/edit` 在迁移确认前保留为兼容入口，
   后续改成调用新 `Step_Audio_EditX` 服务的轻量代理；`/v1/upload_audio`、
   `/v1/check/audio` 和 `PROMPTS_DIR` 仍由主 API 作为共享资产入口维护。
3. `start.sh` 同时启动 `Step_Audio_EditX/main.py`，使用 `uv run --project`，主 API
   的外部端口和 WebUI 地址不变。
4. 代理和新服务都通过同一个 `PROMPTS_DIR`、`GPU_LOCK_FILE` 和离线变量工作，完成
   真实 canary 后再删除旧的 Conda worker 逻辑。

如果后续决定让 WebUI 直接访问独立端口，也必须同时修改 WebUI 的上传、哈希检查、
编辑请求和健康检查地址；这不是本次第一阶段推荐方案。

## 3. `/home/muyi086/tts-depency` 仓库审计

### 3.1 必需仓库

| 仓库 | 是否必需 | 用途 | 迁移处理 |
| --- | --- | --- | --- |
| `/home/muyi086/tts-depency/Step-Audio-EditX` | 是 | 官方 `tts.py`、`tokenizer.py`、`model_loader.py`、`stepvocoder/` 和 `funasr_detach/` | 继续外部引用，不复制进模型权重目录；固定 commit 并记录版本 |

当前仓库证据：

```text
remote: https://github.com/stepfun-ai/Step-Audio-EditX
branch: main
commit: a652e87052c109e26f616d60971376ff47a829d4
```

`stepvocoder/` 和 `funasr_detach/` 已经随该仓库提供，当前运行不需要再单独克隆
CosyVoice 或 FunASR 仓库。上游源代码中实际被当前推理路径使用的关键文件包括：

```text
tts.py
tokenizer.py
model_loader.py
utils.py
config/prompts.py
stepvocoder/cosyvoice2/cli/cosyvoice.py
stepvocoder/cosyvoice2/cli/frontend.py
stepvocoder/cosyvoice2/flow/
stepvocoder/cosyvoice2/hifigan/
stepvocoder/cosyvoice2/bigvgan/
stepvocoder/cosyvoice2/transformer/
funasr_detach/auto/
funasr_detach/models/
funasr_detach/utils/
```

### 3.2 可选仓库

| 仓库 | 是否为推理前提 | 说明 |
| --- | --- | --- |
| `/home/muyi086/tts-depency/vllm` | 否 | 当前环境使用 vLLM 定制预编译 wheel；只有 wheel 对当前 GPU/Python 不可用时才作为源码排查或构建候选 |
| `/home/muyi086/tts-depency/flash-attention` | 否 | 当前环境 `importlib.util.find_spec("flash_attn")` 为 `None`；worker 使用 `enforce_eager=1` 和 `VLLM_ATTENTION_BACKEND=TRITON_ATTN`，不应把它列为首轮依赖 |
| `MOSS-Audio`、`MOSS-TTS`、`LongCat-AudioDiT`、`stable-audio-3` | 否 | 与 Step-Audio-EditX 当前推理路径无依赖关系，不应为了“完整复刻”而加入新项目 |

### 3.3 代码版本策略

迁移时必须记录上游 commit，不要只依赖一个会变化的工作目录：

```bash
git -C "$STEP_AUDIO_EDITX_CODE_PATH" fetch --tags origin
git -C "$STEP_AUDIO_EDITX_CODE_PATH" rev-parse HEAD
git -C "$STEP_AUDIO_EDITX_CODE_PATH" status --short
```

在真实 canary 通过前不要自动 `git pull` 或切换上游版本。若要更新上游，必须重新
跑 import smoke、模型加载和每种编辑类型的 WAV 验证。

## 4. 模型和 Tokenizer 资产

权重继续使用本机 `hf-mirror`，不复制到 Git，也不写入新项目目录：

```text
Step-Audio-EditX 模型：
$HF_MIRROR_DIR/stepfun-ai/Step-Audio-EditX

Step-Audio-Tokenizer：
$HF_MIRROR_DIR/stepfun-ai/Step-Audio-Tokenizer
```

### 4.1 模型目录最低检查

当前模型目录至少应能找到：

```text
config.json
configuration.json
configuration_step1.py
modeling_step1.py
model.safetensors.index.json
model-00001.safetensors（以及 index 指向的全部分片）
tokenizer.model
tokenizer_config.json
CosyVoice-300M-25Hz/FLOW_VERSION
CosyVoice-300M-25Hz/campplus.onnx
CosyVoice-300M-25Hz/cosyvoice.yaml
CosyVoice-300M-25Hz/flow.pt
CosyVoice-300M-25Hz/hift.pt
CosyVoice-300M-25Hz/speech_tokenizer_v1.onnx
```

### 4.2 Tokenizer 目录最低检查

当前 Tokenizer 目录至少应能找到：

```text
linguistic_tokenizer.npy
speech_tokenizer_v1.onnx
dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online/
    am.mvn
    config.yaml
    configuration.json
    model.pt
    seg_dict
    tokens.json
    tokens.txt
```

`StepAudioTokenizer` 会通过 Tokenizer 目录加载 FunASR Paraformer 资源，并读取
`linguistic_tokenizer.npy` 与 `speech_tokenizer_v1.onnx`；只下载主模型而不下载这些
Tokenizer 资源，不能算环境准备完成。

### 4.3 模型下载

只在模型资产缺失时执行，且优先使用已有 `hf-mirror` 本地目录。下载命令示例：

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf download stepfun-ai/Step-Audio-EditX \
  --local-dir "$HF_MIRROR_DIR/stepfun-ai/Step-Audio-EditX"

hf download stepfun-ai/Step-Audio-Tokenizer \
  --local-dir "$HF_MIRROR_DIR/stepfun-ai/Step-Audio-Tokenizer"
```

AWQ 4-bit 模型 `stepfun-ai/Step-Audio-EditX-AWQ-4bit` 是显存不足时的可选资产，
不能与普通模型目录混用；第一阶段先复刻当前已经验证的普通模型。

## 5. 依赖审计

### 5.1 当前环境验证结果

```text
Python:       3.12.13
torch:        2.9.1+cu128
torchaudio:   2.9.1+cu128
torchvision:  0.24.1
triton:       3.5.1
transformers: 4.57.3
vllm:         0.14.0rc2.dev125+gc826c72a9
CUDA:         torch.cuda.is_available() == True
ONNX:         TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
pip check:    No broken requirements found.
flash_attn:   未安装
deepspeed:    未安装
gradio:       未安装
bitsandbytes: 未安装
llmcompressor:未安装
```

### 5.2 新服务首版直接依赖

下面是建议写入 `Step_Audio_EditX/pyproject.toml` 的首版运行依赖。版本优先对齐
当前已经通过 `pip check` 和 CUDA 检查的环境；生成 `uv.lock` 后以锁文件为准。

| Python 包 | 分发名 | 建议版本 | 当前推理路径用途 |
| --- | --- | ---: | --- |
| FastAPI | `fastapi` | `0.141.1` | 新 `main.py` HTTP 服务 |
| Uvicorn | `uvicorn` | `0.52.1` | 启动 HTTP 服务 |
| Pydantic | `pydantic` | `2.13.4` | 请求校验 |
| Python Multipart | `python-multipart` | `0.0.32` | 若新服务直接兼容上传接口 |
| PyTorch | `torch` | `2.9.1`，确认 `+cu128` | vLLM、Tokenizer、CosyVoice 推理 |
| TorchAudio | `torchaudio` | `2.9.1`，确认 `+cu128` | 输入音频和输出 WAV |
| TorchVision | `torchvision` | `0.24.1` | 与当前 Torch 组合保持一致；虽非当前 worker 直接 import，先作为兼容基线 |
| Triton | `triton` | `3.5.1` | Torch/vLLM CUDA kernel 运行时 |
| vLLM | URL wheel | `0.14.0rc2.dev125+gc826c72a9` | Step-Audio LLM 推理；必须使用定制 wheel |
| compressed-tensors | `compressed-tensors` | `0.13.0` | 当前 vLLM wheel 的依赖 |
| Transformers | `transformers` | `4.57.3` | `AutoTokenizer` 和模型配置加载 |
| NumPy | `numpy` | `2.2.6` | 音频、Tensor 和预处理 |
| SciPy | `scipy` | `1.18.0` | CosyVoice 的滤波/音频处理 |
| SoundFile | `soundfile` | `0.14.0` | WAV 读写 |
| Librosa | `librosa` | `0.11.0` | 音频重采样与预处理 |
| ONNX Runtime GPU | `onnxruntime-gpu` | `1.28.0` | Step Tokenizer 和 CosyVoice 的 ONNX 模块 |
| OpenAI Whisper | `openai-whisper` | `20250625` | `tokenizer.py` 导入的 `whisper` 模块 |
| FunASR | `funasr` | `1.4.0` | 上游声明的兼容依赖；当前核心实际使用仓库内 `funasr_detach`，迁移后可在验证后评估是否去除 |
| HyperPyYAML | `hyperpyyaml` | `1.2.3` | CosyVoice 配置加载 |
| einops | `einops` | `0.8.2` | CosyVoice flow decoder；上游 pyproject 漏列，不能漏装 |
| SentencePiece | `sentencepiece` | `0.2.2` | Tokenizer/Transformers 兼容 |
| Hugging Face Hub | `huggingface-hub` | `0.36.2` | 上游 source/本地加载兼容 |
| ModelScope | `modelscope` | `1.39.0` | `model_source` 兼容分支；本地模式不主动下载 |
| Safetensors | `safetensors` | `0.8.0` | 模型权重读取的传递/显式兼容依赖 |
| Tokenizers | `tokenizers` | `0.22.2` | Transformers 传递依赖，锁定以避免 ABI 漂移 |

新 `main.py` 若复用项目内的音频输出、路径和进程管理模块，不需要把整个主 API 的
所有依赖复制进该项目；`httpx` 只有在主 API 代理采用异步 HTTP 客户端时才加入。
优先使用标准库 `urllib` 或在主 API 已有依赖中复用，避免扩大模型项目依赖面。

### 5.3 上游声明但首轮推理不需要的依赖

上游 `pyproject.toml` 还声明了以下包：

```text
ffmpeg-python
gradio
hdbscan
pytorch-memlab
rotary-embedding-torch
sox
torch-complex
torchcodec
typer
whisper
accelerate
bitsandbytes
deepspeed
trl
llmcompressor
cuda-toolkit
datasets
conformer
diffusers
spaces
wandb
matplotlib
pillow
```

处理原则：

- `ffmpeg-python`、`hdbscan`、`pytorch-memlab`、`rotary-embedding-torch`、`sox`、
  `torch-complex`、`torchcodec` 是上游兼容/工具依赖；如果目标是第一版“依赖集
  合复刻”，可以先按当前上游版本安装，但必须用 import smoke 证明它们的必要性后
  再决定是否降为可选组。
- `gradio`、`typer`、`matplotlib`、`pillow` 主要服务官方 Demo 或命令行，不是
  新 HTTP wrapper 的首要依赖。
- `accelerate`、`bitsandbytes`、`deepspeed`、`trl`、`llmcompressor`、`datasets`、
  `conformer`、`diffusers`、`spaces`、`wandb` 属于训练、量化或 Demo 扩展；当前
  环境没有安装其中多数，不能因为上游声明就让 uv 改写已验证 Torch/vLLM 组合。
- `cuda-toolkit` 不应作为 uv 的顶层依赖。Torch cu128 wheel 已带相应用户态运行库，
  系统仍需具备兼容的 NVIDIA 驱动；宿主机 CUDA 驱动不能由 Python 包管理器替代。
- `openai-whisper` 已提供 `whisper` Python 模块，不要在没有实际冲突验证前同时安
  装名字相近但不同的 `whisper` 分发包。

如果以后需要官方 Gradio Demo 或训练功能，应单独建立 `demo`/`training` dependency
group，并重新做兼容性验证，不要把它们混入线上编辑服务的默认 `uv sync`。

### 5.4 推荐 `pyproject.toml` 方向

`Step_Audio_EditX/pyproject.toml` 应以“服务运行依赖”为主体，示意如下。实际版本
写入前要确认清华源或指定 PyTorch 源存在对应 Python 3.12 Linux x86_64 wheel：

```toml
[project]
name = "step-audio-editx-service"
version = "0.1.0"
description = "Unitale Step-Audio-EditX HTTP service"
requires-python = "==3.12.13"
dependencies = [
    "fastapi==0.141.1",
    "uvicorn==0.52.1",
    "pydantic==2.13.4",
    "python-multipart==0.0.32",
    "torch==2.9.1",
    "torchaudio==2.9.1",
    "torchvision==0.24.1",
    "triton==3.5.1",
    "compressed-tensors==0.13.0",
    "transformers==4.57.3",
    "numpy==2.2.6",
    "scipy==1.18.0",
    "soundfile==0.14.0",
    "librosa==0.11.0",
    "onnxruntime-gpu==1.28.0",
    "openai-whisper==20250625",
    "hyperpyyaml==1.2.3",
    "einops==0.8.2",
    "sentencepiece==0.2.2",
    "huggingface-hub==0.36.2",
    "modelscope==1.39.0",
    "safetensors==0.8.0",
    "tokenizers==0.22.2",
    "vllm",
]

[tool.uv]
package = false
index-strategy = "first-index"

[[tool.uv.index]]
name = "pypi-tuna"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
vllm = { url = "https://wheels.vllm.ai/c826c72a9633454679871fcb81fbc31fe03fb150/vllm-0.14.0rc2.dev125%2Bgc826c72a9-cp38-abi3-manylinux_2_31_x86_64.whl" }
```

说明：普通 Python 包默认走清华源；中科大和阿里源可作为替代；Torch CUDA wheel
使用显式 cu128 源以避免 uv 解析到 CPU 或错误 CUDA 变体。若本机确认阿里 CUDA
镜像包含完全相同的 wheel，可以把 `pytorch-cu128` URL 替换为对应阿里镜像，否则
不要为了“全镜像”而牺牲二进制匹配。

## 6. 环境创建和安装命令

### 6.1 创建项目

以下命令是迁移实施时使用的命令，不要在当前 Conda 环境上直接执行 `uv sync`，也
不要修改上游 `/home/muyi086/tts-depency/Step-Audio-EditX` 的 `pyproject.toml`。

```bash
cd /home/muyi086/github/TTS-and-VoiceDesign
uv init --python 3.12.13 Step_Audio_EditX
cd Step_Audio_EditX
uv python pin 3.12.13
```

然后将第 5.4 节的项目配置写入 `pyproject.toml`，再生成锁文件：

```bash
uv lock --refresh
uv sync --locked
uv run python --version
```

期望输出必须包含：

```text
Python 3.12.13
```

### 6.2 手动安装备选命令

如果 uv 在解析 vLLM 的定制 URL 或 PyTorch CUDA wheel 时需要分步处理，可使用下列
命令。最终提交仍应以 `pyproject.toml` 和 `uv.lock` 为准：

```bash
cd /home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX
uv venv --python 3.12.13

uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  fastapi==0.141.1 uvicorn==0.52.1 pydantic==2.13.4 \
  python-multipart==0.0.32 transformers==4.57.3 \
  numpy==2.2.6 scipy==1.18.0 soundfile==0.14.0 \
  librosa==0.11.0 onnxruntime-gpu==1.28.0 \
  openai-whisper==20250625 hyperpyyaml==1.2.3 einops==0.8.2 \
  sentencepiece==0.2.2 huggingface-hub==0.36.2 \
  modelscope==1.39.0 safetensors==0.8.0 tokenizers==0.22.2 \
  compressed-tensors==0.13.0 triton==3.5.1

uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1

uv pip install --python .venv/bin/python \
  "https://wheels.vllm.ai/c826c72a9633454679871fcb81fbc31fe03fb150/vllm-0.14.0rc2.dev125%2Bgc826c72a9-cp38-abi3-manylinux_2_31_x86_64.whl"

uv run python -m pip check
uv run python -c 'import torch, torchaudio, transformers, vllm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torchaudio.__version__, transformers.__version__, vllm.__version__)'
```

清华源不可用时，仅将普通包的 `--index-url` 替换为：

```text
https://pypi.mirrors.ustc.edu.cn/simple
https://mirrors.aliyun.com/pypi/simple
```

vLLM 定制 wheel 仍必须使用其固定 URL。安装后必须检查 `torch.version.cuda`、
`torch.cuda.is_available()` 和 vLLM 版本，不能仅以安装命令返回 0 判定成功。

### 6.3 复制/引用上游源码

新服务可以先通过环境变量引用外部源码，避免复制供应商代码：

```bash
export STEP_AUDIO_EDITX_CODE_PATH=/home/muyi086/tts-depency/Step-Audio-EditX
export STEP_AUDIO_EDITX_MODEL_DIR=/home/muyi086/hf-mirror/stepfun-ai/Step-Audio-EditX
export STEP_AUDIO_TOKENIZER_PATH=/home/muyi086/hf-mirror/stepfun-ai/Step-Audio-Tokenizer
```

`worker.py` 中将该路径加入 `sys.path` 后导入 `StepAudioTokenizer` 和 `StepAudioTTS`。
只有在部署目标不能访问外部依赖仓库时，才考虑把固定 commit 的运行时代码以明确的
vendor 目录复制进项目，并保留许可证、commit 和同步说明；不要把模型权重复制进来。

## 7. `main.py`、worker 和 `start.sh` 实施方案

### 7.1 `main.py`

参照已经成功迁移的 `qwen3_tts/main.py` 和 `moss_voiceGenerator/main.py`，新
`Step_Audio_EditX/main.py` 应负责：

1. 读取环境变量并进行路径展开，不在 import 阶段加载模型。
2. 提供 `GET /v1/health`，返回项目路径、worker 路径、模型/Tokenizer/源码是否存在、
   `sys.executable`、CUDA 状态、`worker_runtime = uv` 和最近错误。
3. 提供 `POST /v1/step-audio-editx/edit`，保持现有字段、Pydantic 校验、`404/500`
   错误语义和 `audio/wav` 响应。
4. 必要时提供 `POST /internal/unload_all`，保持和其他独立 uv 服务一致的内部运维接口。
5. 使用共享 GPU 锁；在 worker 完成、异常、超时和取消后都执行清理与短暂 CUDA 释放等待。

第一阶段不把 `StepAudioTokenizer`、`StepAudioTTS`、vLLM、Torch 或 ONNX Runtime
导入 `main.py`。health 只检查文件和轻量 CUDA 状态，不能因为缺模型而让 API import
直接失败。

### 7.2 `worker.py`

新 worker 需要从当前 `api/step_audio_editx_worker.py` 迁移并修正为 uv 解释器调用，
核心行为保持：

- 从 JSON 读取 `prompt_wav_path`、`prompt_text`、`generated_text`、`edit_type`、
  `edit_info`、模型路径、Tokenizer 路径和推理参数。
- 设置离线环境和 `VLLM_ATTENTION_BACKEND=TRITON_ATTN`。
- 从固定的外部源码路径导入 `StepAudioTokenizer`、`StepAudioTTS`。
- 以 `model_source="local"` 加载本地模型和 Tokenizer。
- 调用 `model.edit(...)`，将 Tensor 转为单声道/合法 WAV，并写入指定输出文件。
- `finally` 中删除模型引用、执行 `gc.collect()` 和 `torch.cuda.empty_cache()`。

新服务的 manager 应使用当前 uv 进程的 `sys.executable` 启动 worker，命令形态应为：

```text
<Step_Audio_EditX>/.venv/bin/python <Step_Audio_EditX>/worker.py \
  --input-json <temporary request> \
  --output-wav <temporary output>
```

命令中不能出现 `conda run`。worker 超时必须杀掉整个进程组并清理临时 JSON/WAV，
空 WAV、非零退出和缺少路径都要返回可诊断错误。

### 7.3 `start.sh`

迁移期间新增而不覆盖旧配置：

```bash
export STEP_AUDIO_EDITX_PROJECT_DIR="${STEP_AUDIO_EDITX_PROJECT_DIR:-$PROJECT_DIR/Step_Audio_EditX}"
export STEP_AUDIO_EDITX_HOST="${STEP_AUDIO_EDITX_HOST:-$HOST}"
export STEP_AUDIO_EDITX_PORT="${STEP_AUDIO_EDITX_PORT:-8316}"
export STEP_AUDIO_EDITX_MODEL_DIR="${STEP_AUDIO_EDITX_MODEL_DIR:-$HF_MIRROR_DIR/stepfun-ai/Step-Audio-EditX}"
export STEP_AUDIO_TOKENIZER_PATH="${STEP_AUDIO_TOKENIZER_PATH:-$HF_MIRROR_DIR/stepfun-ai/Step-Audio-Tokenizer}"
export STEP_AUDIO_EDITX_CODE_PATH="${STEP_AUDIO_EDITX_CODE_PATH:-$HOME/tts-depency/Step-Audio-EditX}"
```

canary 阶段启动：

```bash
HOST="$STEP_AUDIO_EDITX_HOST" PORT="$STEP_AUDIO_EDITX_PORT" \
  setsid uv run --project "$STEP_AUDIO_EDITX_PROJECT_DIR" \
  python "$STEP_AUDIO_EDITX_PROJECT_DIR/main.py" &
```

完成代理和 WebUI 回归前，保留 `STEP_AUDIO_EDITX_RUNTIME=conda` 的回退分支；确认新
服务稳定后再删除 `STEP_AUDIO_EDITX_CONDA_ENV`、`api/step_audio_editx.py` 和旧 worker。
删除 Conda 环境必须是最后一步，且要在真实推理 canary 和完整回归都通过后另行确认。

## 8. scoring-for-TTS 对接

当前评测脚本：

```text
/home/muyi086/github/scoring-for-TTS/modelScript/tts_local_Step_Audio_EditX.py
```

它直接导入上游源码中的 `tokenizer.py` 和 `tts.py`，并支持：

```text
clone / emotion / style / vad / denoise / paralinguistic / speed
```

第一阶段不要把 `scoring-for-TTS` 的所有评测依赖合并到 `Step_Audio_EditX` 服务。
建议评测脚本通过环境变量或一个小型公共 runner 调用新项目的同一套 worker/运行时，
确保评测和 HTTP 服务使用相同的：

- 上游 commit；
- Torch、Transformers、vLLM 版本；
- 模型和 Tokenizer 路径；
- `dtype`、`max_model_len`、显存利用率和 CosyVoice 参数。

对接完成后必须分别验证：

1. `scoring-for-TTS` 的直接脚本调用仍能输出 24 kHz、单声道 WAV。
2. 新 HTTP 服务编辑结果与直接脚本使用相同 prompt 音频、文本和 edit 参数时，输出
   的采样率、通道数和可播放性一致；随机采样不要求字节级相同。
3. 评测仓库的测试/报告路径不依赖新服务项目中的模型权重或缓存。

## 9. 测试和验收门槛

### 9.1 无模型测试

新项目至少增加以下标准库 `unittest` 测试：

- `main.py` 可以在没有权重、没有 CUDA 的环境 import。
- `/v1/health` 的字段、路径检查和 `worker_runtime=uv` 正确。
- 请求模型保留所有字段、默认值、编辑类型和必填校验。
- `/v1/step-audio-editx/edit` 在 mock worker 返回 WAV 时仍返回 `audio/wav`。
- worker 命令的第一个参数是新项目 `.venv` 的 `sys.executable`，不包含 `conda`。
- worker 成功、非零退出、超时、空 WAV、缺路径时都清理临时文件并释放进程组。
- 主 API `8300` 的兼容代理保留 WebUI 使用的上传、检查和编辑路径。

### 9.2 环境 smoke test

```bash
cd /home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX
uv run python --version
uv run python -m pip check
uv run python -c 'import torch, torchaudio, transformers, vllm, onnxruntime, whisper; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torchaudio.__version__, transformers.__version__, vllm.__version__); print(onnxruntime.get_available_providers())'
uv run python -m unittest discover -s ../tests -v
```

预期至少满足：

```text
Python 3.12.13
torch.version.cuda == "12.8"
torch.cuda.is_available() == True
pip check 无 broken requirements
```

### 9.3 本地模型 canary

无模型测试通过后，使用上游示例音频或经授权的本地参考音频执行：

```bash
export LOCAL_FILES_ONLY=1
export STEP_AUDIO_EDITX_CODE_PATH=/home/muyi086/tts-depency/Step-Audio-EditX
export STEP_AUDIO_EDITX_MODEL_DIR=/home/muyi086/hf-mirror/stepfun-ai/Step-Audio-EditX
export STEP_AUDIO_TOKENIZER_PATH=/home/muyi086/hf-mirror/stepfun-ai/Step-Audio-Tokenizer

curl http://127.0.0.1:8316/v1/health
```

至少逐项验证：

| 用例 | 关键检查 |
| --- | --- |
| `emotion` | `prompt_text` 和 `generated_text` 正常；`edit_info` 非空；输出 WAV 可播放 |
| `style` | `edit_info` 非空；参考文本不被丢弃 |
| `paralinguistic` | 目标文本中的 `[sigh]`、`[inhale]`、`[laugh]` 等官方标签原样传递 |
| `denoise` | 可以省略 `prompt_text` 与 `generated_text` |
| `vad` | 可以省略 `prompt_text` 与 `generated_text` |
| `speed` | `edit_info` 非空；输出采样率仍为 24 kHz |
| 连续编辑 | 第二次使用第一次生成结果作为新的 prompt，不覆盖原始台词资产 |

输出检查命令：

```bash
ffprobe -v error -show_entries stream=codec_name,sample_rate,channels \
  -of default=noprint_wrappers=1 edited.wav
```

应为 WAV、`sample_rate=24000`、`channels=1`。真实 canary 还要记录显存峰值、耗时、
worker 是否残留和共享 GPU 锁是否在异常后释放。

## 10. 回滚和删除顺序

迁移顺序必须是：

```text
旧 Conda worker
    ↓ 新 uv 项目 health/mock
旧 Conda worker + 新 uv 服务 canary
    ↓ 主 API 代理和 WebUI 回归
新 uv 服务默认路径，旧 Conda 作为显式 fallback
    ↓ 再次完成真实 GPU 回归
删除旧 api/worker 和 Conda 环境
```

禁止在以下证据出现前删除 `Step-Audio-EditX` Conda 环境：

- 新 uv 项目的 `uv sync --locked` 成功；
- `pip check`、Torch/CUDA/vLLM import smoke 成功；
- `emotion`、`style`、`paralinguistic`、`denoise`、`vad`、`speed` canary 成功；
- WebUI 的上传、连续编辑、独立保存/试听/导入导出回归成功；
- `scoring-for-TTS` 直接脚本和报告流程回归成功；
- 失败、超时、取消路径确认 GPU 锁和 worker 进程没有残留。

如新环境失败，优先只回退 Step 服务的启动命令和代理目标，不回滚其他已完成的
Qwen、MOSS 或主 API 迁移。

## 11. 当前 Conda 环境完整 pip 基线

以下清单来自 2026-08-14 对 `Step-Audio-EditX` 环境执行的
`python -m pip list --format=freeze | sort`，用于迁移后对账。它包含大量 vLLM
传递依赖，**不是建议逐项写入 `pyproject.toml` 的顶层依赖**。

```text
HyperPyYAML==1.2.3
Jinja2==3.1.6
MarkupSafe==3.0.3
PyJWT==2.13.0
PyYAML==6.0.3
Pygments==2.20.0
RapidFuzz==3.14.5
accelerate==1.10.1
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
aliyun-python-sdk-core==2.16.0
aliyun-python-sdk-kms==2.16.5
annotated-doc==0.0.5
annotated-types==0.8.0
anthropic==0.120.2
antlr4-python3-runtime==4.9.3
anyio==4.14.2
apache-tvm-ffi==0.1.13.post0
astor==0.8.1
attrs==26.1.0
audioread==3.1.0
blake3==1.0.9
cachetools==7.1.7
calmsize==0.1.3
cbor2==6.1.4
certifi==2026.7.22
cffi==2.1.0
charset-normalizer==3.4.9
click==8.4.2
cloudpickle==3.1.2
compressed-tensors==0.13.0
crcmod==1.7
cryptography==50.0.0
cuda-bindings==13.3.1
cuda-core==1.0.1
cuda-pathfinder==1.6.0
cuda-python==13.3.1
cupy-cuda12x==14.1.1
decorator==5.3.1
depyf==0.20.0
detect-installer==0.1.0
dill==0.4.1
diskcache==5.6.3
distro==1.9.0
dnspython==2.8.0
docstring_parser==0.18.0
einops==0.8.2
email-validator==2.3.0
fastapi-cli==0.0.32
fastapi-cloud-cli==0.23.0
fastapi==0.141.1
fastar==0.11.0
ffmpeg-python==0.2.0
filelock==3.32.2
flashinfer-python==0.5.3
flatbuffers==25.12.19
frozenlist==1.8.0
fsspec==2026.7.0
funasr==1.4.0
future==1.0.0
gguf==0.19.0
grpcio-reflection==1.83.0
grpcio==1.83.0
h11==0.16.0
hdbscan==0.8.44
hf-xet==1.5.2
httpcore2==2.9.1
httpcore==1.0.9
httptools==0.8.0
httpx2==2.9.1
httpx==0.28.1
huggingface_hub==0.36.2
hydra-core==1.3.4
idna==3.18
ijson==3.5.1
interegular==0.3.3
jaconv==0.5.0
jamo==0.4.1
jieba==0.42.1
jiter==0.16.0
jmespath==0.10.0
joblib==1.5.3
jsonschema-specifications==2025.9.1
jsonschema==4.26.0
kaldiio==2.18.1
lark==1.2.2
lazy-loader==0.5
librosa==0.11.0
llguidance==1.3.0
llvmlite==0.44.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown-it-py==4.2.0
mcp-types==2.0.0
mcp==2.0.0
mdurl==0.1.2
mistral_common==1.11.7
model-hosting-container-standards==0.1.16
modelscope-hub==0.2.0
modelscope==1.39.0
more-itertools==11.1.0
mpmath==1.3.0
msgpack==1.2.1
msgspec==0.21.1
multidict==6.7.1
narwhals==2.24.0
networkx==3.6.1
ninja==1.13.0
numba==0.61.2
numpy==2.2.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cudnn-frontend==1.26.0
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-cutlass-dsl-libs-base==4.5.3
nvidia-cutlass-dsl==4.5.3
nvidia-ml-py==13.610.43
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.3.20
nvidia-nvtx-cu12==12.8.90
omegaconf==2.3.1
onnxruntime-gpu==1.28.0
openai-harmony==0.0.8
openai-whisper==20250625
openai==2.52.0
opencv-python-headless==5.0.0.93
opentelemetry-api==1.44.0
oss2==2.19.1
outlines_core==0.2.11
packaging==26.0
pandas==3.0.5
partial-json-parser==0.2.1.1.post7
pillow==12.3.0
pip==26.2
platformdirs==4.11.0
pooch==1.9.0
prometheus-fastapi-instrumentator==8.1.0
prometheus_client==0.26.0
propcache==0.5.2
protobuf==7.35.1
psutil==7.2.2
py-cpuinfo==9.0.0
pybase64==1.4.3
pycountry==26.2.16
pycparser==3.0
pycryptodome==3.23.0
pydantic-extra-types==2.11.1
pydantic-settings==2.14.2
pydantic==2.13.4
pydantic_core==2.46.4
pynndescent==0.6.0
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
python-json-logger==4.1.0
python-multipart==0.0.32
pytorch-memlab==0.3.2
pyzmq==27.1.0
ray==2.56.1
referencing==0.37.0
regex==2026.7.19
requests==2.34.2
rich-toolkit==0.20.3
rich==15.0.0
rignore==0.8.0
rotary-embedding-torch==0.9.1
rpds-py==2026.6.3
ruamel.yaml.clib==0.2.15
ruamel.yaml==0.18.17
safetensors==0.8.0
scikit-learn==1.9.0
scipy==1.18.0
sentencepiece==0.2.2
sentry-sdk==2.66.1
setproctitle==1.3.7
setuptools==80.10.2
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
soundfile==0.14.0
sox==1.5.0
soxr==1.1.0
sse-starlette==3.4.6
starlette==1.3.1
supervisor==4.3.0
sympy==1.14.0
tabulate==0.10.0
tensorboardX==2.6.5
threadpoolctl==3.6.0
tiktoken==0.13.0
tokenizers==0.22.2
torch-complex==0.4.4
torch==2.9.1
torchaudio==2.9.1
torchcodec==0.9.1
torchvision==0.24.1
tqdm==4.70.0
transformers==4.57.3
triton==3.5.1
truststore==0.10.4
typer==0.27.0
typing-inspection==0.4.2
typing_extensions==4.16.0
umap-learn==0.5.12
urllib3==2.7.0
uvicorn==0.52.1
uvloop==0.22.1
vllm==0.14.0rc2.dev125+gc826c72a9
watchfiles==1.2.0
websockets==17.0.1
wheel==0.47.0
xgrammar==0.1.29
yarl==1.24.5
```

> 注：该附录只作为当前机器的时间点基线。迁移时应以新 `uv.lock` 的可复现解析
> 结果和真实 import/推理验证为准；不要手工把传递依赖逐条复制成顶层依赖。
