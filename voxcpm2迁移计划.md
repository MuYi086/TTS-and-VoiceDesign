# VoxCPM2 迁移计划

> 评估日期：2026-08-15
> 原运行环境：`voxcpm2` Conda，Python `3.10.20`
> 目标环境：`voxcpm2/` uv 项目，Python `3.12.13`
> 当前服务端口：`8306`
> 实施状态：已完成 uv 默认切换；旧 API/Conda 回退按迁移保留要求暂不删除。

## 1. 结论

可以按模型建立 `voxcpm2/` 目录，并使用独立的 uv 项目承载 VoxCPM2 的 API、worker 和推理依赖。推荐采用“一个模型一个 uv 项目”的方式，但“完全复刻 Conda 环境”应理解为复刻可运行的软件依赖和 CUDA 组合，不是把 Conda 的 Python、`libgcc`、`sqlite` 等系统包逐项搬进 `pyproject.toml`。

迁移可行性为：**已通过 Python 3.12.13 的真实 import、CUDA、clone 和 VoiceDesign 本地权重短推理 canary。** 原 Conda 环境是 Python 3.10.20；`voxcpm==2.0.3` 的包元数据声明 `Requires-Python: >=3.10`，官方包说明额外要求 `<3.13`，因此 3.12.13 落在目标兼容区间内。

本次评估没有发现 VoxCPM2 运行必须使用的 `/home/muyi086/tts-depency` 本地仓库，也没有发现必须安装 `flash-attn` 的证据。当前 `voxcpm` 源码使用 PyTorch 原生 `scaled_dot_product_attention`；`optimize` 参数对应 `torch.compile`，不是 FlashAttention 开关。

## 2. 当前运行链和必须保留的行为

### 2.1 当前启动链

当前 `start.sh` 默认使用以下命令启动专用 VoxCPM2 服务；显式设置
`VOXCPM2_RUNTIME=conda` 时才回退到原命令：

```bash
HOST="$VOXCPM2_HOST" PORT="$VOXCPM2_PORT" \
  setsid uv run --no-sync --project "$VOXCPM2_PROJECT_DIR" \
  python "$VOXCPM2_PROJECT_DIR/main.py" &
```

`api/voxcpm2_api.py` 是轻量 HTTP wrapper，模型只在请求时由 worker 子进程加载。worker 通过共享 `GPU_LOCK_FILE` 串行化，完成、异常、超时和取消后都必须退出并释放显存。迁移后只替换 worker/API 的 Python 运行时，不改变这个生命周期边界。

### 2.2 8306 专用 API 契约

以下路径、字段和响应类型必须保持不变：

| 方法 | 路径 | 迁移要求 |
| --- | --- | --- |
| `GET` | `/v1/health` | 保留 `code`、`paths`、`available`、`cuda`、`runtime` 等诊断字段；缺少模型时返回诊断 JSON，不应导入崩溃 |
| `POST` | `/v1/upload_audio` | 保留文件名哈希、SHA-256、大小和同名文件覆盖行为；`prompt_text` sidecar 继续保存 |
| `GET` | `/v1/check/audio` | 保留 `file_name` 查询参数以及 `exists`、`sha256`、`size_bytes`、`has_prompt_text` 等字段 |
| `POST` | `/v2/synthesize` | 保留 `text`、`audio_path`、`prompt_text`、`backend`、`clone_mode`、`control_instruction`、`nonverbal_tags` 和既有兼容字段 |
| `POST` | `/internal/unload_all` | 保留本地内部调用的退出行为 |

合成成功响应必须仍是 `audio/wav`，而不是 JSON。`clone_mode="ultimate"` 使用准确的 `prompt_text`；`clone_mode="controllable"` 只使用 `control_instruction` 和可选的一个 `nonverbal_tags`，二者不能合并。后端最终传给模型的文本格式仍为：

```text
(control_instruction)[tag]正文
```

其中 `control_instruction` 请求字段不带外层括号，参考音频转写不能被打印到 worker 的最终目标文本日志中。

### 2.3 8300 音色设计兼容边界

VoxCPM2 音色设计当前不在 8306 专用 wrapper 内，而是由主 API 的 `api/voxcpm2_voice_design.py`、`api/voxcpm2_voice_design_worker.py` 提供，并由 `api/api.py` 暴露：

```text
POST http://127.0.0.1:8300/v1/voxcpm2/design
```

`TTS-Studio-WebUI` 当前仍访问 8300，而不是直接访问 8306。迁移实现时应把音色设计 worker 也放入 `voxcpm2/`，同时在 8300 保留兼容代理或等价转发；不能因为专用服务迁移而把 WebUI 的音色设计 URL 改成 8306，也不能删除 `/v1/voice-design/providers` 中的 `voxcpm2` 项。

## 3. 可行性证据

| 检查项 | 当前证据 | 判断 |
| --- | --- | --- |
| uv/Python | uv `0.12.3` 可用，uv 管理的 `cpython-3.12.13-linux-x86_64-gnu` 已存在 | 满足目标环境前提 |
| VoxCPM 包 | `voxcpm==2.0.3`，官方包声明 Python `>=3.10`，README 要求 `<3.13` | 3.12.13 值得迁移，但需要 canary |
| 当前 Conda import | `import voxcpm, torch, torchaudio, torchcodec, numpy, soundfile` 成功 | 当前环境功能链完整 |
| CUDA | 当前 `torch==2.12.0+cu130`，`torch.cuda.is_available()` 为 `True`；GPU 为 RTX 4070 Ti SUPER，驱动 `610.47` | CUDA 运行面可复用 |
| 本地权重 | `/home/muyi086/hf-mirror/openbmb/VoxCPM2` 存在，约 `4.7G`，含 `config.json`、`model.safetensors`、`audiovae.pth` 和 tokenizer 文件 | 无需下载或复制权重 |
| uv 解析 | 使用 Python 3.12.13 对目标依赖做 dry-run，解析出 160 个包；使用 PyTorch cu130 专用索引后 `torch==2.12.0`、`torchaudio==2.11.0`、`torchcodec==0.14.0` 可解析 | 可生成 `uv.lock` |
| FlashAttention | `voxcpm` 代码没有 `flash_attn` 导入，注意力实现为 `torch.nn.functional.scaled_dot_product_attention` | 不作为迁移依赖 |

注意：清华镜像在本机的一次 dry-run 出现 TLS handshake EOF；因此配置中首选清华并保留中科大/阿里 Python 镜像的替换方法。PyTorch cu130 专用 wheel 已在阿里索引中核对到 Python 3.12 x86_64 的 torch、torchaudio、torchcodec 文件。

## 4. 依赖清单

### 4.1 目标项目应直接声明的依赖

这些是 `voxcpm2/main.py`、worker 和 API 直接使用或必须固定的顶层依赖。版本先以当前 `voxcpm2` Conda 环境和本项目已验证的 uv 项目为迁移基线，不能在切换前随意升级。

| 包 | 当前基线版本 | 用途 |
| --- | ---: | --- |
| `voxcpm` | `2.0.3` | VoxCPM2 官方推理包 |
| `torch` | `2.12.0+cu130` | CUDA 推理、`torch.compile` 和原生 attention |
| `torchaudio` | `2.11.0+cu130` | 参考音频读取、重采样和模型音频处理 |
| `torchcodec` | `0.14.0+cu130` | `voxcpm` 的声明运行依赖 |
| `numpy` | `2.2.6` | waveform、拼接、随机种子和数组处理 |
| `soundfile` | `0.14.0` | WAV 读写；API 返回前的音频校验依赖它的 Python 绑定 |
| `fastapi` | `0.136.3` | HTTP API |
| `starlette` | `1.3.1` | `BaseHTTPMiddleware` 等直接导入 |
| `uvicorn` | `0.49.0` | 服务启动 |
| `pydantic` | `2.13.4` | 请求模型和 `model_validator` |
| `python-multipart` | `0.0.32` | `UploadFile` / multipart 上传 |

`voxcpm` 会声明并拉取其余运行依赖；不要使用 `pip install voxcpm --no-deps`，否则会得到能 import 包名但不能正常加载模型的假环境。

### 4.2 `voxcpm==2.0.3` 声明的运行依赖

以下依赖来自当前 Conda 环境中 `voxcpm-2.0.3.dist-info/METADATA` 的 `Requires-Dist`，是模型环境应保留的完整直接依赖集合。表中的版本是当前环境观察值，最终精确版本由 `uv.lock` 固化。

| 包 | 当前版本 | 说明 |
| --- | ---: | --- |
| `torch` | `2.12.0+cu130` | CUDA 版 PyTorch |
| `torchaudio` | `2.11.0+cu130` | 音频输入处理 |
| `torchcodec` | `0.14.0+cu130` | 官方声明依赖 |
| `transformers` | `5.12.0` | tokenizer 与模型配置 |
| `einops` | `0.8.2` | VoxCPM 张量变换 |
| `gradio` | `6.18.0` | 官方包声明的运行依赖；当前 HTTP 服务不直接启动 Gradio |
| `inflect` | `7.5.0` | 文本规范化依赖 |
| `addict` | `2.4.0` | 模型/配置辅助依赖 |
| `wetext` | `0.1.4` | 文本规范化 |
| `modelscope` | `1.37.1` | 官方 denoiser 及 ModelScope 兼容路径 |
| `datasets` | `3.6.0` | 官方包声明依赖；训练路径不在本服务范围内 |
| `huggingface-hub` | `1.19.0` | 本地模型快照和 Hugging Face 兼容路径 |
| `pydantic` | `2.13.4` | 模型配置和 HTTP 请求模型 |
| `tqdm` | `4.68.2` | 模型加载/推理进度 |
| `simplejson` | `4.1.1` | 官方包声明依赖 |
| `sortedcontainers` | `2.4.0` | 官方包声明依赖 |
| `soundfile` | `0.14.0` | WAV 读写 |
| `librosa` | `0.11.0` | VoxCPM2 音频特征和静音处理 |
| `matplotlib` | `3.10.9` | 官方包声明依赖 |
| `funasr` | `1.3.9` | 官方包声明依赖；本项目默认不单独启动 FunASR 服务 |
| `spaces` | `0.50.4` | 官方包声明依赖 |
| `argbind` | `0.3.9` | 官方包配置/训练代码依赖 |
| `safetensors` | `0.8.0` | 模型权重加载 |

其中 `gradio`、`spaces`、`datasets`、`funasr`、`argbind` 和 denoiser 相关包不是当前 HTTP worker 的直接 import，但它们属于官方 wheel 的声明运行依赖，第一次使用 uv 建环境时应由解析器安装，不能凭“当前请求暂时没走到”而手工删掉。

### 4.3 传递依赖和 CUDA 运行库

`uv.lock` 应锁定下列类型的传递依赖，不能把它们手工拼到主 API 项目：

- PyTorch cu130：`cuda-toolkit==13.0.2`、`cuda-bindings`、`triton==3.7.0`、`nvidia-cublas`、`nvidia-cudnn-cu13`、`nvidia-cuda-*`、`nvidia-cufft`、`nvidia-curand`、`nvidia-cusolver`、`nvidia-cusparse`、`nvidia-nccl-cu13`、`nvidia-nvtx`、`nvidia-nvjitlink`、`nvidia-nvshmem-cu13` 等。
- 音频/数值：`cffi`、`scipy`、`numba`、`llvmlite`、`soxr`、`audioread`、`pooch`、`scikit-learn`、`joblib`、`threadpoolctl`、`pandas`、`pillow`、`pyarrow`。
- Hugging Face/HTTP：`huggingface-hub` 的 `httpx`、`httpcore`、`requests`、`fsspec`、`filelock`、`tokenizers`、`regex`、`safetensors`、`hf-xet` 等。
- FunASR/文本/ModelScope：`hydra-core`、`omegaconf`、`kaldiio`、`kaldifst`、`jieba`、`jamo`、`jaconv`、`umap-learn`、`editdistance`、`torch-complex`、`tensorboardX`、`oss2`、`aliyun-python-sdk-core`、`aliyun-python-sdk-kms`、`contractions`、`textsearch` 等。
- 服务端：`anyio`、`h11`、`click`、`annotated-types`、`annotated-doc`、`pydantic-core`、`typing-extensions`、`typing-inspection` 等。

传递依赖不应全部复制成手工安装命令；正确的验收方式是新项目执行 `uv lock` / `uv sync` 后检查 `uv.lock`，再执行 `python -m pip check` 和 import smoke test。

### 4.4 当前 Conda 环境的完整 pip 快照

下面是评估时 `conda run -n voxcpm2 python -m pip list --format=freeze` 的完整快照，用于发现遗漏和后续 diff。它是**现状审计记录**，不是建议把所有版本逐条写进 `pyproject.toml`；例如 `pip`、`setuptools`、`wheel` 和系统库由 uv/操作系统管理，CUDA 轮子由 PyTorch 专用索引管理。

```text
Jinja2==3.1.6
MarkupSafe==3.0.3
PyYAML==6.0.3
Pygments==2.20.0
addict==2.4.0
aiohappyeyeballs==2.6.2
aiohttp==3.14.1
aiosignal==1.4.0
aliyun-python-sdk-core==2.16.0
aliyun-python-sdk-kms==2.16.5
annotated-doc==0.0.4
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyascii==0.3.3
anyio==4.13.0
argbind==0.3.9
async-timeout==5.0.1
attrs==26.1.0
audioread==3.1.0
brotli==1.2.0
certifi==2026.5.20
cffi==2.0.0
charset-normalizer==3.4.7
click==8.4.1
contourpy==1.3.2
contractions==0.1.73
crcmod==1.7
cryptography==49.0.0
cuda-bindings==13.3.1
cuda-pathfinder==1.5.5
cuda-toolkit==13.0.2
cycler==0.12.1
datasets==3.6.0
decorator==5.3.1
dill==0.3.8
docstring_parser==0.18.0
editdistance==0.8.1
einops==0.8.2
exceptiongroup==1.3.1
fastapi==0.136.3
filelock==3.29.4
fonttools==4.63.0
frozenlist==1.8.0
fsspec==2025.3.0
funasr==1.3.9
gradio==6.18.0
gradio_client==2.5.0
groovy==0.1.2
h11==0.16.0
hf-gradio==0.4.1
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.19.0
hydra-core==1.3.3
idna==3.18
inflect==7.5.0
jaconv==0.5.0
jamo==0.4.1
jieba==0.42.1
jmespath==0.10.0
joblib==1.5.3
kaldifst==1.8.0
kaldiio==2.18.1
kiwisolver==1.5.0
lazy-loader==0.5
librosa==0.11.0
llvmlite==0.47.0
markdown-it-py==4.2.0
matplotlib==3.10.9
mdurl==0.1.2
modelscope==1.37.1
more-itertools==11.1.0
mpmath==1.3.0
msgpack==1.2.0
multidict==6.7.1
multiprocess==0.70.16
networkx==3.4.2
numba==0.65.1
numpy==2.2.6
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
omegaconf==2.3.1
orjson==3.11.9
oss2==2.19.1
packaging==26.0
pandas==2.3.3
pillow==12.2.0
pip==26.1.1
platformdirs==4.10.0
pooch==1.9.0
propcache==0.5.2
protobuf==7.35.1
pyahocorasick==2.3.1
pyarrow==24.0.0
pycparser==3.0
pycryptodome==3.23.0
pydantic==2.13.4
pydantic_core==2.46.4
pydub==0.25.1
pynndescent==0.6.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
python-multipart==0.0.32
pytz==2026.2
regex==2026.5.9
requests==2.34.2
rich==15.0.0
safehttpx==0.1.7
safetensors==0.8.0
scikit-learn==1.7.2
scipy==1.15.3
semantic-version==2.10.0
sentencepiece==0.2.1
setuptools==81.0.0
shellingham==1.5.4
simplejson==4.1.1
six==1.17.0
sortedcontainers==2.4.0
soundfile==0.14.0
soxr==1.1.0
spaces==0.50.4
starlette==1.3.1
sympy==1.14.0
tensorboardX==2.6.5
textsearch==0.0.24
threadpoolctl==3.6.0
tiktoken==0.13.0
tokenizers==0.22.2
tomlkit==0.14.0
torch-complex==0.4.4
torch==2.12.0
torchaudio==2.11.0
torchcodec==0.14.0
tqdm==4.68.2
transformers==5.12.0
triton==3.7.0
typeguard==4.5.2
typer==0.25.1
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2026.2
umap-learn==0.5.12
urllib3==2.7.0
uvicorn==0.49.0
voxcpm==2.0.3
wetext==0.1.4
wheel==0.46.3
xxhash==3.7.0
yarl==1.24.2
```

## 5. 推荐的 uv 配置

### 5.1 创建项目

在仓库根目录执行：

```bash
uv init --bare --python 3.12.13 --name voxcpm2 --no-workspace voxcpm2
cd voxcpm2
uv python pin 3.12.13
```

`voxcpm2` 是应用项目，不需要构建可发布的 Python wheel；`pyproject.toml` 的 `package = false` 可以避免 uv 把服务脚本当成可发布包。

### 5.2 建议的 `pyproject.toml` 核心内容

以下配置可作为新目录的初始内容。`torchcodec` 也显式绑定到 cu130 索引，避免从普通 PyPI 解析到 CPU/非 CUDA 轮子。

```toml
[project]
name = "voxcpm2-service"
version = "0.1.0"
description = "Unitale VoxCPM2 HTTP service"
requires-python = "==3.12.13"
dependencies = [
    "fastapi==0.136.3",
    "numpy==2.2.6",
    "pydantic==2.13.4",
    "python-multipart==0.0.32",
    "soundfile==0.14.0",
    "starlette==1.3.1",
    "torch==2.12.0",
    "torchaudio==2.11.0",
    "torchcodec==0.14.0",
    "uvicorn==0.49.0",
    "voxcpm==2.0.3",
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
url = "https://mirrors.aliyun.com/pytorch-wheels/cu130/"
format = "flat"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchaudio = { index = "pytorch-cu130" }
torchcodec = { index = "pytorch-cu130" }
```

如果清华源在当前机器继续出现 TLS 问题，只替换 `pypi-tuna` 的 URL，不要把 PyTorch 源和普通 Python 源混成一个无优先级的 `--extra-index-url`：

```text
中科大：https://pypi.mirrors.ustc.edu.cn/simple
阿里云：https://mirrors.aliyun.com/pypi/simple
```

生成锁文件和环境：

```bash
uv lock
uv sync
uv run python -m pip check
```

### 5.3 不建议的安装方式

不建议直接执行下面这种不带来源和版本的命令：

```bash
pip install -U voxcpm torch torchaudio
```

它可能解析到 CPU PyTorch、不同 CUDA 主版本或未经过本项目验证的最新 Transformers/Librosa，破坏当前环境的可重复性。也不要把 `flash-attn`、`vllm` 或其他模型仓库直接加入这个项目来“增强性能”。

## 6. 模型、缓存和外部资产

迁移后继续使用本地 Hugging Face mirror，不在项目内复制权重：

```bash
export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export VOXCPM2_MODEL_DIR="${VOXCPM2_MODEL_DIR:-$HF_MIRROR_DIR/openbmb/VoxCPM2}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
export HF_HOME="${HF_HOME:-$HF_MIRROR_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
```

当前模型目录已经存在且约 4.7G。默认 `VOXCPM2_DENOISE=0`、`VOXCPM2_LOAD_DENOISER=0` 时不需要下载 ZipEnhancer 模型；如果未来启用 `denoise`，需要额外准备 ModelScope 的 `iic/speech_zipenhancer_ans_multiloss_16k_base`，并单独验证离线缓存，不应把它误认为 VoxCPM2 主权重。

运行缓存、上传音频、生成 WAV 和 lock 文件仍按当前项目规则处理：权重、`api/prompts/` 运行数据、`api/tempAudio/`、`.cache/` 和 `.venv/` 不提交 Git。

## 7. `/home/muyi086/tts-depency` 仓库核查

当前目录下有 LongCat、MOSS、Step-Audio-EditX、Stable Audio、vLLM 和 FlashAttention 等仓库，但没有 VoxCPM2 上游源码仓库。对本次迁移的结论如下：

| 本地仓库 | 是否加入 `voxcpm2` | 原因 |
| --- | --- | --- |
| `flash-attention` | 否 | `voxcpm` 没有导入 `flash_attn`；模型 attention 使用 PyTorch SDPA；源码构建会引入 CUDA/PyTorch/编译器耦合 |
| `vllm` | 否 | 官方说明中的高吞吐/vLLM-Omni 是另一套服务架构，不是当前一请求一 worker API 的运行依赖 |
| `LongCat-AudioDiT` | 否 | 另一个模型的 worker 和模型代码 |
| `MOSS-Audio` / `MOSS-TTS` | 否 | 音频模型和声效模型依赖，和 VoxCPM2 无 import 关系 |
| `Step-Audio-EditX` | 否 | 编辑模型独立服务 |
| `stable-audio-3` | 否 | Stable Audio 独立服务 |

因此本次不需要 `uv add --editable /home/muyi086/tts-depency/...`，也不需要复制任何上述仓库。若以后要测试 FlashAttention 性能，必须在迁移完成后另开实验分支，先做 `torch`/CUDA/GPU 架构兼容 canary，再决定是否添加额外依赖。

## 8. 代码迁移边界

以下功能已迁移到 `voxcpm2/`，同时保留 `api/` 原文件和 Conda 回退路径：

```text
voxcpm2/
├── main.py                         # 8306 HTTP wrapper
├── worker.py                       # 克隆一请求一 worker
├── voice_design_worker.py          # 无参考音频的 VoiceDesign worker
├── voxcpm2_helpers.py              # 延迟导入 voxcpm/torch 的 helper
├── audio_trim.py                   # 音频裁剪
├── audio_output.py                # WAV 持久化/校验
├── gpu_runtime.py                 # 进程组清理和 CUDA 状态
├── synthesis_request.py           # 兼容请求模型
├── pyproject.toml
├── uv.lock
└── .python-version
```

`main.py` 不应在导入阶段加载 `voxcpm`、`torch` 或模型权重；重型依赖只在 worker 中延迟导入。新 worker 应继续通过 JSON 临时文件接收参数、输出临时 WAV，确保异常清理和 GPU 锁行为与旧实现一致。

主 API 的 8300 兼容层暂时保留：它可以把 `/v1/voxcpm2/design` 转发给新的 uv 服务，或者通过显式的 uv worker 调度器调用 `voxcpm2/voice_design_worker.py`。在 WebUI 回归完成前，不删除 `api/voxcpm2_api.py`、`api/voxcpm2_worker.py`、`api/voxcpm2_voice_design.py` 和 `api/voxcpm2_voice_design_worker.py`。

## 9. `start.sh` 切换方案

迁移期间先新增项目路径和开关，保留可回退的 Conda 分支：

```bash
export VOXCPM2_PROJECT_DIR="${VOXCPM2_PROJECT_DIR:-$PROJECT_DIR/voxcpm2}"
export VOXCPM2_RUNTIME="${VOXCPM2_RUNTIME:-uv}"
export VOXCPM2_CONDA_ENV="${VOXCPM2_CONDA_ENV:-voxcpm2}"
```

离线契约测试和本地 GPU canary 已通过，`start.sh` 默认使用：

```bash
HOST="$VOXCPM2_HOST" PORT="$VOXCPM2_PORT" \
  setsid uv run --no-sync --project "$VOXCPM2_PROJECT_DIR" \
  python "$VOXCPM2_PROJECT_DIR/main.py" &
```

默认端口必须仍为 `8306`。`bash start.sh` 仍应可用；如果 uv 服务失败，`VOXCPM2_RUNTIME=conda bash start.sh` 应能恢复旧路径。确认完全迁移后再删除 `VOXCPM2_CONDA_ENV` 分支和旧 API 文件，删除动作由后续任务单独执行。

## 10. 验收顺序

### 10.1 无模型/无 CUDA 的契约测试

- `uv run --project voxcpm2 python -c "import fastapi, pydantic, uvicorn"` 成功。
- `main.py` 导入不加载 `voxcpm`、`torch` 或权重。
- `/v1/health` 在模型目录缺失时仍返回 JSON 诊断。
- `/v1/upload_audio` 和 `/v1/check/audio` 的 SHA-256、大小、sidecar 和同名覆盖行为与旧实现一致。
- `/v2/synthesize` 对 `ultimate` / `controllable` 的字段互斥校验、非语言标签白名单和错误状态不变。
- worker 成功、非零退出、超时、空 WAV、取消和异常时，临时文件、子进程组和 GPU 锁均能清理。
- 8300 `/v1/voxcpm2/design` 兼容代理和 `/v1/voice-design/providers` 仍存在。

### 10.2 Python 3.12.13 运行面 canary（已完成）

在已经准备好的本地权重上执行，不允许联网下载：

```bash
uv run --project voxcpm2 python - <<'PY'
import sys
import torch
import torchaudio
import torchcodec
import voxcpm

print(sys.version)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torchaudio.__version__)
print(torchcodec.__version__)
print(voxcpm.__file__)
PY
```

本次已使用本地权重和短参考 WAV 完成 clone canary，并完成无参考 VoiceDesign canary：

- `voxcpm==2.0.3`、`torch==2.12.0+cu130`、CUDA 和 RTX 4070 Ti SUPER import 成功。
- `clone_mode="ultimate"` + 准确 `prompt_text` 返回 `audio/wav`，48 kHz 单声道，worker 约 19 秒完成并清理。
- `/v1/voxcpm2/design` 返回 `audio/wav`，最终模型文本为 `(温柔、清晰、自然的成年女性声音)你好。`，worker 约 13 秒完成并清理。
- `/home/muyi086/tts-depency/flash-attention` 存在，但 `voxcpm` 使用原生 SDPA；`flash-attn` 不加入 uv 依赖。

后续 WebUI 回归仍可按下面的场景继续验证：

1. 旧兼容路径（不指定 `clone_mode`）合成一次。
2. `clone_mode="ultimate"` + 准确 `prompt_text` 合成一次。
3. `clone_mode="controllable"` + `control_instruction` 合成一次。
4. 一个 `nonverbal_tags` 合成一次，并检查 worker 打印的最终文本为 `(指令)[tag]正文`。
5. `/v1/voxcpm2/design` 无参考音频生成一次，确认 8300 兼容路径返回 `audio/wav`。

每次都核对 WAV 可读、采样率、非空、响应 MIME、显存释放和 WebUI 对应的 IndexedDB/工程资产逻辑没有变化。只有这些结果全部通过，才允许把 `start.sh` 的默认 VoxCPM2 runtime 切换为 uv。

## 11. 回滚与删除条件

切换前保留：

- `api/voxcpm2_api.py` 及其 worker/helper；
- `api/voxcpm2_voice_design.py` 及其 worker；
- `VOXCPM2_RUNTIME=conda` 和 `VOXCPM2_CONDA_ENV=voxcpm2`；
- 8300 的兼容音色设计路由和 8306 的原端口。

只有在 uv 运行路径完成上述离线契约、Python 3.12.13 import、GPU canary 和 WebUI 回归后，才可以在后续任务中删除旧 `api/` VoxCPM2 逻辑，并在确认不再需要 Conda 回退后删除 `voxcpm2` Conda 环境。删除前应先保存环境版本快照和迁移提交，不能以“uv 能启动”作为唯一删除依据。
