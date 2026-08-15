# Stable Audio 3 Medium -> stable_audio_3_medium uv 迁移计划

> 原始模型/环境：`stable_audio_3_medium`
>
> 目标项目目录：`TTS-and-VoiceDesign/stable_audio_3_medium/`
>
> 目标 Python：`3.12.13`
>
> 评估日期：`2026-08-15`
>
> 依据：`task34.md`、`项目升级评估.md`、`api/stable_audio_3_medium_api.py`、
> `api/stable_audio_3_medium_worker.py`、`api/local_worker.py`、
> `api/gpu_runtime.py`、`api/audio_output.py`、`start.sh`、
> `qwen3_tts/pyproject.toml`、实际 Conda 环境、
> `/home/muyi086/tts-depency/stable-audio-3` 和
> `/home/muyi086/tts-depency/flash-attention`。

## 1. 结论

可以按模型建立 `stable_audio_3_medium/`，使用 `uv init` 创建 Python
`3.12.13` 项目，迁移 Stable Audio 3 Medium 的 HTTP wrapper 和一次性
worker，并在验证完成后让 `start.sh` 用该项目提供原来的 8313 服务。

但不能把当前 Conda 环境做成“逐包、逐版本、逐文件的字面复制”。当前
环境包含 Python 运行时、Conda 的系统库、Torch 的 CUDA 传递依赖、测试工具
和一次性临时路径；其中 `torch`、`torchaudio` 和 `flash-attn` 还是针对
Python 3.10 编译的二进制包。合理的“完全复刻”应定义为：

1. 用 `uv` 锁定 Python 3.12.13、HTTP 依赖、模型直接依赖和 Torch/CUDA
   Python wheels。
2. 通过目标 Python 3.12 的 FlashAttention 2 wheel 或本机源码编译，不能
   复用当前 cp310 wheel。
3. 继续从 `/home/muyi086/tts-depency/stable-audio-3` 加载官方源码，或
   使用固定 commit 的等价源码依赖；不把本机绝对路径写入 `pyproject.toml`
   或 `uv.lock`。
4. 继续从 `hf-mirror` 读取 Stable Audio 3 Medium 权重；`uv sync` 不下载
   模型权重。
5. 保留一请求一 worker、GPU 锁、超时/异常时的进程组清理、WAV 响应和
   WebUI 现有字段。迁移验证完成前保留 `api/` 下旧 wrapper 作为回退。

因此本迁移可行。进一步检查当前官方源码后确认：FlashAttention 是 Medium
的推荐高性能路径，但不是这份 `stable-audio-3` 源码的硬性 import 依赖；
源码在缺少它时会回退到 flex-attention/分块 SDPA。新 uv worker 默认允许
该回退，并通过 `STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN=1` 恢复严格模式。
无论选择哪种模式，都不能把“uv 能完成依赖解析”当作真实迁移成功，仍需
完成目标环境 import 和 GPU canary。

## 2. 当前实现和必须保留的服务契约

当前 API 进程本身不加载 Stable Audio 模型，而是把请求写成 JSON，调用
`stable_audio_3_medium` Conda 环境中的 `api/stable_audio_3_medium_worker.py`，
读取 worker 生成的 WAV 后返回。模型 worker 每次请求结束后退出，以释放显存。

| 方法 | 路由 | 兼容要求 |
| --- | --- | --- |
| `GET` | `/v1/health` | 保留 `code`、`paths`、`available`、`cuda`、`runtime`、`last_errors`；缺少权重或源码时返回可诊断状态 |
| `POST` | `/v1/generate` | JSON 请求；成功返回 `audio/wav` 原始字节 |
| `POST` | `/v2/synthesize` | `/v1/generate` 的兼容别名；不能删除 |
| `POST` | `/internal/unload_all` | 仅允许本机调用；无常驻模型时仍返回成功 JSON |

请求字段和当前默认值：

| 字段 | 当前行为 |
| --- | --- |
| `prompt` | 必填，去除首尾空白后不能为空，长度不超过 2000 |
| `seconds` | 兼容 WebUI 的字段，`(0, 380]`；缺省为 7 秒 |
| `duration` | 官方 Stable Audio 拼法；可替代 `seconds`，两者同时传入时必须相同 |
| `steps` | `1..100`，缺省 8 |
| `cfg_scale` | `0..100`，缺省 1.0 |
| `seed` | 缺省 -1 |
| `device` | 只能是 `cuda` |
| `dtype` | 只能是 `float16` |

不能在迁移中改变以下外部行为：

- 服务端口默认仍为 `8313`，环境变量覆盖名保持为
  `STABLE_AUDIO_3_MEDIUM_*`。
- `/v1/generate` 和 `/v2/synthesize` 都返回 WAV，而不是 JSON 文件路径。
- CORS、`LOCAL_FILES_ONLY`、`GPU_LOCK_FILE`、请求超时和输出目录行为保持。
- `STABLE_AUDIO_3_MEDIUM_MODEL_DIR` 继续默认指向
  `$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium`。
- WebUI 的 `stable-audio-3-medium` 客户端继续向
  `http://127.0.0.1:8313/v1/generate` 发送 `{prompt, seconds}`，并要求
  `audio/*` 响应；该模型使用 `dialogue.sfx_plan.prompt_en`，不能把中文
  `prompt` 静默当成英文提示词。

## 3. 现有环境基线

### 3.1 Conda 和运行验证

本机实际环境：

~~~text
Conda environment: stable_audio_3_medium
Python:            3.10.20
torch:             2.7.1+cu126
torchaudio:        2.7.1+cu126
flash-attn:        2.6.3
CUDA available:    True
GPU:               NVIDIA GeForce RTX 4070 Ti SUPER
compute capability:8.9
VRAM:              16376 MiB
pip check:         No broken requirements found
~~~

当前环境的直接 import smoke test 已通过：

~~~python
import torch
import torchaudio
import soundfile
import flash_attn
from flash_attn import flash_attn_func
from stable_audio_3 import StableAudioModel
~~~

这只能证明现有 Python 3.10 环境可用，不能证明 Python 3.12、FlashAttention
和新的 uv 项目已经可用。迁移完成前必须在目标 `.venv` 重新执行同一组导入和
真实 GPU canary。

### 3.2 当前 Conda 包的来源问题

当前环境中有两个不能直接搬进提交物的临时来源：

- `stable-audio-3==0.1.0` 的 `direct_url.json` 指向
  `/tmp/tmp.CByse80swx/stable-audio-3`。
- `torch` 和 `torchaudio` 的 `direct_url.json` 指向 `/tmp` 下的 cp310
  wheel。
- `flash-attn` 来自 GitHub 预编译 wheel，文件名为
  `flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl`。

这些路径和 wheel 不能复制到目标 `uv.lock`。目标项目应使用 Python 3.12
对应的公开 wheel、固定 Git commit 或显式的外部本地源码安装。

## 4. 依赖清单

### 4.1 目标项目的 HTTP 控制面依赖

当前 Stable Audio wrapper 实际 import：FastAPI、Uvicorn、Pydantic 和
Starlette。原 Conda 环境没有这些控制面包，因为旧 wrapper 是由
`qwen3_tts/.venv` 启动的；迁移后 `stable_audio_3_medium/main.py` 自己
提供 8313 服务，所以目标 uv 项目必须声明它们：

~~~text
fastapi==0.136.3
starlette==1.3.1
uvicorn==0.49.0
pydantic==2.13.4
~~~

这些版本与已成功迁移的 `qwen3_tts/pyproject.toml` 保持一致。当前服务只
接收 JSON，不上传文件，因此 `python-multipart` 不是 Stable Audio 服务的
直接依赖；如果目标项目复用包含上传路由的通用 API 模块，再额外加入
`python-multipart==0.0.32`。

`fcntl`、`json`、`os`、`pathlib`、`signal`、`shutil`、`subprocess`、
`tempfile`、`threading`、`time`、`traceback`、`typing`、`uuid` 等均为
Python 标准库，不应写进依赖列表。

### 4.2 worker 和 Stable Audio 3 直接运行依赖

以下是当前 worker、官方 `stable_audio_3` 源码和当前环境中实际使用的
模型依赖。首轮迁移建议固定到当前已验证版本：

~~~text
einops==0.8.2
einops-exts==0.0.4
huggingface-hub==1.27.0
numpy==2.2.6
packaging==26.2
safetensors==0.8.0
soundfile==0.14.0
torch==2.7.1
torchaudio==2.7.1
tqdm==4.70.0
transformers==5.15.0
flash-attn==2.6.3  # 需要重新取得 cp312 兼容构建，不能使用当前 cp310 wheel
~~~

其中 `torch` 和 `torchaudio` 必须来自同一个 CUDA 12.6 wheel channel，且
版本必须分别保持 `2.7.1`；不要让 PyPI 解析成 CPU wheel或让两个包版本
漂移。`stable-audio-3` 上游 `pyproject.toml` 的正式依赖范围为：

~~~text
einops>=0.8.2
einops-exts>=0.0.4
numpy>=2.2.6
packaging>=26.0
safetensors>=0.7.0
torch==2.7.1
torchaudio==2.7.1
tqdm>=4.67.3
huggingface-hub>=1.7.1
transformers>=5.8.0
soundfile>=0.13.1
~~~

`flash-attn` 没有写入上游 `pyproject.toml`。原 `api` worker 曾主动将它
作为硬性前置；但当前上游 `transformer.py` 明确实现了无 FlashAttention 时
的 flex-attention/分块 SDPA fallback。因此新 uv worker 默认把它作为可选的
性能依赖，严格兼容旧 worker 的部署可设置
`STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN=1`。

### 4.3 uv lock 会解析的传递依赖

当前环境还包含下列由 Torch、Transformers、Hugging Face Hub、pytest 或
FlashAttention 带入的包。它们不建议全部手写进 `[project].dependencies`，
应让 uv 根据上面的直接依赖生成 `uv.lock`，但必须在 `uv sync --locked`
后用 import smoke test 和 `pip check` 验证：

~~~text
annotated-doc==0.0.5    anyio==4.14.2          certifi==2026.7.22
cffi==2.1.1             click==8.4.2           exceptiongroup==1.3.1
filelock==3.32.2        fsspec==2026.7.0       hf-xet==1.6.0
h11==0.16.0             httpcore==1.0.9        idna==3.18
Jinja2==3.1.6           markdown-it-py==4.2.0  MarkupSafe==3.0.3
mdurl==0.1.2            mpmath==1.3.0          networkx==3.4.2
packaging==26.2         pyyaml==6.0.3          regex==2026.7.19
safetensors==0.8.0      sympy==1.14.0          tokenizers==0.22.2
triton==3.3.1           typing-extensions==4.16.0
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.26.2
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
~~~

`pytest==9.1.1`、`pluggy==1.6.0`、`pygments==2.20.0`、`rich==15.0.0`、
`shellingham==1.5.4`、`typer==0.27.1` 和 `tomli==2.4.1` 只用于测试或
命令行/诊断，不是生产 worker 的必需依赖。可以放在 `dev` dependency
group，不要因为它们出现在当前 Conda 环境就把它们当成模型运行依赖。

### 4.4 系统和 GPU 依赖

uv 无法替代这些宿主机条件：

- NVIDIA 驱动，当前驱动为 `610.47`；
- 可用 CUDA runtime，与 Torch `cu126` 和 FlashAttention 构建匹配；
- Ampere 或更新架构；当前 GPU compute capability 为 `8.9`；
- Linux/glibc、C/C++ 编译工具链；
- 如果从源码编译 FlashAttention，需要 CUDA devel toolkit、`nvcc`、
  `ninja` 和足够的内存/磁盘。

当前机器可找到 `gcc`、`ffmpeg` 和 `nvidia-smi`，但当前 shell 中没有
`nvcc`，也没有找到 `ninja`。因此不能把本机源码编译 FlashAttention 当作
已经可执行的迁移方案；必须先准备 CUDA devel 环境，或取得 Python 3.12
兼容的预编译 wheel。

## 5. 本地源码仓库和权重

### 5.1 必须保留的 Stable Audio 3 源码

~~~text
路径：  /home/muyi086/tts-depency/stable-audio-3
远端：  https://github.com/Stability-AI/stable-audio-3
commit：a0b57f5483c4588f827f3552b7d5c6ca2a9687be
子包：  stable_audio_3/
~~~

当前 Conda 环境中安装的 `stable-audio-3` 指向 `/tmp` 临时目录，不能继续
使用。迁移时应将 `STABLE_AUDIO_3_REPO_PATH` 作为环境变量传给 worker，并在
worker 中将该目录加入 `sys.path`，或者安装固定 commit 的 Git/local
editable 包。禁止把 `/home/muyi086` 绝对路径写入仓库的锁文件。

推荐优先级：

1. 本机部署：`uv pip install --no-deps --editable "$STABLE_AUDIO_3_REPO_PATH"`
   并保持 `STABLE_AUDIO_3_REPO_PATH` 可配置。
2. 可复现部署：使用上面 commit 的 Git source 依赖生成 lock；离线部署则
   预先准备源码 checkout，再使用相同 commit 的 local editable 安装。
3. 不要将 Gradio、LoRA 训练和上游全量 demo 依赖放进服务环境。

### 5.2 FlashAttention 仓库

本机有：

~~~text
路径：  /home/muyi086/tts-depency/flash-attention
branch：main
commit：c75d019dea9d910312974417bc28f190dfdda6d9
描述：  fa4-v4.0.0.beta24-1-gc75d019
~~~

该仓库的 `setup.py` 仍包含 `flash_attn_2_cuda` 构建目标，可作为源码构建
候选；但当前 checkout 是较新的主分支，且实际 dry-run 失败于缺少
`csrc/cutlass` 子模块。当前机器也没有 `nvcc` 和 `ninja`，不能仅凭仓库
存在就断言它与 Torch 2.7.1、Python 3.12 和本 GPU 完全兼容。源码构建必须
在准备完整 CUDA devel 工具链、补齐子模块后执行，并通过 `flash_attn_func`
import 和生成 canary。

当前 Conda 的 wheel 是：

~~~text
flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl
~~~

它不能安装到 Python 3.12。目标环境的安装顺序应为：

1. 先安装并验证 `torch==2.7.1`、`torchaudio==2.7.1`、CUDA 12.6 版本；
2. 查找与 `cp312`、`torch2.7`、`cu126` 匹配的 FlashAttention 2 wheel；
3. 没有匹配 wheel 时，准备 CUDA devel + `nvcc` + `ninja`，固定源码
   commit 后编译，并针对 compute capability 8.9 设置架构；
4. 若两者都不可得，迁移不能宣称完成。虽然上游部分代码有 SDPA fallback，
   当前项目 worker 明确要求 FlashAttention，不能未经验证将 fallback 当作
   Stable Audio Medium 的兼容实现。

## 6. 推荐 uv 配置和安装命令

### 6.1 初始化项目

在仓库根目录执行：

~~~bash
uv init --name stable-audio-3-medium-service --python 3.12.13 stable_audio_3_medium
cd stable_audio_3_medium
uv python pin 3.12.13
~~~

目标 `pyproject.toml` 建议使用应用项目模式，先不把 `api/` 变成可发布的
Python package：

~~~toml
[project]
name = "stable-audio-3-medium-service"
version = "0.1.0"
description = "Unitale Stable Audio 3 Medium HTTP service"
requires-python = "==3.12.13"
dependencies = [
    "fastapi==0.136.3",
    "starlette==1.3.1",
    "uvicorn==0.49.0",
    "pydantic==2.13.4",
    "einops==0.8.2",
    "einops-exts==0.0.4",
    "huggingface-hub==1.27.0",
    "numpy==2.2.6",
    "packaging==26.2",
    "safetensors==0.8.0",
    "soundfile==0.14.0",
    "torch==2.7.1",
    "torchaudio==2.7.1",
    "tqdm==4.70.0",
    "transformers==5.15.0",
]

[dependency-groups]
dev = [
    "pytest==9.1.1",
    "ruff>=0.15.9",
]

[tool.uv]
package = false
index-strategy = "first-index"

[[tool.uv.index]]
name = "pypi-tuna"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

[[tool.uv.index]]
name = "aliyun-pytorch-cu126"
url = "https://mirrors.aliyun.com/pytorch-wheels/cu126/"
format = "flat"
explicit = true

[tool.uv.sources]
torch = { index = "aliyun-pytorch-cu126" }
torchaudio = { index = "aliyun-pytorch-cu126" }
~~~

上述配置不把 FlashAttention 直接写进普通依赖解析，因为目标 cp312 wheel
来源尚未确定；也不把本机绝对路径写入项目。FlashAttention 安装完成后，
用 `uv sync --inexact` 保留该编译扩展，或者在确认 cp312 wheel 来源稳定后
再把它加入项目的可审查安装配置。

如果阿里 PyTorch 镜像没有目标 cp312/cu126 wheel，允许将两个显式源替换为
官方 PyTorch cu126 index；必须保持 `torch` 和 `torchaudio` 版本一致，并在
计划的最终验证记录实际下载来源。普通 Python 包仍优先使用清华源。

### 6.2 安装基础依赖

~~~bash
cd stable_audio_3_medium
uv sync --group dev
uv run python -VV
uv run python -c 'import fastapi, pydantic, uvicorn; print("control plane ok")'
~~~

如果目标环境必须严格只安装生产依赖，使用 `uv sync --no-dev`，测试时再
使用 `uv sync --group dev`。

### 6.3 安装官方本地源码

~~~bash
export STABLE_AUDIO_3_REPO_PATH=/home/muyi086/tts-depency/stable-audio-3
uv pip install --python .venv/bin/python --no-deps --editable "$STABLE_AUDIO_3_REPO_PATH"
uv run python -c 'from stable_audio_3 import StableAudioModel; print(StableAudioModel)'
~~~

如果要让 `uv.lock` 直接记录源码版本，应改用固定 commit 的 Git source，
或者在所有部署机器约定相同的相对源码布局后再使用相对 path dependency；
不能把 `/home/muyi086/tts-depency` 或 `/tmp` 写入提交物。

### 6.4 安装 FlashAttention

优先使用与目标解释器 ABI、Torch 和 CUDA 全部匹配的 cp312 wheel，例如
安装前先核对 wheel 文件名中的：

~~~text
cp312-cp312-linux_x86_64
cu126
torch2.7
~~~

安装后验证：

~~~bash
uv pip install --python .venv/bin/python \
  /path/to/flash_attn-*-cu126torch2.7-cp312-cp312-linux_x86_64.whl

uv run python - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_func
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("flash_attn:", flash_attn.__version__, flash_attn_func)
PY
~~~

源码构建仅在准备好 `nvcc` 后执行，且必须使用固定 checkout：

~~~bash
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export FLASH_ATTN_CUDA_ARCHS=89
export MAX_JOBS=4
uv pip install --python .venv/bin/python ninja
uv pip install --python .venv/bin/python \
  --no-build-isolation --no-deps \
  --editable /home/muyi086/tts-depency/flash-attention
~~~

上面的源码构建命令在当前机器暂不能执行，因为当前 `nvcc` 和 `ninja`
均未找到；它是准备 CUDA devel 工具链后的候选命令，不是已验证结果。

### 6.5 模型权重

现有模型目录已经存在且文件完整，关键文件包括：

~~~text
$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium/model_config.json
$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium/model.safetensors
$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium/t5gemma-b-b-ul2/config.json
$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium/t5gemma-b-b-ul2/model.safetensors
$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium/t5gemma-b-b-ul2/tokenizer.json
~~~

本机已核对 `model.safetensors` 约 9.22 GB，T5Gemma 权重约 1.18 GB。权重
不进 Git、不由 `uv sync` 管理。新机器需要从 hf-mirror 准备：

~~~bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_MIRROR_DIR=${HF_MIRROR_DIR:-$HOME/hf-mirror}
uv run hf download stabilityai/stable-audio-3-medium \
  --local-dir "$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium"
~~~

服务运行时默认设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，避免
worker 在本地权重缺失时偷偷联网。

## 7. 目标项目结构和实现边界

建议目标结构：

~~~text
stable_audio_3_medium/
├── .python-version                 # 3.12.13
├── pyproject.toml                  # 控制面 + worker 依赖
├── uv.lock                         # 提交，不能含机器绝对路径
├── README.md                       # 安装、权重、FlashAttention、canary
├── main.py                         # 8313 HTTP API
├── worker.py                       # 一请求一进程的模型推理
├── runtime.py                      # GPU 状态、锁和 worker 进程组清理
└── tests/
    └── test_migration.py           # 不加载模型、不调用 CUDA 的契约测试
~~~

`main.py` 可参考现有 `api/stable_audio_3_medium_api.py`，但不能原样保留
`LocalWorkerConfig(conda_env=...)` 的调用链。迁移后的 worker 应使用目标
项目的 `.venv/bin/python`（或 `sys.executable`）启动，避免 uv 项目内部
再次调用 Conda；必要时保留显式 `STABLE_AUDIO_3_MEDIUM_RUNTIME=conda`
作为回退模式，但默认目标路径为 uv。

必须迁移并测试：

- Pydantic 请求校验、`seconds`/`duration` 归一化和全部默认值；
- health 的路径、权重文件细节、CUDA 状态和最后错误字段；
- `GPU_LOCK_FILE` 的 `flock` 生命周期；
- worker 的 JSON/WAV 临时文件、超时、非零退出、空 WAV 和整个进程组清理；
- `persist_audio_bytes` 的生成结果落盘行为；
- `LOCAL_FILES_ONLY`、`HF_MIRROR_DIR`、`STABLE_AUDIO_3_REPO_PATH`、
  `STABLE_AUDIO_3_MEDIUM_MODEL_DIR` 等环境变量覆盖；
- 只对本机开放的 `/internal/unload_all`。

不要在目标 API 进程导入 `torch`、`stable_audio_3` 或模型权重；模型依赖
只应由 one-shot worker 导入。

## 8. start.sh 切换方案

task34 阶段只生成本计划，不切换当前启动链路。当前 `start.sh` 仍由
`qwen3_tts/.venv` 启动 `api/stable_audio_3_medium_api.py`，该 wrapper 再
调用 `stable_audio_3_medium` Conda worker。

迁移代码和 canary 完成后，新增目标项目变量并替换 8313 启动命令：

~~~bash
STABLE_AUDIO_3_MEDIUM_PROJECT_DIR="${STABLE_AUDIO_3_MEDIUM_PROJECT_DIR:-$PROJECT_DIR/stable_audio_3_medium}"

HOST="$STABLE_AUDIO_3_MEDIUM_HOST" PORT="$STABLE_AUDIO_3_MEDIUM_PORT" \
  setsid uv run --no-sync --project "$STABLE_AUDIO_3_MEDIUM_PROJECT_DIR" \
  python "$STABLE_AUDIO_3_MEDIUM_PROJECT_DIR/main.py" &
~~~

切换前至少保留一个明确的回退开关，例如：

~~~bash
STABLE_AUDIO_3_MEDIUM_RUNTIME=uv       # 默认
STABLE_AUDIO_3_MEDIUM_RUNTIME=conda    # 回退到旧 api wrapper/worker
~~~

切换时不得改变端口、环境变量名、路由、请求字段、WAV MIME 类型或 GPU 锁
路径。旧 `api/stable_audio_3_medium_api.py` 和旧 worker 要等用户确认新服务
完整成功后再删除；删除 Conda 环境属于后续清理任务，不属于本计划的准备阶段。

## 9. 验证顺序和验收门槛

### 9.1 无模型单元和契约测试

默认测试不得下载权重、调用 CUDA 或访问外部服务：

~~~bash
cd stable_audio_3_medium
uv run python -m unittest discover -s tests -v
uv run python -m pytest -q tests
~~~

至少覆盖：

- `main.py` 可导入且不加载模型；
- health 在缺少模型、源码、Conda 或 CUDA 时仍返回诊断 JSON；
- `prompt`、`seconds`、`duration`、`steps`、`cfg_scale`、`seed` 的校验；
- `/v1/generate` 和 `/v2/synthesize` 的 WAV 响应 MIME 类型；
- mock worker 成功、异常、超时、非零退出、空 WAV 后的临时文件和锁清理；
- `/internal/unload_all` 的本机限制；
- `STABLE_AUDIO_3_MEDIUM_*` 环境变量覆盖。

### 9.2 目标环境预检

~~~bash
uv sync --locked --project stable_audio_3_medium
uv run --project stable_audio_3_medium python -VV
uv run --project stable_audio_3_medium python -m pip check
uv run --project stable_audio_3_medium python - <<'PY'
import fastapi, pydantic, uvicorn
import torch, torchaudio, soundfile
import flash_attn
from flash_attn import flash_attn_func
from stable_audio_3 import StableAudioModel

print("control-plane imports: ok")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("flash_attn:", flash_attn.__version__, flash_attn_func)
print("stable_audio:", StableAudioModel)
PY
~~~

必须同时满足 Python `3.12.13`、CUDA 可用、compute capability 至少 8.0、
FlashAttention 函数可导入、Stable Audio 源码可导入和 `pip check` 无错误。

### 9.3 临时端口和真实 GPU canary

先不要占用生产端口：

~~~bash
HOST=127.0.0.1 PORT=18313 \
  uv run --project stable_audio_3_medium python stable_audio_3_medium/main.py
~~~

另一个终端执行：

~~~bash
curl -fsS http://127.0.0.1:18313/v1/health
curl -fsS -X POST http://127.0.0.1:18313/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A short glass shattering sound effect, crisp and dramatic", "seconds":1}' \
  -o /tmp/stable_audio_3_medium_uv.wav
file /tmp/stable_audio_3_medium_uv.wav
ffprobe -v error -show_entries stream=sample_rate,channels,duration \
  -of default=noprint_wrappers=1 /tmp/stable_audio_3_medium_uv.wav
~~~

真实 canary 至少验证：

1. 生成成功，响应为非空 WAV，采样率 44.1 kHz、双声道；
2. `seconds=1` 和多个 `steps`/`seed` 请求行为正常；
3. 连续两次生成不会因上一次模型残留而 OOM；
4. 成功、异常、超时和取消后 GPU lock、worker 进程组和临时文件均清理；
5. `HF_HUB_OFFLINE=1` 下不会联网，权重确实来自 hf-mirror；
6. 对照旧 8313 wrapper，health 关键字段、请求兼容字段和 audio/wav 响应不变；
7. WebUI 的 `prompt_en`、`seconds` 请求可以直接生成，旧工程缺失
   `prompt_en` 时仍由前端按既有兼容逻辑处理，而不是由后端改写中文提示词。

## 10. 实施顺序

| 阶段 | 工作 | 完成标志 |
| --- | --- | --- |
| 0 | 保存当前 Conda/import/CUDA/WAV 基线 | 当前环境证据可复现 |
| 1 | 创建 `stable_audio_3_medium/`、`.python-version`、`pyproject.toml` | Python 3.12.13，基础依赖可 `uv sync` |
| 2 | 安装 Torch/cu126、官方源码和 FlashAttention cp312 或源码构建 | 目标环境 import、`pip check` 和 CUDA 预检通过 |
| 3 | 迁移 `main.py`、worker、runtime helper 和测试 | 无模型契约测试通过 |
| 4 | 临时端口启动并做真实 GPU canary | WAV、生命周期、锁和离线权重验证通过 |
| 5 | `start.sh` 8313 切换到目标 uv 项目 | 原端口和 WebUI 请求无需修改 |
| 6 | 观察旧/新路径并由用户确认 | 确认后再删除旧 API 和 Conda 环境 |

## 11. 当前风险和明确不做的事

- 不把当前 Conda 的所有 100 多个条目无差别复制到 `pyproject.toml`；
  Conda 系统库、测试工具和 GPU 传递依赖交给 Conda/uv 的各自解析层。
- 不把模型权重、WAV、`.venv`、Conda 目录、`/tmp` wheel 或机器绝对路径
  提交到 Git。
- 不把 Gradio UI、LoRA 训练、`pytorch-lightning`、`dill`、`matplotlib`
  等上游可选依赖安装到 HTTP 推理服务，除非后续需求明确包含训练或 UI。
- 不因为上游代码存在 SDPA fallback 就跳过当前 worker 要求的 FlashAttention
  验证；如果确实要支持无 FlashAttention 模式，必须另行设计、测试并明确
  性能和音质差异。
- 不在 task34 阶段删除 `api/stable_audio_3_medium_api.py`、旧 worker、
  Conda 环境或改变 `start.sh`。这些动作必须在目标 uv 服务真实 canary
  和 WebUI 契约回归通过后执行。

## 12. task34 交付状态

- [x] 已确认按模型建立 `stable_audio_3_medium/` 并使用 uv 管理的方案可行。
- [x] 已核对原 API、worker、start.sh、qwen3_tts 成功案例和 WebUI 8313 契约。
- [x] 已核对当前 `stable_audio_3_medium` Conda 环境的 Python、Torch、CUDA、
      FlashAttention、源码和权重现状。
- [x] 已列出 HTTP、模型、传递、GPU/系统依赖及镜像安装方案。
- [x] 已记录 `/home/muyi086/tts-depency/stable-audio-3` 和
      `/home/muyi086/tts-depency/flash-attention` 的用途与 commit。
- [ ] 目标 Python 3.12.13 项目实际安装并生成 `uv.lock`（下一实施任务）。
- [ ] 目标 worker/main.py 实现、start.sh 切换和真实 GPU canary（下一实施任务）。
