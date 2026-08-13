# Qwen3-TTS 迁移计划：Conda 到 uv

> 评估日期：2026-08-13  
> 适用仓库：TTS-and-VoiceDesign  
> 目标目录：仓库根目录下的 qwen3_tts/  
> 相关评估：项目升级评估.md

## 1. 结论

这次迁移可行，但“完全复刻”应定义为：在 Python 3.12.13 下复刻 Qwen3-TTS 的 Python 包、API 行为、模型调用参数和运行时数据路径；不能定义为把 Conda 的系统库、编译器、动态库和 CUDA 驱动逐字复制到 uv 环境。

推荐边界：

```
qwen3_tts/.venv（uv 管理）
    ├── qwen3_tts/main.py       HTTP API，替代 api/qwen3_tts_api.py
    ├── qwen3_tts/worker.py     一请求一进程的 Qwen 推理 worker
    └── pyproject.toml/uv.lock   Python 依赖和锁文件

主机层
    ├── NVIDIA 驱动和 CUDA 能力
    ├── ~/hf-mirror/Qwen/...    模型权重，继续由本地路径引用
    ├── api/prompts/             参考音频和 prompt sidecar
    ├── api/tempAudio/           生成 WAV
    └── api/.cache/runtime/      HF、Numba、锁和 worker 临时文件
```

迁移后 Qwen3-TTS 的 HTTP 服务可由 uv run 启动，Qwen worker 直接使用同一个 uv 虚拟环境的 Python 解释器。其它模型仍由现有 start.sh 使用各自的 Conda 环境；不要把其它模型依赖放进 qwen3_tts/pyproject.toml。

当前 Qwen 环境很适合先迁移：Conda history 只有 Python=3.12，Python 包全部来自 PyPI，且实际 Python 已是 3.12.13。uv 不会复制 Conda 的系统层包，因此迁移后的验收必须包括 import、CUDA 和真实模型 canary。

## 2. 当前实现和功能边界

api/qwen3_tts_api.py 是轻量 FastAPI 包装器，当前路由必须原样保留：

- GET /v1/health
- POST /internal/unload_all
- POST /v1/upload_audio
- GET /v1/check/audio
- POST /v2/synthesize

它还负责 CORS、multipart 上传、文件名哈希、参考文本 sidecar、Pydantic 请求校验、GPU 文件锁和 worker 超时/清理。

api/qwen3_tts_worker.py 才导入 numpy、soundfile、torch 和 qwen_tts.Qwen3TTSModel，完成模型加载、参考音频克隆、x-vector-only 回退、长文本分片、音频拼接、前导静音裁剪、WAV 写出和 CUDA 清理。当前包装器通过 conda run -n qwen3-tts 启动 worker；迁移只替换进程启动方式，不改变请求 JSON、worker 生命周期或 GPU 锁语义。

已核对的本机证据：

- qwen_tts、torch、torchaudio 可以在 qwen3-tts 环境导入；
- torch==2.12.0+cu130，torch.cuda.is_available() 为 True；
- GPU 为 NVIDIA GeForce RTX 4070 Ti SUPER，驱动为 610.47；
- 当前 Qwen wrapper 导入成功，路由与上表一致；
- ~/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-Base 已存在；
- api/prompts/ 已有 WAV 和 .prompt.txt sidecar；
- 系统已有 /usr/bin/sox 和 /usr/bin/ffmpeg。

## 3. 依赖清单

### 3.1 应显式写入 pyproject.toml 的直接依赖

| 包 | 当前版本 | 原因 |
| --- | ---: | --- |
| fastapi | 0.136.3 | HTTP 应用、请求模型、上传和响应 |
| starlette | 1.3.1 | API 直接导入 BaseHTTPMiddleware |
| uvicorn | 0.49.0 | 启动 ASGI 服务 |
| pydantic | 2.13.4 | CloneSynthesisRequest 和 Qwen 请求模型 |
| python-multipart | 0.0.32 | UploadFile、File、Form 的 multipart 解析 |
| qwen-tts | 0.1.1 | Qwen3TTSModel 和官方推理实现 |
| torch | 2.12.0（当前 cu130） | CUDA、模型推理、dtype、显存清理 |
| torchaudio | 2.11.0 | qwen-tts 官方音频依赖 |
| numpy | 2.4.6 | 音频转 mono、RMS、拼接 |
| soundfile | 0.14.0 | 读取/写出 WAV |

qwen-tts==0.1.1 的发行包还声明：

```
transformers==4.57.3
accelerate==1.12.0
gradio
librosa
torchaudio
soundfile
sox
onnxruntime
einops
```

gradio 虽不是当前 wrapper 直接使用的 UI，但它是 qwen-tts 的正常发行依赖，第一版应保留。不要直接用 qwen-tts --no-deps 伪造一个“更小”的环境。

uv.lock 会锁定 Hugging Face、音频、CUDA/PyTorch、Web、数值计算和序列化传递依赖。pip、setuptools、wheel 和 Conda 基础包不应作为业务顶层依赖。

### 3.2 FlashAttention

当前 qwen3-tts 环境没有 flash-attn。Qwen 会回退到 PyTorch 实现，当前 API 的 attn_implementation=auto 在没有该包时使用 sdpa，所以它不是当前功能的必需依赖。

/home/muyi086/tts-depency/flash-attention 只有在需要 FlashAttention 2 且显卡、CUDA、编译器均满足要求时才使用：

```
MAX_JOBS=4 uv pip install --python .venv/bin/python \
  --no-build-isolation \
  ~/tts-depency/flash-attention
```

第一阶段保持 sdpa，不把 FlashAttention 加入默认验收；如果安装了但没有写入项目依赖，使用 uv sync --inexact，避免被 exact sync 移除。

## 4. tts-depency 仓库审计

| 仓库 | 与本次 Qwen 迁移的关系 |
| --- | --- |
| flash-attention | 可选性能/显存优化；当前环境未安装 |
| LongCat-AudioDiT | LongCat 独立服务，不属于 Qwen |
| MOSS-Audio | MOSS Audio 独立模型，不属于 Qwen |
| MOSS-TTS | MOSS TTS/VoiceGenerator，不属于 Qwen |
| Step-Audio-EditX | 独立编辑服务，不属于 Qwen |
| stable-audio-3 | Stable Audio 独立服务，不属于 Qwen |
| vllm | 当前 worker 未使用 vLLM |

当前 API 不需要从 tts-depency 拷贝 Qwen 上游仓库。qwen-tts==0.1.1 已从 PyPI 安装，api/vendor/qwen_libs 是被 .gitignore 忽略的可选 sidecar，且 QWEN3_TTS_USE_QWEN_LIBS 默认关闭。新 uv 项目默认使用 PyPI 的 qwen-tts，不要把未跟踪 vendor 目录当成迁移输入。

## 5. 当前环境完整 pip 快照

以下是本次核对得到的 pip 侧完整快照（108 项）。它是迁移后的对账基线，不代表 108 项都要成为顶层依赖；最终以 uv.lock 和 import/推理验证为准。

```
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
einops==0.8.2
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
huggingface_hub==0.36.2
idna==3.18
Jinja2==3.1.6
joblib==1.5.3
lazy-loader==0.5
librosa==0.11.0
llvmlite==0.47.0
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
msgpack==1.2.0
narwhals==2.22.1
networkx==3.6.1
numba==0.65.1
numpy==2.4.6
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
orjson==3.11.9
packaging==26.0
pandas==3.0.3
pillow==12.2.0
pip==26.1.1
platformdirs==4.10.0
pooch==1.9.0
protobuf==7.35.1
psutil==7.2.2
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
pydub==0.25.1
Pygments==2.20.0
python-dateutil==2.9.0.post0
python-multipart==0.0.32
pytz==2026.2
PyYAML==6.0.3
qwen-tts==0.1.1
regex==2026.5.9
requests==2.34.2
rich==15.0.0
safehttpx==0.1.7
safetensors==0.8.0
scikit-learn==1.9.0
scipy==1.17.1
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
tokenizers==0.22.2
tomlkit==0.14.0
torch==2.12.0
torchaudio==2.11.0
tqdm==4.68.2
transformers==4.57.3
triton==3.7.0
typer==0.26.7
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.7.0
uvicorn==0.49.0
wheel==0.46.3
```

原始快照命令：

```
conda run -n qwen3-tts python -m pip list --format=freeze | sort -f \
  > qwen3_tts-conda-pip-freeze-20260813.txt
```

说明：快照中包含 pip、setuptools、wheel 以及大量传递依赖，不要直接把它当作最终 uv 依赖文件。

## 6. 初始化 uv 项目

```
cd ~/github/TTS-and-VoiceDesign
uv python install 3.12.13
uv init --app --python 3.12.13 --no-readme qwen3_tts
cd qwen3_tts
uv python pin 3.12.13
```

uv init qwen3_tts 本身可用；增加 --app、--python 3.12.13 和 --no-readme，是为了明确这是应用目录，不生成虚假的可发布库。

建议将 uv init 生成的 pyproject.toml 调整为：

```
[project]
name = "qwen3-tts-service"
version = "0.1.0"
description = "Unitale Qwen3-TTS HTTP service"
requires-python = "==3.12.13"
dependencies = [
    "fastapi==0.136.3",
    "starlette==1.3.1",
    "uvicorn==0.49.0",
    "pydantic==2.13.4",
    "python-multipart==0.0.32",
    "qwen-tts==0.1.1",
    "torch==2.12.0",
    "torchaudio==2.11.0",
    "numpy==2.4.6",
    "soundfile==0.14.0",
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
torch = { index = "pytorch-cu130" }
torchaudio = { index = "pytorch-cu130" }
```

普通 Python 包优先走清华源，也可换成中科大 https://pypi.mirrors.ustc.edu.cn/simple 或阿里 https://mirrors.aliyun.com/pypi/simple/。CUDA 13.0 的 PyTorch wheel 使用官方 PyTorch CUDA index，因为国内 PyPI 镜像是否及时同步 CUDA wheel 不能假设。explicit=true 防止普通包误走 PyTorch index。

如果 cu130 索引没有当前版本，可使用 uv backend 重新解析：

```
uv pip install --python .venv/bin/python --torch-backend cu130 \
  torch==2.12.0 torchaudio==2.11.0
```

然后把实际解析出的版本写回项目配置并重新 uv lock，不要留下未锁定的手工环境。

安装和锁定：

```
cd ~/github/TTS-and-VoiceDesign/qwen3_tts
uv lock --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --default-index https://pypi.tuna.tsinghua.edu.cn/simple

# 清华源不可用时二选一：
# uv lock --default-index https://pypi.mirrors.ustc.edu.cn/simple
# uv sync --default-index https://mirrors.aliyun.com/pypi/simple/

uv run python -VV
uv run python -m pip list --format=freeze | sort -f
uv tree
```

## 7. main.py 和 start.sh 迁移

### 7.1 第一版结构

```
qwen3_tts/
├── .python-version
├── pyproject.toml
├── uv.lock
├── main.py
├── worker.py
├── audio_output.py
├── synthesis_request.py
└── gpu_runtime.py
```

可以先以现有 api/qwen3_tts_api.py、api/qwen3_tts_worker.py 为行为基线，再做最小路径改动：

1. QWEN3_TTS_WORKER_SCRIPT 指向 qwen3_tts/worker.py；
2. worker 使用当前 uv 项目的解释器，推荐 sys.executable，不再使用 conda run；
3. QWEN3_TTS_CONDA_ENV 保留为兼容配置/健康信息，但不再决定 Qwen worker 环境；
4. 共享模块使用明确的本地导入，不依赖偶然的 sys.path；
5. main.py 仍支持 uvicorn.run(app, host=..., port=...) 直接启动。

worker 仍必须一请求一进程。uv 只负责解释器和依赖，不应借机改成常驻模型。

### 7.2 HF mirror 和运行数据

```
export HF_MIRROR_DIR="$HOME/hf-mirror"
export HF_HOME="$HF_MIRROR_DIR"
export QWEN3_TTS_MODEL_DIR="$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
export PROMPTS_DIR="$PWD/../api/prompts"
export QWEN3_TTS_OUTPUT_DIR="$PWD/../api/tempAudio"
export RUNTIME_CACHE_DIR="$PWD/../api/.cache/runtime"
export GPU_LOCK_FILE="$RUNTIME_CACHE_DIR/gpu-runtime.lock"
export LOCAL_FILES_ONLY=1
```

部署脚本应继续设置 HF_HUB_OFFLINE=1 和 TRANSFORMERS_OFFLINE=1。模型权重不放进 qwen3_tts/，不随代码提交，也不在 uv.lock 中管理。

### 7.3 渐进切换

不要一开始删除旧入口。增加 QWEN3_TTS_RUNNER 开关：

- QWEN3_TTS_RUNNER=conda：保留当前 8305，作为回退路径；
- QWEN3_TTS_RUNNER=uv：只将 8305 替换为 uv run --project "$PROJECT_DIR/qwen3_tts" python main.py；
- 主 API、8306、8307、8308、8311、8313 的启动方式不随本次 Qwen 迁移改变。

8305、路由、响应 MIME、请求字段和模型路径确认通过后，才把 uv 设为 Qwen 默认值。CONDA_ENV 仍可继续服务其它包装器，不能全局删除。

## 8. 兼容性验收

控制面检查：

```
cd ~/github/TTS-and-VoiceDesign/qwen3_tts
uv run python -c 'import fastapi, pydantic, uvicorn, qwen_tts; print("imports ok")'
uv run python -m py_compile main.py worker.py audio_output.py synthesis_request.py gpu_runtime.py
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

健康检查必须返回可诊断 JSON；缺少模型时不能在 import 阶段崩溃。健康 JSON 至少保留 paths、available、cuda、runtime 和 last_errors 顶层结构。

HTTP 契约：

| 检查 | 预期 |
| --- | --- |
| GET /v1/health | 8305，报告模型、worker、CUDA、锁和本地模式 |
| POST /v1/upload_audio | audio、full_path、可选 prompt_text 保持不变 |
| GET /v1/check/audio | file_name、哈希文件名和 has_prompt_text 保持不变 |
| POST /v2/synthesize | text、audio_path、prompt_text 和 Qwen 参数保持不变 |
| 成功合成 | HTTP 200，Content-Type: audio/wav |
| worker 超时/异常 | HTTP 500 有错误摘要，进程组结束，GPU 锁释放 |
| /internal/unload_all | 仍只允许本机访问，返回兼容 JSON |

真实模型 canary 必须在控制面检查通过后运行，第一轮固定 attn_implementation=sdpa。使用现有参考音频依次验证上传、检查、单句合成、长文本分片、prompt_text、x-vector-only、连续请求显存释放，以及超时/中断后的 worker 和锁清理。

## 9. 回滚和执行顺序

回滚点：

1. 不删除现有 api/qwen3_tts_api.py、api/qwen3_tts_worker.py 和 Conda 环境；
2. start.sh 通过 QWEN3_TTS_RUNNER=conda 恢复原 8305；
3. 新旧服务使用相同的 PROMPTS_DIR、GPU_LOCK_FILE、模型目录和端口；
4. 不提交 qwen3_tts/.venv、模型、参考音频、生成 WAV 或 vendor 副本；
5. CUDA wheel、torchaudio、transformers 或驱动不兼容时先回退运行器，不修改接口绕过错误。

实施顺序：

1. 保存 Conda pip freeze、uv/GPU/模型路径基线；
2. 创建 qwen3_tts/，固定 Python 3.12.13，写入 pyproject.toml；
3. 使用国内 PyPI 镜像和 PyTorch cu130 源完成 uv lock、uv sync；
4. 完成 import、版本、CUDA 和 py_compile 检查；
5. 搬运/提取 main.py、worker.py 及共享运行时模块，保持原路由和请求模型；
6. 增加无模型契约测试和 worker mock 测试；
7. 用 QWEN3_TTS_RUNNER=uv 启动 8305，完成健康、上传、克隆、WAV 和异常 canary；
8. canary 通过后才设为默认，保留 Conda 回退；
9. 最后再决定是否安装 FlashAttention，以及是否迁移其它模型。

## 10. 最终判断

Qwen3-TTS 适合率先迁移到 uv：当前 Conda history 几乎只有 Python，Python 包已经全部来自 PyPI，API/worker 边界清楚，Python 版本也已是 3.12.13。可行目标是“uv 管理 Qwen API/worker 的 Python 运行时，主机提供 CUDA，HF mirror 提供权重”。

不应追求“用 uv 消除 Conda、系统 CUDA、驱动和所有本地依赖”。那不是当前 API 的必要条件，并会把 PyTorch CUDA wheel、编译扩展和其它模型耦合起来，增加原有功能回归风险。
