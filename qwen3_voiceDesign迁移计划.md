# Qwen3-TTS VoiceDesign 迁移计划：qwen3-voiceDesign Conda 到 qwen3_voiceDesign uv

> 评估日期：2026-08-14  
> 适用仓库：/home/muyi086/github/TTS-and-VoiceDesign  
> 目标目录：仓库根目录下的 qwen3_voiceDesign/  
> 相关评估：项目升级评估.md

## 1. 结论

可以按模型建立 qwen3_voiceDesign/，用 Python 3.12.13 的 uv 项目承载 Qwen3-TTS VoiceDesign 的 HTTP 控制面和一次性推理 worker。当前 Conda 环境已经是 Python 3.12.13，且 Python 包主要来自 PyPI，迁移条件较好。

这里的“完全复刻”应定义为：复刻 Python 依赖、模型调用、请求字段、WAV 响应、超时、GPU 锁和 worker 生命周期；不应把 Conda 的系统库、动态库、NVIDIA 驱动或模型权重复制进 uv 目录。

推荐边界如下：

~~~text
qwen3_voiceDesign/.venv（uv 管理，Python 3.12.13）
    ├── main.py       独立的 VoiceDesign HTTP 服务/兼容入口
    ├── worker.py     一请求一进程的 Qwen3-TTS VoiceDesign 推理
    ├── pyproject.toml
    └── uv.lock

主机层
    ├── NVIDIA 驱动和 CUDA 能力
    ├── $HOME/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    ├── api/.cache/runtime/   worker 临时文件和 GPU 锁
    └── api/tempAudio/        如后续统一保留生成 WAV
~~~

迁移后不要让新服务直接抢占主 API 的 8300。当前 /v1/qwen/design 与 MOSS、MiMo、VoxCPM2、Step-Audio-EditX 共用 api/api.py 的 8300。建议新 main.py 使用一个可配置的内部端口（例如 8314），再由 8300 的兼容 facade 转发该路由；或者在最后阶段把整个主 API 控制面一并迁移。直接让独立服务绑定 8300 会与现有主 API 冲突，也会破坏 WebUI 现有地址。

## 2. 当前实现和必须保留的契约

本次核对发现，当前没有独立的 api/qwen3_voiceDesign_api.py。对应功能由以下两部分组成：

- api/api.py：定义 QwenDesignRequest、POST /v1/qwen/design、Conda worker 调度、GPU 文件锁和 HTTP 错误处理。
- api/qwen_voice_design_worker.py：在 qwen3-voiceDesign 中加载模型、执行 VoiceDesign、分片拼接 WAV 并清理 CUDA。

### 2.1 HTTP 路由

| 项目 | 当前行为 | 迁移要求 |
| --- | --- | --- |
| 公共入口 | POST http://127.0.0.1:8300/v1/qwen/design | WebUI 地址和路径不变 |
| 成功响应 | audio/wav 原始 WAV 字节 | MIME 类型和响应体不变 |
| 错误响应 | worker 非零退出或异常时主 API 返回 HTTP 500 | 保留可诊断错误摘要 |
| 串行化 | 共享 GPU_LOCK_FILE，请求期间独占 GPU | 新 worker 继续使用同一锁语义 |
| 生命周期 | 一请求一个子进程，完成/异常后退出释放显存 | 不改成常驻模型 |
| 回退 | 当前 Conda worker 继续保留到 canary 通过 | 支持显式回退 |

### 2.2 QwenDesignRequest 字段

新服务应接受下面的全部字段。save_as 当前虽然被模型传递，但旧 worker 不用它，第一版仍需接受以保持请求兼容。

| 字段 | 类型 | 当前默认值 | 用途 |
| --- | --- | --- | --- |
| voice_description | str | 必填 | 音色描述；worker 会拒绝空值 |
| text | str | 这是生成的参考音频预览。 | 待生成文本 |
| save_as | str 或 null | designed_voice.wav | 兼容字段，不改变现有文件命名行为 |
| language | str 或 null | Chinese | Qwen 语言名称 |
| max_chars_per_chunk | int 或 null | 0 | 0 表示不分片 |
| pause_ms | int 或 null | 250 | 分片之间的静音长度 |
| max_new_tokens | int 或 null | 2048 | 生成上限 |
| top_p | float 或 null | null | 可选采样参数 |
| temperature | float 或 null | null | 可选采样参数 |
| dtype | str 或 null | auto | auto 当前解析为 bfloat16 |
| attn_implementation | str 或 null | auto | 无 FlashAttention 时回退 sdpa |
| device_map | str 或 null | cuda:0 | 模型设备映射 |

worker 仍需保留：文本规范化、中文/英文标点分片、每个分片使用相同音色指令、音频转单声道、分片间插入静音、非空 WAV 校验，以及异常路径的 gc/torch.cuda.empty_cache()。

## 3. 当前环境证据

本机核对命令及结果：

~~~text
环境：qwen3-voiceDesign
Python：3.12.13
qwen-tts：0.1.1
torch：2.12.0+cu130
torch.version.cuda：13.0
torch.cuda.is_available()：True
flash_attn：未安装
模型目录：/home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
模型目录大小：约 4.3G
/usr/bin/sox：存在
/usr/bin/ffmpeg：存在
~~~

模型目录已经存在，包含 config.json、generation_config.json、model.safetensors、speech_tokenizer/model.safetensors、Tokenizer 配置和词表。模型权重继续从 HF_MIRROR_DIR 引用，不复制到 qwen3_voiceDesign/，也不写入 uv.lock。

Conda 的 conda env export --from-history 结果主要是 Python、系统库和基础工具；真正的 Qwen Python 栈由 pip 安装。因此迁移应以 Python 包及其版本为基线，而不是把整个 conda env export 原样交给 uv。

## 4. 依赖清单

### 4.1 应显式写入 pyproject.toml 的依赖

| 包 | 当前版本 | 作用 |
| --- | ---: | --- |
| fastapi | 0.136.3 | 新服务 HTTP、Pydantic 请求模型和 WAV 响应 |
| starlette | 1.3.1 | CORS/middleware 等 FastAPI 运行基础；若代码不直接导入也可由 FastAPI 锁定 |
| uvicorn | 0.49.0 | 启动 ASGI 服务 |
| pydantic | 2.13.4 | QwenDesignRequest 校验 |
| python-multipart | 0.0.32 | 参考 qwen3 项目保持服务上传能力；VoiceDesign JSON 本身不依赖它 |
| qwen-tts | 0.1.1 | Qwen3TTSModel 和官方 VoiceDesign 推理实现 |
| torch | 2.12.0（实际 CUDA wheel 为 +cu130） | CUDA、模型推理、dtype 和显存清理 |
| torchaudio | 2.11.0（实际 CUDA wheel 为 +cu130） | Qwen-TTS 官方音频依赖 |
| numpy | 2.4.6 | worker 音频数组转换、单声道和拼接 |
| soundfile | 0.14.0 | WAV 写出；worker 直接导入 |

qwen-tts 0.1.1 的发行元数据声明以下运行依赖，不能用 --no-deps 跳过：

~~~text
transformers==4.57.3
accelerate==1.12.0
gradio
librosa
torchaudio
soundfile
sox
onnxruntime
einops
~~~

gradio 虽不是当前 wrapper 直接使用的 UI，但它是 qwen-tts 的正常发行依赖，第一版应保留。

### 4.2 关键传递依赖和硬件包

uv.lock 会自动锁定下列依赖，不建议把它们全部当作顶层项目依赖手工维护：

- Hugging Face：huggingface-hub 0.36.2、tokenizers 0.22.2、safetensors 0.8.0、fsspec 2026.4.0、regex 2026.5.9。
- 音频/数值：librosa 0.11.0、soxr 1.1.0、audioread 3.1.0、numba 0.65.1、llvmlite 0.47.0、scipy 1.17.1、joblib 1.5.3、scikit-learn 1.9.0。
- CUDA/PyTorch：cuda-toolkit 13.0.2、cuda-bindings 13.3.1、triton 3.7.0 以及 nvidia-* CUDA 13 运行时包。
- Web/序列化：anyio、httpx、requests、pydantic-core、orjson、PyYAML、protobuf、gradio-client 等。

pip、setuptools、wheel 和 Conda 的 glibc/OpenMP/系统库不应写成 uv 的业务顶层依赖。NVIDIA 驱动仍由宿主机提供，uv 不能替代驱动安装。

### 4.3 FlashAttention

当前 qwen3-voiceDesign 环境没有 flash-attn，旧 worker 的 attn_implementation=auto 会使用 PyTorch sdpa，因此它不是当前功能的必需依赖。第一阶段不安装它，先固定 sdpa 完成迁移 canary。

task17 实测环境为 NVIDIA GeForce RTX 4070 Ti SUPER、compute capability 8.9，硬件属于 FlashAttention-2 支持的 Ada 范围；但当前 uv 环境使用 torch 2.12.0+cu130，系统 CUDA 编译器为 nvcc 12.0，且 qwen3_voiceDesign 环境没有 ninja。/home/muyi086/tts-depency/flash-attention 当前 HEAD 是 fa4-v4.0.0.beta24，不是已经针对本项目锁定的 FA2 wheel。仓库源码编译需要 CUDA/PyTorch 工具链匹配，当前组合不适合未经验证直接编译安装。因此本次迁移明确不把 flash-attn 加入默认依赖，健康接口会报告 flash_attn=false，运行日志会明确使用 sdpa。

只有在确认显卡、CUDA、编译器和 PyTorch ABI 后，才考虑可选安装：

~~~bash
cd /home/muyi086/github/TTS-and-VoiceDesign/qwen3_voiceDesign
MAX_JOBS=4 uv pip install --python .venv/bin/python \
  --no-build-isolation /home/muyi086/tts-depency/flash-attention
~~~

这不是默认安装步骤，也不应因为安装了可选包就修改默认 attn_implementation。

## 5. /home/muyi086/tts-depency 仓库审计

已发现的本地仓库及其与本次迁移的关系：

| 本地仓库 | 是否属于 Qwen3 VoiceDesign | 说明 |
| --- | --- | --- |
| LongCat-AudioDiT | 否 | LongCat 独立服务 |
| MOSS-Audio | 否 | MOSS Audio 独立模型 |
| MOSS-TTS | 否 | MOSS TTS/VoiceGenerator/SoundEffect |
| Step-Audio-EditX | 否 | Step-Audio-EditX 独立编辑服务 |
| stable-audio-3 | 否 | Stable Audio 独立服务 |
| vllm | 否 | 当前 worker 不使用 vLLM |
| flash-attention | 可选 | 当前环境未安装，第一阶段不纳入默认依赖 |

结论：当前没有必须从 tts-depency 拷贝的 Qwen 本地仓库。qwen-tts 0.1.1 已在 Conda 环境中从 PyPI 安装，发行包主页是 https://github.com/Qwen/Qwen3-TTS。迁移时优先继续使用 PyPI 包，不要把上游源码仓库或运行时权重复制进项目。

api/vendor/qwen_libs 是被 .gitignore 忽略的本地 sidecar，里面也有 qwen-tts 0.1.1 及若干依赖，但它不是本次迁移的输入；新 uv 项目应使用环境内正常安装的 qwen-tts，不要依赖偶然的 sys.path 注入。

## 6. 当前 Conda 环境的 Python 包完整基线

下面是本次核对得到的 qwen3-voiceDesign pip 包基线。它用于迁移后的对账，不代表每项都需要成为顶层依赖；uv.lock 负责传递依赖的最终锁定。pip @ file:///... 这类机器本地路径已故意排除，不能写入项目配置。

~~~text
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
~~~

对账命令：

~~~bash
conda run -n qwen3-voiceDesign python --version
conda run -n qwen3-voiceDesign python -m pip freeze --all | sort
~~~

## 7. 初始化 uv 项目和镜像配置

建议使用应用项目模式，不生成指向不存在源码包的可安装库声明：

~~~bash
cd /home/muyi086/github/TTS-and-VoiceDesign
uv python install 3.12.13
uv init --app --python 3.12.13 --no-readme qwen3_voiceDesign
cd qwen3_voiceDesign
uv python pin 3.12.13
~~~

然后将 pyproject.toml 调整为下面的最小可复现配置。普通包走清华源；PyTorch CUDA wheel 使用单独的阿里 cu130 flat index，避免普通包被错误解析到 PyTorch 索引。

~~~toml
[project]
name = "qwen3-voicedesign-service"
version = "0.1.0"
description = "Unitale Qwen3-TTS VoiceDesign HTTP service"
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
url = "https://mirrors.aliyun.com/pytorch-wheels/cu130/"
format = "flat"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchaudio = { index = "pytorch-cu130" }
~~~

清华源不可用时，可将 pypi-tuna 替换为中科大 https://pypi.mirrors.ustc.edu.cn/simple 或阿里 https://mirrors.aliyun.com/pypi/simple/。不要把 HF_MIRROR_DIR 与 Python 包镜像混淆：前者只负责模型权重。

安装、锁定和对账：

~~~bash
cd /home/muyi086/github/TTS-and-VoiceDesign/qwen3_voiceDesign
uv lock
uv sync

uv run python -VV
uv run python -c 'import fastapi, pydantic, uvicorn, qwen_tts, torch, torchaudio, numpy, soundfile; print("imports ok")'
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
uv tree
uv run python -m pip list --format=freeze | sort
~~~

如果 PyTorch 镜像没有对应的 Linux x86_64 wheel，先不要偷偷回退到 CPU 版；应记录实际解析结果，并保持 CUDA wheel 与 torchaudio 版本匹配。uv.lock 生成后必须提交，.venv 不提交。

## 8. 运行路径和配置迁移

新目录建议先形成如下结构：

~~~text
qwen3_voiceDesign/
├── .python-version
├── pyproject.toml
├── uv.lock
├── main.py
└── worker.py
~~~

main.py 应实现 /v1/health 和与旧接口相同的 VoiceDesign JSON 路由；worker.py 可从 api/qwen_voice_design_worker.py 提取，改为使用新项目的 sys.executable。迁移期间不要直接删除或覆盖旧文件，保留它们作为回退和行为对照。

推荐的环境变量边界：

~~~bash
export HF_MIRROR_DIR="$HOME/hf-mirror"
export QWEN_VOICEDESIGN_MODEL_DIR="$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
export QWEN_VOICEDESIGN_HOST="127.0.0.1"
export QWEN_VOICEDESIGN_PORT="8314"
export QWEN_VOICEDESIGN_DEVICE_MAP="cuda:0"
export QWEN_VOICEDESIGN_DTYPE="auto"
export QWEN_VOICEDESIGN_ATTN_IMPLEMENTATION="auto"
export QWEN_VOICEDESIGN_REQUEST_TIMEOUT="900"
export PROMPTS_DIR="$PWD/../api/prompts"
export RUNTIME_CACHE_DIR="$PWD/../api/.cache/runtime"
export GPU_LOCK_FILE="$RUNTIME_CACHE_DIR/gpu-runtime.lock"
export LOCAL_FILES_ONLY="1"
export HF_HOME="$HF_MIRROR_DIR"
~~~

模型和运行数据不放进 uv 项目；LOCAL_FILES_ONLY=1 时继续设置 HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1。

### 8.1 start.sh 切换建议

当前 start.sh 中：

- QWEN_VOICEDESIGN_CONDA_ENV 默认是 qwen3-voiceDesign；
- 主 API 8300 通过 conda run ... api/api.py 提供 /v1/qwen/design；
- 8305 是另一套 Qwen3-TTS Base 服务，不能与本次 VoiceDesign 迁移混淆。

因此推荐增加显式运行器开关：

~~~text
QWEN_VOICEDESIGN_RUNNER=conda   旧路径，调用 api/api.py 中的 Conda worker
QWEN_VOICEDESIGN_RUNNER=uv      新路径，启动 qwen3_voiceDesign/main.py
~~~

新服务先只监听内部 8314（端口可通过变量覆盖），8300 的 /v1/qwen/design 保持 facade。等新服务健康检查、请求契约和真实模型 canary 通过后，再把 8300 的该路由改为调用新服务；主 API 的其它 provider 不应被本次改动重启逻辑或改端口。确认 WebUI 行为完全一致后，才删除旧的 api/qwen_voice_design_worker.py 引用。

## 9. 验收清单

### 9.1 无模型/无 CUDA 的控制面检查

- uv sync 成功，不下载模型权重。
- uv run python -c "import fastapi, pydantic, uvicorn, qwen_tts" 成功。
- uv run python -m py_compile main.py worker.py 成功。
- /v1/health 在模型目录缺失时返回可诊断 JSON，而不是 import 阶段崩溃。
- 主 API 和新服务的 mock 测试不初始化 CUDA、不调用外部服务。

### 9.2 VoiceDesign HTTP 契约

| 检查 | 预期 |
| --- | --- |
| POST /v1/qwen/design | 8300 路径不变；请求字段与旧 schema 一致 |
| 最小请求 | 只传 voice_description 也能使用旧默认 text |
| 参数请求 | language、分片、采样、dtype、attention、device_map 均能传到 worker |
| 成功生成 | HTTP 200，Content-Type: audio/wav，WAV 非空 |
| 空文本/空描述 | 在校验或 worker 处得到明确错误 |
| worker 超时/非零退出 | 子进程组结束、临时文件清理、HTTP 500 有错误摘要 |
| GPU 锁 | 成功、异常、超时后都释放；不残留 worker |

### 9.3 真实模型 canary

控制面检查通过后，使用当前本地 VoiceDesign 权重运行：

1. attn_implementation=sdpa 完成单句中文音色设计。
2. 检查 WAV 采样率、声道和非空内容。
3. 完成一次长文本分片，确认分片静音和顺序保持。
4. 依次测试 language、dtype、top_p、temperature 和 device_map 覆盖。
5. 连续两次请求后检查子进程退出和显存释放。
6. 人为制造错误/超时，确认锁文件仍可被下一请求取得。
7. 用 WebUI 的真实请求确认 8300 facade 的响应仍为 WAV，且其它音色设计 provider 不受影响。

## 10. 回滚和实施顺序

回滚点：

1. 保留 api/api.py、api/qwen_voice_design_worker.py 和 qwen3-voiceDesign Conda 环境。
2. QWEN_VOICEDESIGN_RUNNER=conda 时恢复旧 worker 调度。
3. 新旧实现共用模型目录、GPU_LOCK_FILE、运行缓存和模型参数，不修改接口来绕过兼容问题。
4. 不提交 .venv、模型权重、参考音频、生成 WAV、缓存和 api/vendor。
5. CUDA wheel、torchaudio、驱动或模型加载不兼容时只回退 VoiceDesign，不影响 8305、8306、8307、8308、8311、8313。

建议执行顺序：

1. 保存本文件所依据的 Conda Python 版本、pip 快照、模型目录和 GPU 检查结果。
2. 创建 qwen3_voiceDesign/，固定 Python 3.12.13，写入 pyproject.toml。
3. 使用国内 PyPI 镜像和 PyTorch cu130 源完成 uv lock、uv sync。
4. 完成 import、版本、CUDA 和 py_compile 检查。
5. 从旧 worker 提取实现，保留原 API 字段、默认值、WAV 输出和错误清理。
6. 写无模型契约测试和 worker subprocess mock 测试。
7. 先用内部 8314 运行新服务，再让 8300 facade 转发 /v1/qwen/design。
8. 完成长文本、异常、锁释放和 WebUI 真实请求 canary。
9. canary 通过后再将 uv runner 设为默认；用户确认迁移完整后才删除旧引用。

## 11. 最终判断

方案可行，但不是“把 qwen3-voiceDesign 的 Conda 目录整体复制到 uv”。正确做法是：用 pyproject.toml/uv.lock 复刻 Python 依赖，用 PyTorch cu130 专用源处理 CUDA wheel，用 HF_MIRROR_DIR 继续挂载模型权重，并保留主 API 8300 的兼容 facade、GPU 锁和一次性 worker。

本次 tts-depency 审计没有发现必须引入的 Qwen 本地仓库；默认不需要 flash-attention、vllm 或其它模型仓库。完成上述安装和验收后，才进入下一步的 main.py 实现与 start.sh 切换。
