# LongCat-AudioDiT-3.5B-bf16 → LongCat_AudioDiT_3.5B_bf16 迁移计划

> 原始模型环境：LongCat-AudioDiT-3.5B-bf16
>
> 目标 uv 项目/环境：LongCat_AudioDiT_3.5B_bf16/
>
> 评估日期：2026-08-14
>
> 评估依据：task25.md、项目升级评估.md、当前 LongCat API/worker/start.sh、实际
> Conda 环境、/home/muyi086/tts-depency/LongCat-AudioDiT 和 hf-mirror 资产。

## 1. 结论

### 1.1 可行性

可以按模型创建 LongCat_AudioDiT_3.5B_bf16/，使用 Python 3.12.13 的 uv 项目
承载 LongCat 的 HTTP 服务和一次性推理 worker，再由 start.sh 启动新项目。
当前 LongCat 推理路径是官方 Python 源码加 PyTorch、Transformers 和音频处理库，
没有必须依赖 Conda 专有 Python 扩展，这是迁移可行的基础。

但 uv 不能逐字替代一个 Conda 环境：NVIDIA 驱动、宿主机 CUDA 能力、系统动态库、
ffmpeg/libsndfile 等 native runtime 和模型权重不属于 Python 依赖解析范围。
因此本计划中的“复刻”定义为：

1. 用 uv 锁定 Python 3.12.13 和已验证的 Python 运行依赖。
2. 继续使用宿主机 NVIDIA 驱动和 CUDA 运行能力。
3. 继续从外部路径引用官方 LongCat 源码、hf-mirror 模型和 UMT5 tokenizer。
4. 通过无模型 import/health、worker dry-run 和真实 GPU canary 证明行为一致。

如果要求 Conda 导出的每个 native 包也在 uv 中有一一对应物，则不可行；那种要求
应保留 Conda 环境，或另外维护系统/容器层依赖清单。

### 1.2 推荐目录边界

```text
LongCat_AudioDiT_3.5B_bf16/
├── pyproject.toml       # API + worker 依赖、uv 镜像配置
├── uv.lock              # 目标机成功解析后提交
├── .python-version      # 3.12.13
├── main.py              # 原 longcat_audiodit_api.py 的 HTTP 控制面
├── worker.py            # 一请求一进程的模型推理
└── runtime.py           # uv worker 启动、超时和清理边界

外部资产（不复制、不提交 Git）：
├── $HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16
├── $HF_MIRROR_DIR/google/umt5-base
└── $LONGCAT_AUDIODIT_REPO_PATH
```

第一阶段不需要把官方仓库复制到模型项目，也不需要把权重复制到项目目录。
worker 继续通过 LONGCAT_AUDIODIT_REPO_PATH 注入 audiodit 的 import path。

### 1.3 当前环境不能直接照搬

当前环境是 Python 3.10.20，不是目标 Python 3.12.13；当前环境还包含 Gradio、
FunASR、SpeechBrain、ModelScope、评测和其他历史残留包。pip check 通过只说明
当前元数据没有冲突，不能证明这些包都是 LongCat 所需。

当前模型配置要求 Transformers 5.3.0，实际环境也正是该版本；PyTorch 实际为
2.13.0+cu130。这两个版本应先作为迁移基线，而不是按旧 requirements.txt 的
最小下限直接解析。

## 2. 当前运行链路和必须保留的功能

### 2.1 当前启动链路

当前 start.sh 的 LongCat 逻辑是：

```text
start.sh
  └─ conda run -n "$CONDA_ENV" python api/longcat_audiodit_api.py
       └─ conda run -n "$LONGCAT_AUDIODIT_CONDA_ENV" python api/longcat_audiodit_worker.py
```

当前 CONDA_ENV 默认是 moss-soundEffect，而 LONGCAT_AUDIODIT_CONDA_ENV 默认是
LongCat-AudioDiT-3.5B-bf16。也就是说 HTTP 包装器并不在 LongCat Conda 环境
中启动；迁移后新项目应同时承担 HTTP 控制面和 worker，消除这层隐含关系。

### 2.2 路由和端口

| 方法 | 路由 | 作用 | 响应约束 |
| --- | --- | --- | --- |
| GET | /v1/health | 检查路径、worker、模型、CUDA 和运行配置 | JSON，保留现有关键字段 |
| POST | /v1/upload_audio | 上传参考音频和可选逐字稿 | JSON；保存 SHA-256 和 prompt sidecar |
| GET | /v1/check/audio | 检查参考音频是否存在 | JSON；保留 exists、大小、SHA-256、has_prompt_text |
| POST | /v2/synthesize | LongCat 准确 prompt_text 声音克隆 | 成功返回 audio/wav 原始字节 |
| POST | /internal/unload_all | 本机内部兼容路由 | 仅接受本机请求，返回 JSON |

默认端口继续为 8307，并继续支持 HOST、PORT 和现有 LongCat 环境变量覆盖。

### 2.3 请求字段和运行约束

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| text | 是 | 正文；继续进行标题/列表清理和非空校验 |
| audio_path | 是 | /v1/upload_audio 使用的逻辑文件名 |
| prompt_text | 否 | 参考音频逐字稿；未传时从 sidecar 读取，最终仍必须存在 |
| max_chars_per_chunk | 否 | 默认 180；0 表示不分片 |
| pause_ms | 否 | 分片之间静音，默认 250 ms |
| nfe | 否 | 推理步数，默认 16，最小 2 |
| guidance_strength | 否 | 默认 4.0 |
| guidance_method | 否 | cfg 或 apg，默认 apg |
| seed | 否 | 默认 20260614，负数表示随机 |
| duration_scale | 否 | 默认 1.0，必须大于 0 |
| vae_dtype | 否 | float16 或 float32，默认 float16 |

必须保留：

- 参考音频 24 kHz、单声道和准确 prompt_text 要求。
- 模型配置 sampling_rate=24000、latent_hop=2048、max_wav_duration=60。
- 请求期间加载模型；请求结束显式清理 CUDA 并退出 worker 进程。
- 所有服务继续共享 GPU_LOCK_FILE，LongCat 不能绕过独占锁。
- LOCAL_FILES_ONLY=1 时继续设置 HF_HUB_OFFLINE=1 和 TRANSFORMERS_OFFLINE=1。
- worker 超时、非零退出、空输出必须转换为可诊断错误，不能返回空 WAV。

## 3. 本机资产和外部仓库审计

### 3.1 必需源码仓库

| 路径 | 是否必需 | 当前证据 | 处理方式 |
| --- | --- | --- | --- |
| /home/muyi086/tts-depency/LongCat-AudioDiT | 是 | 含 audiodit/、inference.py、requirements.txt；commit 12c76b51d2a8aa6b6c9af5b25cd5ff8f7aa8178a | 外部引用并固定 commit |

remote：

```text
https://github.com/meituan-longcat/LongCat-AudioDiT
```

官方源码的推理 import 闭包为：

```text
audiodit/__init__.py
audiodit/configuration_audiodit.py
audiodit/modeling_audiodit.py
```

worker 通过 maybe_add_repo_path() 将该仓库加入 sys.path；不要把 audiodit
误写成一个 PyPI 包名。

### 3.2 不需要额外克隆的仓库

本次 LongCat 推理没有发现需要从 /home/muyi086/tts-depency 额外引入的 vllm、
flash-attention、CosyVoice、FunASR、MOSS 或 stable-audio-3 仓库。它们可能
属于本机其他模型环境，但不属于当前 LongCat import 路径。

### 3.3 模型和 tokenizer

```text
模型目录：$HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16
模型文件：config.json、model.safetensors（约 7.2 GiB）
Tokenizer：$HF_MIRROR_DIR/google/umt5-base（约 21 MiB）
Tokenizer：config.json、special_tokens_map.json、spiece.model、
           tokenizer.json、tokenizer_config.json

config.json:
  model_type:           audiodit
  text_encoder_model:  google/umt5-base
  sampling_rate:       24000
  latent_hop:          2048
  max_wav_duration:    60
  transformers_version: 5.3.0
```

资产继续由 HF_MIRROR_DIR、LONGCAT_AUDIODIT_MODEL_DIR 和
LONGCAT_AUDIODIT_TOKENIZER_PATH 指向，不提交 Git，也不由 uv sync 下载。
资产缺失时才使用 hf-mirror：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"

uv run --project LongCat_AudioDiT_3.5B_bf16 hf download \
  drbaph/LongCat-AudioDiT-3.5B-bf16 \
  --local-dir "$HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16"

uv run --project LongCat_AudioDiT_3.5B_bf16 hf download \
  google/umt5-base \
  --local-dir "$HF_MIRROR_DIR/google/umt5-base"
```

## 4. Conda 环境依赖审计

### 4.1 当前基线

```text
environment: LongCat-AudioDiT-3.5B-bf16
prefix:      /home/muyi086/miniconda3/envs/LongCat-AudioDiT-3.5B-bf16
Python:      3.10.20
Torch:       2.13.0+cu130
CUDA:        torch.version.cuda == 13.0
CUDA usable: torch.cuda.is_available() == True
pip check:   No broken requirements found.
```

### 4.2 worker 直接依赖

这是从当前 worker 和官方 audiodit 代码逐层核对得到的实际推理依赖：

| 分发包 | 当前版本 | 直接用途 | 首轮处理 |
| --- | ---: | --- | --- |
| torch | 2.13.0+cu130 | 模型、CUDA、张量和推理 | 必须；Python 3.12.13 重新验证 |
| transformers | 5.3.0 | AutoTokenizer、模型配置和 AudioDiT 注册 | 必须；与模型 config 对齐 |
| protobuf | 7.35.1 | Transformers UMT5 tokenizer 的 SentencePiece/protobuf 解析回退 | 必须显式锁定 |
| safetensors | 0.8.0 | model.safetensors 权重读取 | 必须/锁定 |
| librosa | 0.11.0 | 参考音频加载和重采样 | 必须 |
| soundfile | 0.14.0 | 输出 WAV 写入 | 必须 |
| numpy | 2.2.6 | 音频、时长估算和分片拼接 | 必须 |
| einops | 0.8.2 | audiodit 张量重排 | 必须 |
| torchaudio | 2.11.0 | 官方 requirements 兼容基线，worker 不直接 import | 首轮保留并做 ABI smoke |

官方 requirements.txt 的下限是：

```text
transformers>=5.3.0
torch>=2.0.0
torchaudio>=2.0.0
safetensors>=0.4.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
einops>=0.8.0
```

不要用这些下限直接解析生产环境；先锁已验证版本，再通过 canary 做版本瘦身。

### 4.3 HTTP 控制面依赖

| 分发包 | 当前版本 | 用途 |
| --- | ---: | --- |
| fastapi | 0.139.0 | HTTP 路由和请求模型 |
| uvicorn | 0.50.0 | ASGI 服务 |
| pydantic | 2.13.4 | 请求校验 |
| python-multipart | 0.0.32 | upload_audio multipart 表单 |

main.py 不得在导入阶段导入 torch、audiodit 或加载权重；重型依赖只在
worker.py 子进程中导入。

### 4.4 当前环境中不应无差别搬迁的包

这些包当前环境确实存在，但不属于 LongCat 首轮 worker/API import 闭包：

```text
accelerate、aiohttp、aliyun-python-sdk-core、aliyun-python-sdk-kms、
antlr4-python3-runtime、datasets、funasr、gradio、gradio-client、
hf-gradio、hydra-core、hyperpyyaml、jiwer、kaldiio、modelscope、
onnxruntime、orjson、oss2、pandas、pillow、pydub、pypinyin、pytest、
rapidfuzz、rich、ruamel.yaml、speechbrain、tensorboardX、tiktoken、
timm、torchvision、umap-learn、utmosv2。
```

scipy、numba、scikit-learn、joblib、pooch、soxr、audioread、decorator、
lazy_loader、msgpack 等由 librosa 的依赖闭包解析；不需要手工照搬当前版本。
torch 的 triton、nvidia-*、cuda-bindings 等由 CUDA wheel 元数据解析，不要从
通用 PyPI 手工混装另一套 CUDA。

### 4.5 Conda native 软件和宿主机约束

conda list 还包含 python=3.10.20、ffmpeg=8.0.1、libsndfile=1.2.2、alsa-lib、
OpenMP、glibc 相关库、X11/字体和音视频库。这些不是 uv Python wheel 依赖：

| 类别 | 处理方式 |
| --- | --- |
| Python | 目标固定 ==3.12.13 |
| PyTorch CUDA wheel | 使用显式 CUDA 13.0 index 并锁定 |
| NVIDIA driver/CUDA | 宿主机提供；用 nvidia-smi 和 torch.cuda.is_available() 验证 |
| ffmpeg/libsndfile | 缺失时在系统包或容器层补充 |
| GUI、字体、桌面库 | LongCat API 首轮不需要 |

## 5. 目标 pyproject.toml

在新目录初始化后，建议采用以下首轮基线。版本来自当前实际可 import 环境；
uv.lock 必须在 Python 3.12.13 目标机重新生成并提交。

```toml
[project]
name = "longcat-audiodit-service"
version = "0.1.0"
description = "Unitale LongCat-AudioDiT-3.5B voice cloning service"
requires-python = "==3.12.13"
dependencies = [
    "fastapi==0.139.0",
    "uvicorn==0.50.0",
    "pydantic==2.13.4",
    "python-multipart==0.0.32",
    "torch==2.13.0",
    "torchaudio==2.11.0",
    "transformers==5.3.0",
    "protobuf==7.35.1",
    "safetensors==0.8.0",
    "librosa==0.11.0",
    "soundfile==0.14.0",
    "numpy==2.2.6",
    "einops==0.8.2",
]

[tool.uv]
package = false
index-strategy = "first-index"

[[tool.uv.index]]
name = "pypi-tuna"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

[[tool.uv.index]]
name = "pytorch-cu130-aliyun"
url = "https://mirrors.aliyun.com/pytorch-wheels/cu130/"
format = "flat"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130-aliyun" }
torchaudio = { index = "pytorch-cu130-aliyun" }
```

注意：

- 如果 CUDA 13.0 索引没有 Python 3.12.13 对应 wheel，不能静默换 CPU 包或另一套
  CUDA；应改用已验证的 CUDA wheel，并重新做模型 canary。
- torchvision 当前环境虽存在，但官方 worker 不 import，首轮不写入依赖。
- torchaudio 是官方兼容基线，但 worker 不直接使用；若与目标 torch ABI 不兼容，
  应先移出它，而不是破坏核心推理栈。
- transformers==5.3.0 不能降级，模型 config 和 audiodit 都依赖其 UMT5/自定义
  模型支持。

## 6. 初始化、安装和锁定

以下命令在仓库根目录执行；本轮只输出计划，不创建 .venv 或下载模型：

```bash
cd /home/muyi086/github/TTS-and-VoiceDesign
uv python install 3.12.13
uv init --bare --python 3.12.13 LongCat_AudioDiT_3.5B_bf16
cd LongCat_AudioDiT_3.5B_bf16
# 用第 5 节内容覆盖 pyproject.toml
printf '3.12.13\n' > .python-version
uv lock
uv sync --locked
```

若旧 uv 不支持 --bare，则执行 uv init LongCat_AudioDiT_3.5B_bf16 后手工
删除模板代码，设置 package=false、requires-python 和第 5 节依赖，再运行
uv lock 与 uv sync --locked。

排查时可使用镜像安装命令，但最终复现以 pyproject.toml/uv.lock 为准：

```bash
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  fastapi==0.139.0 uvicorn==0.50.0 pydantic==2.13.4 \
  python-multipart==0.0.32 transformers==5.3.0 safetensors==0.8.0 \
  librosa==0.11.0 soundfile==0.14.0 numpy==2.2.6 einops==0.8.2

uv pip install --python .venv/bin/python \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu130/ \
  torch==2.13.0 torchaudio==2.11.0
```

最终服务使用 uv run --no-sync --project，禁止启动时联网重新解析依赖。

## 7. main.py、worker.py 和 start.sh 改造方案

### 7.1 main.py

以当前 api/longcat_audiodit_api.py 为行为基线迁移 FastAPI app、CORS、本机
访问限制、上传/check/audio、prompt sidecar、Pydantic schema、health JSON、
默认值和环境变量、WAV 持久化、GPU 锁和错误处理。

main.py 不应依赖 api/ 的隐式工作目录 import。共享代码要么复制为目标项目内的
轻量模块，要么使用明确的包 import；不要因为复用 audio_output.py 而在服务启动
时导入模型依赖。

### 7.2 worker.py

以当前 api/longcat_audiodit_worker.py 为基线，保留 repo/model/tokenizer 检查、
AudioDiTModel.from_pretrained、24 kHz 约束、参考音频处理、文本分片、时长估算、
cfg/apg、NFE、seed、VAE dtype、静音裁剪、WAV 输出和 finally CUDA 清理。

迁移后 worker 不再通过 conda run 启动，而由 runtime.py 使用同一项目的
.venv/bin/python 或 uv run --no-sync --project ... python worker.py。请求 JSON、
输出临时 WAV、超时和进程组终止行为继续保留。

### 7.3 start.sh

增加：

```bash
export LONGCAT_AUDIODIT_PROJECT_DIR="${LONGCAT_AUDIODIT_PROJECT_DIR:-$PROJECT_DIR/LongCat_AudioDiT_3.5B_bf16}"
export LONGCAT_AUDIODIT_RUNTIME="${LONGCAT_AUDIODIT_RUNTIME:-uv}"
```

后续将当前 LongCat 启动行替换为：

```bash
if [[ "$LONGCAT_AUDIODIT_RUNTIME" == "uv" ]]; then
  HOST="$LONGCAT_AUDIODIT_HOST" PORT="$LONGCAT_AUDIODIT_PORT" \
    setsid uv run --no-sync --project "$LONGCAT_AUDIODIT_PROJECT_DIR" \
    python "$LONGCAT_AUDIODIT_PROJECT_DIR/main.py" &
else
  # canary 失败时的旧路径；确认稳定后再删除。
  HOST="$LONGCAT_AUDIODIT_HOST" PORT="$LONGCAT_AUDIODIT_PORT" \
    setsid conda run --no-capture-output -n "$CONDA_ENV" \
    python "$API_DIR/longcat_audiodit_api.py" &
fi
```

旧 LONGCAT_AUDIODIT_CONDA_ENV 第一阶段可以保留以便回滚，但默认路径应明确
显示 uv，并使用已经提交的 uv.lock。

## 8. 验证门槛

### Gate 0：依赖解析

```bash
uv python pin 3.12.13
uv lock
uv sync --locked
uv run python --version
uv run python -m pip check
```

期望 Python 为 3.12.13，无 broken requirements；Torch 必须是 CUDA wheel，
不能是 CPU wheel。

### Gate 1：无模型 import 和 health

```bash
PYTHONPATH="$LONGCAT_AUDIODIT_REPO_PATH" \
  uv run --no-sync python -c \
  'import audiodit, torch, transformers, librosa, soundfile, fastapi; print(torch.__version__, torch.version.cuda)'

HOST=127.0.0.1 PORT=8317 LOCAL_FILES_ONLY=1 \
  uv run --no-sync python LongCat_AudioDiT_3.5B_bf16/main.py
curl --fail http://127.0.0.1:8317/v1/health
```

模型缺失时 health 应报告 model_dir/model_required_files=false，而不是 import
阶段崩溃。

### Gate 2：无 CUDA 契约测试

mock worker/runtime，不下载模型、不调用 CUDA，覆盖上传/覆盖上传、SHA-256、
prompt sidecar、check/audio、字段校验、audio/wav 响应、worker 超时/非零退出/
空输出、临时文件清理、GPU 锁释放和 unload 的本机访问限制。

### Gate 3：真实 LongCat canary

使用现有 prompt 音频，输出到 mktemp 目录，不提交生成 WAV：

```bash
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
PYTHONPATH="$LONGCAT_AUDIODIT_REPO_PATH" \
  uv run --no-sync python LongCat_AudioDiT_3.5B_bf16/worker.py \
  --input-json "$tmp_dir/request.json" --output-wav "$tmp_dir/output.wav"
file "$tmp_dir/output.wav"
uv run --no-sync python -c \
  'import soundfile as sf, sys; x, sr = sf.read(sys.argv[1]); assert sr == 24000; assert x.size; print(sr, x.shape)' \
  "$tmp_dir/output.wav"
```

对比旧/新服务的 HTTP 状态、MIME、24 kHz 单声道非空输出、同 seed 时长/分片和
worker 结束后的显存释放。

### Gate 4：整套服务回归

```bash
bash start.sh
curl --fail http://127.0.0.1:8307/v1/health
uv run --project qwen3_tts python -m unittest discover -s tests -v
```

确认其他服务与 LongCat 共享 GPU 锁时没有互相破坏；真实 canary 通过后才删除
旧 Conda 分支。

## 9. 回滚和安全边界

- 不覆盖或删除旧 API/worker、模型、参考音频、tempAudio 和运行缓存；迁移期保留。
- 先用 LONGCAT_AUDIODIT_RUNTIME=uv 显式切换，失败时设置为 conda 回退，不改变
  端口和 WebUI 请求。
- 不提交 .venv、缓存、模型权重、上传音频、生成 WAV、HF_MODULES_CACHE 或绝对
  机器路径。
- 不把 MIMO_API_KEY、部署密钥或凭据写入 pyproject.toml、lock 或计划。
- 外部仓库先固定 commit；上游更新必须重新跑 import、health 和真实 canary。

## 10. 审计复现命令和当前快照

```bash
conda list -n LongCat-AudioDiT-3.5B-bf16
conda env export -n LongCat-AudioDiT-3.5B-bf16 --no-builds
conda run --no-capture-output -n LongCat-AudioDiT-3.5B-bf16 \
  python -m pip list --format=freeze
conda run --no-capture-output -n LongCat-AudioDiT-3.5B-bf16 \
  python -m pip check
conda run --no-capture-output -n LongCat-AudioDiT-3.5B-bf16 \
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
git -C /home/muyi086/tts-depency/LongCat-AudioDiT rev-parse HEAD
```

当前 pip list --format=freeze 的完整 Python 包快照按功能分组如下。第一组是目标基线；其余
包是当前环境存在但不应无差别搬迁的残留或传递依赖：

```text
# 目标基线
einops==0.8.2
fastapi==0.139.0
librosa==0.11.0
numpy==2.2.6
pydantic==2.13.4
python-multipart==0.0.32
safetensors==0.8.0
soundfile==0.14.0
torch==2.13.0
torchaudio==2.11.0
transformers==5.3.0
uvicorn==0.50.0

# 当前环境额外包
accelerate==1.14.0
aiohappyeyeballs==2.6.2
aiohttp==3.14.1
aiosignal==1.4.0
aliyun-python-sdk-core==2.16.0
aliyun-python-sdk-kms==2.16.5
annotated-doc==0.0.4
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.13.0
async-timeout==5.0.1
attrs==26.1.0
audioread==3.1.0
brotli==1.2.0
certifi==2026.5.20
cffi==2.0.0
charset-normalizer==3.4.7
click==8.4.1
coloredlogs==15.0.1
contourpy==1.3.2
crcmod==1.7
cryptography==49.0.0
cuda-bindings==13.3.1
cuda-pathfinder==1.5.5
cuda-toolkit==13.0.3.0
cycler==0.12.1
datasets==5.0.0
decorator==5.3.1
dill==0.4.1
editdistance==0.8.1
exceptiongroup==1.3.1
filelock==3.29.4
flatbuffers==25.12.19
fonttools==4.63.0
frozenlist==1.8.0
fsspec==2026.4.0
funasr==1.3.9
gradio==6.19.0
gradio-client==2.5.0
groovy==0.1.2
h11==0.16.0
hf-gradio==0.4.1
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface==0.0.1
huggingface-hub==1.19.0
humanfriendly==10.0
hydra-core==1.3.3
hyperpyyaml==1.2.3
idna==3.18
iniconfig==2.3.0
jaconv==0.5.0
jamo==0.4.1
jieba==0.42.1
Jinja2==3.1.6
jmespath==0.10.0
joblib==1.5.3
jiwer==4.0.0
kaldiio==2.18.1
kiwisolver==1.5.0
lazy-loader==0.5
llvmlite==0.47.0
markdown-it-py==4.2.0
markupsafe==3.0.3
matplotlib==3.10.9
mdurl==4.2.0
modelscope==1.37.1
mpmath==1.3.0
msgpack==1.2.0
multidict==6.7.1
multiprocess==0.70.19
networkx==3.4.2
numba==0.65.1
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
onnxruntime==1.23.2
orjson==3.11.9
oss2==2.19.1
pandas==2.3.3
pillow==12.2.0
packaging==26.0
pip==26.2.1
platformdirs==4.10.0
pluggy==1.6.0
pooch==1.9.0
propcache==0.5.2
protobuf==7.35.1
psutil==7.2.2
pyarrow==24.0.0
pycparser==3.0
pycryptodome==3.23.0
pydub==0.25.1
pydantic_core==2.46.4
pygments==2.20.0
pynndescent==0.6.0
pyparsing==3.3.2
pypinyin==0.55.0
pytest==9.1.1
python-dateutil==2.9.0.post0
pytz==2026.2
pyyaml==6.0.3
rapidfuzz==3.14.5
regex==2026.5.9
requests==2.34.2
rich==15.0.0
ruamel.yaml==0.18.17
ruamel.yaml.clib==0.2.15
safehttpx==0.1.7
scikit-learn==1.7.2
scipy==1.15.3
semantic-version==2.10.0
sentencepiece==0.2.1
setuptools==81.0.0
shellingham==1.5.4
six==1.17.0
soxr==1.1.0
speechbrain==1.1.0
starlette==1.3.1
sympy==1.14.0
tensorboardx==2.6.5
threadpoolctl==3.6.0
tiktoken==0.13.0
timm==1.0.28
tokenizers==0.22.2
tomli==2.4.1
tomlkit==0.14.0
torch-complex==0.4.4
torchvision==0.28.0
triton==3.7.1
typer==0.25.0
tqdm==4.68.2
typing-extensions==4.15.0
typing-inspection==0.4.2
tzdata==2026.2
umap-learn==0.5.12
urllib3==2.7.0
utmosv2==1.3.1.dev0
wheel==0.47.0
xxhash==3.7.0
yarl==1.24.2
```

额外包快照仅用于审计当前环境，不是目标 pyproject.toml 的安装清单。最终以
uv.lock、pip check、无模型契约测试和真实 GPU canary 作为迁移完成依据。

## 11. task26 实施记录：FlashAttention 与 uv 服务

已在 `LongCat_AudioDiT_3.5B_bf16/` 实现 `main.py`、`worker.py`、`runtime.py`
和 `audio_trim.py`，并将 `start.sh` 的 8307 默认入口切换为：

```bash
uv run --no-sync --project "$LONGCAT_AUDIODIT_PROJECT_DIR" \
  python "$LONGCAT_AUDIODIT_PROJECT_DIR/main.py"
```

旧 `api/longcat_audiodit_api.py` 和 `api/longcat_audiodit_worker.py` 未删除，已
标记为 legacy rollback/reference；确认迁移成功前保留。

当前 FlashAttention 审计结果：

| 项目 | 结果 |
| --- | --- |
| LongCat 官方源码是否 import `flash_attn` | 否 |
| 新 uv 环境是否安装 `flash_attn` | 否 |
| 当前 GPU | RTX 4070 Ti SUPER，compute capability 8.9（Ada） |
| 本地仓库 | `/home/muyi086/tts-depency/flash-attention`，commit `c75d019dea9d910312974417bc28f190dfdda6d9` |
| 本地仓库定位 | FlashAttention-4 beta；FlashAttention-3/4 主要面向 Hopper/Blackwell |
| 结论 | LongCat 不需要，不加入依赖，不编译安装 |

新服务的 `/v1/health` 会明确返回：

```json
{
  "available": {"flash_attn": false},
  "runtime": {
    "flash_attention_policy": "not required; official audiodit uses native PyTorch/Transformers attention"
  }
}
```
