# MOSS-SoundEffect -> moss_soundEffect uv 迁移计划

> 原始环境：moss-soundEffect
>
> 目标目录：TTS-and-VoiceDesign/moss_soundEffect/
>
> 目标 Python：3.12.13；评估日期：2026-08-15
>
> 依据：task31.md、项目升级评估.md、api/soundeffect_api.py、
> api/soundeffect_worker.py、api/gpu_runtime.py、start.sh、soundEffect/，
> 实际 Conda 环境和 /home/muyi086/tts-depency。

> task32 implementation status (2026-08-15): `moss_soundEffect/` now contains
> the uv HTTP wrapper, one-shot worker, runtime helpers, README, and mock
> contract tests. `start.sh` defaults port 8311 to this service and retains an
> explicit `MOSS_SOUNDEFFECT_RUNTIME=conda` fallback to the legacy `api/`
> entry points. The legacy files remain until the migration is confirmed.

## 1. 结论

可以按模型创建 moss_soundEffect/，使用 uv init，迁移原 SoundEffect API
wrapper 和一次性 worker，再让 start.sh 通过 uv 启动 8311。这与已完成的
qwen3_tts、dots_tts_soar 迁移边界一致。

但 uv 不能替代 NVIDIA 驱动、CUDA/显存、Linux fcntl 与进程组、nvidia-smi、
系统 ffmpeg、上游源码和外置 Hugging Face 权重。因此“完全复刻”定义为：
用 uv 锁定 Python、HTTP 依赖、推理依赖和上游源码版本，保留宿主机 GPU/FFmpeg
与 hf-mirror 权重，并以 mock 契约测试和真实 GPU canary 证明行为一致；不把
当前 Conda 中无关的 166 个包全部复制。

迁移实现不删除旧 API 或 Conda，不下载或提交约 11G 权重，也不安装
FlashAttention、vLLM、训练或 Gradio demo 依赖。

## 2. 必须保留的服务行为

目标进程链：

~~~
start.sh
  └─ uv run --no-sync --project moss_soundEffect python moss_soundEffect/main.py
       └─ sys.executable moss_soundEffect/worker.py
~~~

| 方法 | 路由 | 约束 |
| --- | --- | --- |
| GET | /v1/health | JSON；保留 code、paths、available、cuda、runtime、last_errors |
| POST | /v1/generate | 成功返回原始 audio/wav 字节 |
| POST | /v2/synthesize | /v1/generate 的兼容别名 |
| POST | /internal/unload_all | 仅本机可调用；无常驻模型时仍返回 JSON |

请求继续支持 prompt、seconds、num_inference_steps、cfg_scale、sigma_shift、
seed、device、torch_dtype；prompt 非空，seconds 范围为 (0, 30]。默认值、
CORS、LOCAL_FILES_ONLY、超时和错误状态不变。WebUI 8311 使用中文
sfx_plan.prompt，8313 才使用 prompt_en，不能改端口、字段或响应类型。

worker 必须继续一请求一进程，使用 start_new_session=True；异常/超时终止整个
进程组，删除临时 JSON/WAV，执行 CUDA cleanup，并在模型加载、生成、退出期间
持有 GPU_LOCK_FILE。TORCHDYNAMO_DISABLE=1 默认保留。

## 3. 当前基线和模型资产

实际检查结果：

~~~
Conda:        moss-soundEffect
Python:       3.12.13
Torch:        2.9.0+cu128, CUDA 12.8
CUDA:         torch.cuda.is_available() == True
MOSS:         moss-soundeffect-v2 0.1.0
Diffusers:    0.37.1
Transformers: 4.57.1
SoundFile:    0.13.1
pip check:    No broken requirements found
GPU:          NVIDIA GeForce RTX 4070 Ti SUPER, compute 8.9, 16376 MiB
~~~

无权重导入 from moss_soundeffect_v2 import MossSoundEffectPipeline 已通过。
模型目录 /home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0 已完整，
关键文件包括 model_index.json、transformer/diffusion_pytorch_model.safetensors、
text_encoder 两个 safetensors、tokenizer/tokenizer.json 和 vae/vae_128d_48k.pth。
model_index.json 表明模型为 1.3B DiT + DAC VAE + Qwen3 text encoder，48 kHz，
最大时长 30 秒。

继续使用：

~~~
MOSS_SOUNDEFFECT_MODEL_DIR=$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0
HF_HOME=$HF_MIRROR_DIR
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
~~~

缺失权重时单独用 hf-mirror 下载；uv sync 不负责权重：

~~~
export HF_ENDPOINT=https://hf-mirror.com
hf download OpenMOSS-Team/MOSS-SoundEffect-v2.0 \
  --local-dir $HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0
~~~

## 4. 依赖清单

### 4.1 HTTP 控制面

~~~
fastapi==0.139.0
uvicorn==0.51.0
pydantic==2.13.4
starlette==1.3.1
~~~

fcntl、json、os、pathlib、shutil、signal、subprocess、tempfile、threading、
time、typing 是标准库。health 只用 nvidia-smi，API 不应 import Torch。当前
专用 API 不上传文件，因此 python-multipart 不是直接依赖。

### 4.2 worker/模型推理

以下版本来自上游
/home/muyi086/tts-depency/MOSS-TTS/moss_soundeffect_v2/pyproject.toml，
并已在当前环境通过 pip check：

~~~
numpy==1.26.4
einops==0.8.2
pillow==12.2.0
tqdm==4.67.3
safetensors==0.7.0
transformers==4.57.1
diffusers==0.37.1
ftfy==6.3.1
regex==2026.4.4
soundfile==0.13.1
imageio==2.37.3
typing-extensions>=4.10
descript-audiotools==0.7.2
torch==2.9.0+cu128
~~~

上游 metadata 还声明 gradio==6.11.0，但它仅供 demo。torchaudio、
torchvision、torchcodec 属于官方 torch-cu128 extra；当前 worker 用 SoundFile
写 WAV，首轮可不装，若要字面复刻官方 extra 再加入。huggingface-hub、
tokenizers、filelock、fsspec 等传递依赖交给 uv lock，不从当前 freeze 的
166 个条目手工复制。系统级还需 Linux/glibc、NVIDIA driver、nvidia-smi 和
ffmpeg（当前机器为 6.1.1）。

## 5. 依赖源码仓库和 FlashAttention 结论

必需仓库：

~~~
路径：/home/muyi086/tts-depency/MOSS-TTS
远端：https://github.com/OpenMOSS/MOSS-TTS
当前 commit：58b20a0d5fcc6766658d50967a90a9d890009a46
需要子目录：moss_soundeffect_v2/
~~~

MossSoundEffectPipeline、Wan Audio DiT、DAC VAE、Qwen3 text encoder、scheduler
和 prompter 都在这里。当前 Conda editable 包 direct URL 指向临时
/tmp/MOSS-TTS-sparse，迁移不能继续依赖它；应使用 MOSS_SOUNDEFFECT_CODE_PATH、
固定 Git commit，或经过审查的相对源码依赖。

MOSS-Audio 是另一套模型，不需要；vllm 不被当前本地 pipeline 使用；LongCat、
Step-Audio-EditX、stable-audio-3 属于其他服务，也不需要。

MOSS v2 对 flash_attn、flash_attn_interface、sageattention 都是可选探测，
缺失时回退 torch.nn.functional.scaled_dot_product_attention。当前三者均不可用
但服务可运行。当前 GPU 为算力 8.9，本地 flash-attention 是 Hopper/Blackwell
路径的 FlashAttention 4 开发代码，所以首轮不安装、不编译、不写入配置。

## 6. 推荐 uv 配置

初始化：

~~~
uv init --name moss-soundeffect-service --python 3.12.13 moss_soundEffect
cd moss_soundEffect
uv python pin 3.12.13
~~~

初始 pyproject.toml：

~~~toml
[project]
name = "moss-soundeffect-service"
version = "0.1.0"
description = "Unitale MOSS-SoundEffect v2 HTTP service"
requires-python = "==3.12.13"
dependencies = [
    "fastapi==0.139.0",
    "uvicorn==0.51.0",
    "pydantic==2.13.4",
    "starlette==1.3.1",
    "numpy==1.26.4",
    "einops==0.8.2",
    "pillow==12.2.0",
    "tqdm==4.67.3",
    "safetensors==0.7.0",
    "transformers==4.57.1",
    "diffusers==0.37.1",
    "ftfy==6.3.1",
    "regex==2026.4.4",
    "soundfile==0.13.1",
    "imageio==2.37.3",
    "typing-extensions>=4.10",
    "descript-audiotools==0.7.2",
    "torch==2.9.0+cu128",
]

[tool.uv]
package = false
index-strategy = "first-index"

[[tool.uv.index]]
name = "aliyun-pypi"
url = "https://mirrors.aliyun.com/pypi/simple"
default = true

[[tool.uv.index]]
name = "aliyun-pytorch-cu128"
url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
format = "flat"
explicit = true

[tool.uv.sources]
torch = { index = "aliyun-pytorch-cu128" }
~~~

如需严格复刻官方 extra，再加入 torchaudio==2.9.0+cu128、
torchvision==0.24.0+cu128、torchcodec==0.8.0+cu128，并将它们也指向
aliyun-pytorch-cu128。默认先保持最小推理环境。

上游源码不应把本机绝对路径写进提交文件：

~~~
export MOSS_SOUNDEFFECT_CODE_PATH=$HOME/tts-depency/MOSS-TTS
uv sync
uv pip install --python .venv/bin/python --no-deps --editable \
  $MOSS_SOUNDEFFECT_CODE_PATH/moss_soundeffect_v2
~~~

若要求 uv.lock 也锁住源码，改用固定 commit 的 Git source 或相对路径依赖后
再 uv lock；不能提交 /home/muyi086 或 /tmp/MOSS-TTS-sparse。

## 7. 后续实施和验收

1. 新建 main.py、worker.py、runtime.py、.python-version、pyproject.toml、
   uv.lock，不改旧服务。
2. 补 tests/test_moss_soundeffect_migration.py，mock worker、超时、非零退出、
   空 WAV、进程组/临时文件清理、GPU lock、路由和 audio/wav 响应；不得下载
   权重或调用 CUDA。
3. 先用临时端口 canary：

~~~
uv sync --project moss_soundEffect --locked
HOST=127.0.0.1 PORT=18311 \
  uv run --project moss_soundEffect python moss_soundEffect/main.py
curl http://127.0.0.1:18311/v1/health
curl -X POST http://127.0.0.1:18311/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"门吱吱作响的声音，刺耳急促","seconds":1}' \
  -o /tmp/moss_soundeffect_uv.wav
ffprobe -v error -show_entries stream=sample_rate,channels,duration \
  -of default=noprint_wrappers=1 /tmp/moss_soundeffect_uv.wav
~~~

必须验证两次生成、48 kHz 单声道、时长上限、离线权重、异常后再次生成、
GPU lock 释放和 API 没有常驻模型。

4. canary 通过后才修改 start.sh，并保留
   MOSS_SOUNDEFFECT_RUNTIME=conda bash start.sh 回退。只有新旧服务在相同
   fixture 下通过 WebUI 顺序生成、health JSON、WAV 格式、错误状态、GPU
   生命周期和完整 unittest 后，才删除旧 API 与 Conda 环境。

最终验收：

- [ ] Python 3.12.13；uv sync --locked、pip check、完整 unittest 通过。
- [ ] FastAPI/Uvicorn/Pydantic、Torch CUDA、MOSS pipeline 导入成功。
- [ ] 8311、四个路由、字段、CORS、health JSON、audio/wav 不变。
- [ ] 一请求一 worker；成功、异常、超时、取消均清理进程和临时文件。
- [ ] 输出为 48 kHz 单声道、最长 30 秒，离线模式不下载模型。
- [ ] 至少两次真实 GPU canary 和无模型 mock 契约测试通过。
- [ ] 模型、缓存、音频和机器绝对路径未进入 Git。

参考证据：项目升级评估.md、api/soundeffect_api.py、api/soundeffect_worker.py、
api/gpu_runtime.py、start.sh、soundEffect/README.md、
soundEffect/test_moss_soundeffect_v2.py、上游 MOSS-TTS/moss_soundeffect_v2/
pyproject.toml、模型 model_index.json、conda env export 和 pip check。
