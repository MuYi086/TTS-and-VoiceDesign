# ACE-Step 1.5 XL Turbo Diffusers 接入 Unitale 有声书工作流实施方案

> 适用仓库：
>
> - `MuYi086/TTS-and-VoiceDesign`
> - `MuYi086/TTS-Studio-WebUI`
>
> 目标模型：
>
> - `ACE-Step/acestep-v15-xl-turbo-diffusers`
> - 本地权重：`~/hf-mirror/ACE-Step/acestep-v15-xl-turbo-diffusers`
>
> 目标硬件：
>
> - NVIDIA RTX 4070 Ti Super 16GB
> - 48GB 系统内存
>
> 文档基于两个仓库在 2026-08-18 的当前结构编写。

---

## 1. 结论先行

这次接入**不要把 ACE-Step 塞进现有 Stable Audio 3 Medium 的 8311 服务**，也不要让模型常驻 GPU。

推荐保持你当前后端的架构原则：

```text
轻量 HTTP API 常驻
        ↓
获取仓库级 GPU_LOCK_FILE
        ↓
每个请求启动一次性 ACE-Step worker
        ↓
worker 加载模型并生成 WAV
        ↓
worker 清理 CUDA + 进程退出
        ↓
显存彻底释放
        ↓
返回 audio/wav
```

新增服务建议：

```text
8311  Stable Audio 3 Medium   → ambience / texture / 辅助音乐
8312  MOSS-SoundEffect v2     → 明确事件型 SFX
8313  ACE-Step 1.5 XL Turbo   → 正式 BGM / OST / underscore
```

前端不要把 ACE-Step 注册到 `js/soundeffect-client.js`。

新增：

```text
js/bgm-client.js
```

并将当前“背景音乐管理”升级为：

```text
本地导入 BGM
+
AI 生成 BGM
```

ACE-Step 生成成功后，不创建一套新的资产系统，而是**直接写入当前已有的 `bgmLibrary + IndexedDB assets`**。

这样现有能力可以全部复用：

- BGM 试听
- trimStart / trimEnd
- BGM 默认音量
- 时间轴 `play / stop`
- 循环播放
- 2 秒淡入 / 淡出
- 完整工程导出
- 完整工程恢复
- WAV 最终混音

第一阶段因此**不建议修改 `schemaVersion: 4`**。

---

# 2. 为什么采用这个方案

## 2.1 当前 TTS-and-VoiceDesign 已经有正确的模型生命周期

当前后端已经明确采用：

```text
一个服务一个 uv project
+
HTTP API 不加载重模型
+
一次性 worker
+
GPU_LOCK_FILE 串行 GPU 使用
+
worker 退出释放显存
```

Stable Audio 3 Medium 当前就是最适合复制的模板：

```text
stable_audio_3_medium/
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
├── runtime.py
├── worker.py
├── tests/
└── uv.lock
```

ACE-Step 建议按照同样的生命周期实现。

**不要改造成常驻 Pipeline。**

原因不是代码洁癖，而是你的机器只有 16GB VRAM。

ACE-Step XL Turbo Diffusers 仓库约 11.1GB；官方 ACE-Step 对 XL 系列的说明是 XL DiT 权重显存规模约 9GB，16–20GB 显卡需要 CPU offload。你的 16GB 卡属于“可以运行 XL，但必须认真管理显存”的区间。

因此：

```text
模型效果优先：
ACE-Step XL Turbo

稳定性保证：
CPU offload + 一次性 worker + GPU lock
```

是比常驻模型更符合当前项目的方案。

---

## 2.2 当前 WebUI 已经拥有完整 BGM 播放和混音链路

当前 `TTS-Studio-WebUI` 已经存在：

```text
bgmLibrary
bgmForm
handleBgmFileUpload()
saveBgm()
editBgm()
deleteBgm()
addBgmBlock()
```

工程 BGM 核心字段已经是：

```js
{
    id,
    name,
    description,
    filename,
    assetKey,
    trimStart,
    trimEnd,
    volume,
    enabled
}
```

生成式 BGM 根本没有必要创造第二套结构。

正确方案是：

```text
ACE-Step 返回 Blob
        ↓
createAssetKey('bgm', filename)
        ↓
saveAssetToDB()
        ↓
registerLocalAsset()
        ↓
bgmLibrary.push(...)
        ↓
loadAudioBuffer()
        ↓
triggerAutoSave()
```

从这一刻开始，AI 生成的 BGM 与用户自己上传的 WAV 对 WebUI 来说没有区别。

---

# 3. 最终目标架构

```text
┌──────────────────────────────────────────┐
│          TTS-Studio-WebUI                │
│                                          │
│ 小说 → LLM导演 → 台词 / BGM控制 / SFX   │
│                                          │
│ BGM资源库                                │
│ ├── 本地导入                             │
│ └── AI生成 ──────────────┐               │
└──────────────────────────│───────────────┘
                           │
                           │ POST
                           ▼
               http://127.0.0.1:8313
               /v1/aceStep/bgm
                           │
                           ▼
┌──────────────────────────────────────────┐
│       TTS-and-VoiceDesign                │
│                                          │
│ ace_step_1_5/main.py                     │
│       ↓                                  │
│ GPU_LOCK_FILE                            │
│       ↓                                  │
│ ace_step_1_5/worker.py                   │
│       ↓                                  │
│ AceStepPipeline                          │
│       ↓                                  │
│ BF16 + CPU Offload + VAE Tiling          │
│       ↓                                  │
│ 48kHz Stereo WAV                         │
└───────────────────┬──────────────────────┘
                    │
                    ▼
             浏览器 Audio Blob
                    │
                    ▼
               IndexedDB
                    │
                    ▼
               bgmLibrary
                    │
                    ▼
             Script Timeline
                    │
                    ▼
          dialogue + SFX + BGM
                    │
                    ▼
              WAV Offline Mix
```

---

# 4. 模型目录检查

你已经下载：

```bash
~/hf-mirror/ACE-Step/acestep-v15-xl-turbo-diffusers
```

首先检查目录：

```bash
cd ~/hf-mirror/ACE-Step/acestep-v15-xl-turbo-diffusers

find . -maxdepth 2 -type f | sort | head -100
```

Diffusers 版本至少应该存在这些顶层组件：

```text
model_index.json
silence_latent.pt

condition_encoder/
scheduler/
text_encoder/
tokenizer/
transformer/
vae/
```

不要在代码里只检查某一个固定的：

```text
diffusion_pytorch_model.safetensors
```

因为未来模型可能改为 safetensors 分片。

建议检查逻辑：

```python
REQUIRED_PATHS = (
    "model_index.json",
    "condition_encoder",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)
```

并额外检查：

```python
list((model_dir / "transformer").glob("*.safetensors"))
```

至少存在一个权重文件。

---

# 5. 后端改造：新增 ace_step_1_5 独立服务

## 5.1 新目录

在 `TTS-and-VoiceDesign` 根目录新增：

```text
ace_step_1_5/
├── .python-version
├── README.md
├── main.py
├── runtime.py
├── worker.py
├── pyproject.toml
├── uv.lock
└── tests/
    ├── test_api.py
    └── test_runtime.py
```

建议目录叫：

```text
ace_step_1_5
```

而不是：

```text
acestep-v15-xl-turbo-diffusers
```

原因：

模型版本属于配置，服务目录属于业务能力。

未来即使换：

```text
2B Turbo
XL SFT
新的 XL checkpoint
```

API 服务目录仍然无需改名。

---

# 6. Python 环境

当前项目服务统一使用：

```text
Python 3.12.13
```

ACE-Step 也保持一致。

`ace_step_1_5/.python-version`：

```text
3.12.13
```

---

# 7. pyproject.toml 建议

不要一开始把 ACE-Step 安装进 Stable Audio 的环境。

建立独立 uv project。

可以先复制：

```text
stable_audio_3_medium/pyproject.toml
```

然后调整为：

```toml
[project]
name = "ace-step-1-5-service"
version = "0.1.0"
description = "Unitale ACE-Step 1.5 BGM service"
requires-python = "==3.12.13"

dependencies = [
    "fastapi",
    "starlette",
    "uvicorn",
    "pydantic",

    "accelerate",
    "diffusers",
    "transformers",
    "safetensors",
    "soundfile",

    "numpy",
    "torch==2.7.1",
    "torchaudio==2.7.1",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff",
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

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]
```

这里**不建议直接照抄一个未经验证的 diffusers 版本号**。

先：

```bash
cd TTS-and-VoiceDesign/ace_step_1_5

uv sync
```

然后验证：

```bash
uv run python - <<'PY'
from diffusers import AceStepPipeline
print(AceStepPipeline)
PY
```

如果能成功：

```text
from diffusers import AceStepPipeline
```

再执行：

```bash
uv lock
```

并提交 `uv.lock`。

如果当前 PyPI 镜像中的 Diffusers 尚未包含 `AceStepPipeline`，再考虑官方 Diffusers Git 版本；不要第一步就把整个 ACE-Step 原仓库作为运行依赖。

---

# 8. 16GB 显存的运行策略

这是整个接入最重要的部分。

你的默认配置应该是：

```text
dtype                = bfloat16
batch                = 1
steps                = 8
offload              = model
VAE tiling           = true
pipeline resident    = false
```

即：

```python
pipe = AceStepPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()
```

注意：

**使用 `enable_model_cpu_offload()` 时不要再执行：**

```python
pipe.to("cuda")
```

否则失去 CPU offload 的主要意义。

---

## 8.1 OOM 降级顺序

默认：

```text
ACESTEP_OFFLOAD=model
```

如果实际长音乐仍发生 OOM：

```text
ACESTEP_OFFLOAD=sequential
```

对应：

```python
pipe.enable_sequential_cpu_offload()
```

Sequential offload 更省显存，但 CPU/GPU 数据搬运会更多。

对你的机器建议顺序：

```text
① model CPU offload
↓
OOM
② sequential CPU offload
↓
仍不理想
③ 缩短单次 BGM duration
↓
仍不理想
④ 后续增加 2B Turbo 作为 Production 模型
```

不要第一步就量化 Diffusers XL checkpoint。

---

# 9. 新增环境变量

在 `start.sh` 顶部项目目录区域增加：

```bash
ACESTEP_PROJECT_DIR="${ACESTEP_PROJECT_DIR:-$PROJECT_DIR/ace_step_1_5}"
```

模型路径：

```bash
export ACESTEP_MODEL_DIR="${ACESTEP_MODEL_DIR:-$HF_MIRROR_DIR/ACE-Step/acestep-v15-xl-turbo-diffusers}"
```

输出目录：

```bash
export BGM_STORAGE_DIR="${BGM_STORAGE_DIR:-$STORAGE_DIR/bgm}"
export ACESTEP_OUTPUT_DIR="${ACESTEP_OUTPUT_DIR:-$BGM_STORAGE_DIR}"
```

运行参数：

```bash
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-cuda}"
export ACESTEP_DTYPE="${ACESTEP_DTYPE:-bfloat16}"

export ACESTEP_OFFLOAD="${ACESTEP_OFFLOAD:-model}"
export ACESTEP_VAE_TILING="${ACESTEP_VAE_TILING:-1}"

export ACESTEP_DEFAULT_SECONDS="${ACESTEP_DEFAULT_SECONDS:-60}"
export ACESTEP_DEFAULT_STEPS="${ACESTEP_DEFAULT_STEPS:-8}"
export ACESTEP_DEFAULT_SEED="${ACESTEP_DEFAULT_SEED:--1}"

export ACESTEP_REQUEST_TIMEOUT="${ACESTEP_REQUEST_TIMEOUT:-1800}"

export ACESTEP_HOST="${ACESTEP_HOST:-$HOST}"
export ACESTEP_PORT="${ACESTEP_PORT:-8313}"
```

并修改目录初始化：

```bash
mkdir -p \
  "$TIMBRE_STORAGE_DIR" \
  "$SOUNDEFFECT_STORAGE_DIR" \
  "$BGM_STORAGE_DIR" \
  "$CLONE_STORAGE_DIR" \
  "$PROMPTS_DIR" \
  "$HF_MODULES_CACHE" \
  "$NUMBA_CACHE_DIR" \
  "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME" \
  "$(dirname "$GPU_LOCK_FILE")"
```

---

# 10. 为什么新增 storage/bgm

当前：

```text
storage/
├── timbre/
├── soundEffect/
├── clone/
└── .cache/
```

建议调整为：

```text
storage/
├── timbre/
├── soundEffect/
├── bgm/
├── clone/
└── .cache/
```

职责清晰：

```text
soundEffect = 短事件型声音
bgm         = 音乐资产
```

不要让：

```text
ACE-Step OST
```

继续写入：

```text
storage/soundEffect/
```

---

# 11. 后端 API 契约

新增：

```http
POST /v1/aceStep/bgm
```

请求：

```json
{
  "prompt": "Dark cinematic ambient underscore, sparse felt piano, low cello drone, restrained dissonant strings, subtle metallic textures, minimal percussion, designed to sit underneath spoken narration",
  "seconds": 60,
  "steps": 8,
  "bpm": 58,
  "keyscale": "D minor",
  "timesignature": "4",
  "seed": -1
}
```

成功：

```http
HTTP/1.1 200 OK
Content-Type: audio/wav
```

响应 Body：

```text
WAV binary
```

---

## 11.1 API 字段建议

| 字段 | 类型 | 默认 | 限制 | 说明 |
|---|---|---:|---:|---|
| `prompt` | string | 必填 | 1–2000 | 建议英文音乐描述 |
| `seconds` | number | 60 | 10–600 | 音乐时长 |
| `steps` | int | 8 | 1–20 | Turbo 默认固定 8 最合理 |
| `bpm` | int/null | null | 30–240 | null 让模型自行判断 |
| `keyscale` | string/null | null | - | 如 `D minor` |
| `timesignature` | string/null | null | - | 如 `4` |
| `seed` | int | -1 | - | -1 表示随机 |

第一阶段**不要暴露**：

```text
guidance_scale
```

因为 XL Turbo 是 guidance-distilled checkpoint，CFG 不是你需要调的生产参数。

第一阶段也**不要暴露歌词**。

对有声小说：

```python
lyrics=""
```

固定纯配乐方向。

---

# 12. main.py 设计

可以直接按照：

```text
stable_audio_3_medium/main.py
```

的生命周期复制。

核心结构：

```python
#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from runtime import (
    WorkerConfig,
    cuda_status,
    gpu_runtime_lock,
    module_available,
    persist_audio_bytes,
    run_local_worker,
)


PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent

HF_MIRROR_DIR = Path(
    os.path.expanduser(os.getenv("HF_MIRROR_DIR", "~/hf-mirror"))
).resolve()

STORAGE_DIR = Path(
    os.path.expanduser(os.getenv("STORAGE_DIR", str(PROJECT_ROOT / "storage")))
).resolve()

BGM_STORAGE_DIR = Path(
    os.path.expanduser(os.getenv("BGM_STORAGE_DIR", str(STORAGE_DIR / "bgm")))
).resolve()

ACESTEP_MODEL_DIR = Path(
    os.path.expanduser(
        os.getenv(
            "ACESTEP_MODEL_DIR",
            str(HF_MIRROR_DIR / "ACE-Step/acestep-v15-xl-turbo-diffusers"),
        )
    )
).resolve()

RUNTIME_CACHE_DIR = Path(
    os.path.expanduser(
        os.getenv(
            "RUNTIME_CACHE_DIR",
            str(STORAGE_DIR / ".cache/runtime"),
        )
    )
).resolve()

GPU_LOCK_FILE = Path(
    os.path.expanduser(
        os.getenv(
            "GPU_LOCK_FILE",
            str(RUNTIME_CACHE_DIR / "gpu-runtime.lock"),
        )
    )
).resolve()

ACESTEP_OUTPUT_DIR = Path(
    os.path.expanduser(
        os.getenv("ACESTEP_OUTPUT_DIR", str(BGM_STORAGE_DIR))
    )
).resolve()

API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8313"))

ACESTEP_DTYPE = os.getenv("ACESTEP_DTYPE", "bfloat16")
ACESTEP_OFFLOAD = os.getenv("ACESTEP_OFFLOAD", "model")
ACESTEP_VAE_TILING = os.getenv("ACESTEP_VAE_TILING", "1") == "1"

ACESTEP_DEFAULT_SECONDS = float(
    os.getenv("ACESTEP_DEFAULT_SECONDS", "60")
)
ACESTEP_DEFAULT_STEPS = int(
    os.getenv("ACESTEP_DEFAULT_STEPS", "8")
)
ACESTEP_DEFAULT_SEED = int(
    os.getenv("ACESTEP_DEFAULT_SEED", "-1")
)
ACESTEP_REQUEST_TIMEOUT = float(
    os.getenv("ACESTEP_REQUEST_TIMEOUT", "1800")
)

CUDA_RELEASE_DELAY = float(
    os.getenv("CUDA_RELEASE_DELAY", "2.0")
)

WORKER_SCRIPT = PROJECT_DIR / "worker.py"
WORKER_TMP_DIR = RUNTIME_CACHE_DIR / "ace_step_1_5_worker"

MAX_SECONDS = 600.0
MIN_SECONDS = 10.0
```

---

## 12.1 请求模型

```python
class AceStepBgmRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)

    seconds: float = Field(
        default=ACESTEP_DEFAULT_SECONDS,
        ge=MIN_SECONDS,
        le=MAX_SECONDS,
    )

    steps: int = Field(
        default=ACESTEP_DEFAULT_STEPS,
        ge=1,
        le=20,
    )

    bpm: Optional[int] = Field(
        default=None,
        ge=30,
        le=240,
    )

    keyscale: Optional[str] = Field(
        default=None,
        max_length=64,
    )

    timesignature: Optional[str] = Field(
        default=None,
        max_length=16,
    )

    seed: int = Field(
        default=ACESTEP_DEFAULT_SEED,
    )

    @field_validator("prompt")
    @classmethod
    def trim_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt 不能为空")
        return value
```

---

# 13. health API

继续保持整个后端统一协议：

```http
GET /v1/health
```

不要在 health 中加载 ACE-Step。

只做：

```text
nvidia-smi
Python 模块检测
模型目录检测
模型组件目录检测
worker.py 检测
```

示例：

```python
def model_status() -> dict:
    required = {
        "model_index.json": (ACESTEP_MODEL_DIR / "model_index.json").is_file(),
        "transformer": (ACESTEP_MODEL_DIR / "transformer").is_dir(),
        "vae": (ACESTEP_MODEL_DIR / "vae").is_dir(),
        "text_encoder": (ACESTEP_MODEL_DIR / "text_encoder").is_dir(),
        "tokenizer": (ACESTEP_MODEL_DIR / "tokenizer").is_dir(),
        "condition_encoder": (ACESTEP_MODEL_DIR / "condition_encoder").is_dir(),
        "scheduler": (ACESTEP_MODEL_DIR / "scheduler").is_dir(),
    }

    transformer_weights = list(
        (ACESTEP_MODEL_DIR / "transformer").glob("*.safetensors")
    )

    return {
        "required": required,
        "transformer_weights": bool(transformer_weights),
        "complete": all(required.values()) and bool(transformer_weights),
    }
```

health 返回建议：

```json
{
  "code": 200,
  "paths": {
    "model_dir": "/home/.../hf-mirror/ACE-Step/acestep-v15-xl-turbo-diffusers",
    "output_dir": "/.../storage/bgm",
    "worker_script": "/.../ace_step_1_5/worker.py",
    "gpu_lock_file": "/.../storage/.cache/runtime/gpu-runtime.lock"
  },
  "available": {
    "uv": true,
    "diffusers": true,
    "accelerate": true,
    "torch": true,
    "cuda": true,
    "model_complete": true
  },
  "runtime": {
    "port": 8313,
    "model": "ACE-Step/acestep-v15-xl-turbo-diffusers",
    "dtype": "bfloat16",
    "offload": "model",
    "vae_tiling": true,
    "sample_rate": 48000,
    "channels": 2,
    "min_seconds": 10,
    "max_seconds": 600,
    "default_steps": 8,
    "model_lifecycle": "one request -> one worker -> process exit releases VRAM"
  }
}
```

---

# 14. main.py 的生成路由

```python
@app.post("/v1/aceStep/bgm")
async def generate_bgm(request: AceStepBgmRequest) -> Response:
    with gpu_runtime_lock(
        GPU_LOCK_FILE,
        "ace_step_1_5/generate",
    ):
        with manager.lock:
            try:
                payload = manager.build_worker_payload(request)

                audio = manager.run_worker(payload)

                saved = persist_audio_bytes(
                    audio,
                    "ace_step_1_5",
                    ACESTEP_OUTPUT_DIR,
                )

                print(
                    f"[ACE-Step] 已保存 BGM: {saved}"
                )

                return Response(
                    content=audio,
                    media_type="audio/wav",
                )
            except Exception as exc:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=str(exc),
                ) from exc
            finally:
                if CUDA_RELEASE_DELAY > 0:
                    time.sleep(CUDA_RELEASE_DELAY)
```

---

# 15. worker payload

`main.py` 只组装数据：

```python
def build_worker_payload(
    self,
    request: AceStepBgmRequest,
) -> dict:
    return {
        "prompt": request.prompt,
        "seconds": request.seconds,
        "steps": request.steps,
        "bpm": request.bpm,
        "keyscale": request.keyscale,
        "timesignature": request.timesignature,
        "seed": request.seed,

        "model_path": str(ACESTEP_MODEL_DIR),

        "dtype": ACESTEP_DTYPE,
        "offload": ACESTEP_OFFLOAD,
        "vae_tiling": ACESTEP_VAE_TILING,

        "local_files_only": True,
        "runtime_cache_dir": str(RUNTIME_CACHE_DIR),
        "hf_mirror_dir": str(HF_MIRROR_DIR),
    }
```

不要让 API 进程 import：

```python
torch
diffusers
transformers
```

这些全部留给 worker。

---

# 16. runtime.py

第一阶段最稳妥的方式：

**复制 `stable_audio_3_medium/runtime.py` 为 `ace_step_1_5/runtime.py`。**

保留：

```text
WorkerConfig
module_available
cuda_status
process_is_running
terminate_process_group
worker_error_excerpt
gpu_runtime_lock
persist_audio_bytes
```

修改 `run_local_worker()` 里的 Python 环境变量：

原：

```python
os.environ.get(
    "STABLE_AUDIO_3_MEDIUM_PYTHON",
    sys.executable,
)
```

改为：

```python
os.environ.get(
    "ACESTEP_PYTHON",
    sys.executable,
)
```

临时文件前缀：

```text
ace_step_1_5_req_
ace_step_1_5_out_
```

第一阶段不建议急着把 Stable Audio 和 ACE-Step 的 runtime 抽成公共库。

原因：

```text
先稳定接入
>
先做公共抽象
```

等两个服务都稳定之后再提取：

```text
shared/worker_runtime.py
```

风险更小。

---

# 17. worker.py：核心推理实现

这是最重要的文件。

核心版本建议：

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import secrets
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one ACE-Step BGM generation request"
    )

    parser.add_argument(
        "--input-json",
        required=True,
    )

    parser.add_argument(
        "--output-wav",
        required=True,
    )

    return parser.parse_args()


def read_payload(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Worker input must be a JSON object")

    return payload
```

---

## 17.1 Worker 初始化缓存

继续沿用你 Stable Audio 的实践：

```python
def configure_runtime_cache(payload):
    cache_dir = payload.get("runtime_cache_dir")

    if cache_dir:
        cache_dir = Path(cache_dir).expanduser()

        os.environ.setdefault(
            "HF_MODULES_CACHE",
            str(cache_dir / "hf_modules"),
        )

        os.environ.setdefault(
            "XDG_CACHE_HOME",
            str(cache_dir / "xdg"),
        )

    hf_mirror_dir = payload.get("hf_mirror_dir")

    if hf_mirror_dir:
        os.environ.setdefault(
            "HF_HOME",
            str(Path(hf_mirror_dir).expanduser()),
        )

    if payload.get("local_files_only", True):
        os.environ.setdefault(
            "HF_HUB_OFFLINE",
            "1",
        )

        os.environ.setdefault(
            "TRANSFORMERS_OFFLINE",
            "1",
        )

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )

    os.environ.setdefault(
        "CUDA_MODULE_LOADING",
        "LAZY",
    )
```

---

# 18. Pipeline 加载

```python
def load_pipeline(
    model_path: Path,
    dtype_name: str,
    offload: str,
    vae_tiling: bool,
):
    import torch
    from diffusers import AceStepPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ACE-Step requires CUDA on this deployment"
        )

    if dtype_name != "bfloat16":
        raise ValueError(
            "当前生产配置只允许 bfloat16"
        )

    pipe = AceStepPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if vae_tiling:
        pipe.vae.enable_tiling()

    if offload == "model":
        pipe.enable_model_cpu_offload()

    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()

    elif offload == "none":
        pipe.to("cuda")

    else:
        raise ValueError(
            f"Unsupported offload mode: {offload}"
        )

    pipe.set_progress_bar_config(disable=True)

    return pipe, torch
```

对于你的 4070 Ti Super：

```text
offload=model
```

是首选。

---

# 19. 生成音乐

```python
def run_generation(
    pipe,
    torch,
    payload,
):
    seed = int(payload.get("seed", -1))

    if seed < 0:
        seed = secrets.randbelow(2**31 - 1)

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(seed)

    kwargs = {
        "prompt": payload["prompt"],
        "lyrics": "",
        "audio_duration": float(
            payload["seconds"]
        ),
        "num_inference_steps": int(
            payload.get("steps", 8)
        ),
        "generator": generator,
    }

    bpm = payload.get("bpm")
    keyscale = payload.get("keyscale")
    timesignature = payload.get("timesignature")

    if bpm is not None:
        kwargs["bpm"] = int(bpm)

    if keyscale:
        kwargs["keyscale"] = str(keyscale)

    if timesignature:
        kwargs["timesignature"] = str(
            timesignature
        )

    with torch.inference_mode():
        result = pipe(**kwargs)

    return result, seed
```

重要：

```python
lyrics=""
```

而不是生成歌词。

你的模型职责是：

```text
有声小说纯背景配乐
```

不是 Song Generation。

---

# 20. 保存 48kHz Stereo WAV

官方 Diffusers ACE-Step 输出：

```text
48 kHz stereo
```

保存：

```python
import soundfile as sf

audio = result.audios[0]

if hasattr(audio, "detach"):
    audio = (
        audio
        .detach()
        .cpu()
        .float()
        .numpy()
    )

# ACE-Step 通常为:
#
# [channels, samples]
#
# soundfile 需要:
#
# [samples, channels]

if audio.ndim == 2:
    audio = audio.T

sf.write(
    str(output_path),
    audio,
    pipe.sample_rate,
    subtype="FLOAT",
)
```

不要在后端先转成：

```text
44.1 kHz
```

保持 ACE-Step 原生 48kHz。

你的 Web Audio 最终导出现在会统一进入 OfflineAudioContext，浏览器负责重采样。

后续如果要做专业母带，可以再把整个最终工程统一提升到：

```text
48kHz OfflineAudioContext
```

但**不要把这个变化和 ACE-Step 第一阶段接入绑在一起**。

---

# 21. Worker 显存清理

```python
def clear_cuda(torch):
    gc.collect()

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
```

worker：

```python
pipe = None
result = None

try:
    pipe, torch = load_pipeline(...)

    result, resolved_seed = run_generation(
        pipe,
        torch,
        payload,
    )

    save_audio(...)

finally:
    if result is not None:
        del result

    if pipe is not None:
        del pipe

    clear_cuda(torch)
```

真正保证释放的最后一道保险不是：

```python
torch.cuda.empty_cache()
```

而是：

```text
worker process exit
```

这也是为什么要保持你现有的一次性 worker 架构。

---

# 22. 完整 worker 主流程

伪代码：

```python
def main():
    args = parse_args()

    payload = read_payload(
        args.input_json
    )

    configure_runtime_cache(
        payload
    )

    model_path = Path(
        payload["model_path"]
    ).expanduser().resolve()

    validate_model(
        model_path
    )

    output_path = Path(
        args.output_wav
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipe = None
    result = None
    torch = None

    try:
        pipe, torch = load_pipeline(
            model_path=model_path,
            dtype_name=payload["dtype"],
            offload=payload["offload"],
            vae_tiling=bool(
                payload["vae_tiling"]
            ),
        )

        result, seed = run_generation(
            pipe,
            torch,
            payload,
        )

        write_audio(
            result,
            pipe,
            output_path,
        )

        print(
            f"[ACE-Step] generation done "
            f"seed={seed} "
            f"path={output_path}"
        )

    finally:
        if result is not None:
            del result

        if pipe is not None:
            del pipe

        if torch is not None:
            clear_cuda(torch)


if __name__ == "__main__":
    main()
```

---

# 23. internal/unload_all

继续和其他模型保持兼容：

```http
POST /internal/unload_all
```

但语义应该是：

```json
{
  "code": 200,
  "msg": "ACE-Step 服务无常驻模型；worker 退出后显存已释放。"
}
```

不要尝试从 API 进程操作 Pipeline。

因为 API 进程压根不应该持有 Pipeline。

---

# 24. start.sh 接入

## 24.1 uv sync 列表

README 和安装脚本中的：

```bash
for project in ...
```

加入：

```text
ace_step_1_5
```

例如：

```bash
for project in \
  qwen3_tts \
  mimo_tts \
  voxcpm2 \
  LongCat_AudioDiT_3.5B_bf16 \
  dots_tts_soar \
  moss_soundEffect \
  stable_audio_3_medium \
  ace_step_1_5 \
  qwen3_voiceDesign \
  moss_voiceGenerator \
  Step_Audio_EditX
do
    uv sync \
      --project "$project" \
      --locked
done
```

---

## 24.2 启动服务

在 Stable Audio 后增加：

```bash
HOST="$ACESTEP_HOST" \
PORT="$ACESTEP_PORT" \
setsid uv run \
  --no-sync \
  --project "$ACESTEP_PROJECT_DIR" \
  python "$ACESTEP_PROJECT_DIR/main.py" &

acestep_pid=$!
```

---

## 24.3 PID 初始化

增加：

```bash
acestep_pid=""
```

---

## 24.4 cleanup

所有 PID 列表加入：

```bash
"$acestep_pid"
```

包括：

```text
SIGTERM
SIGKILL
wait
```

三个位置都要加。

---

## 24.5 wait -n

增加：

```bash
"$acestep_pid"
```

最终类似：

```bash
wait -n \
  "$main_pid" \
  "$mimo_tts_pid" \
  "$soundeffect_pid" \
  "$stable_audio_3_medium_pid" \
  "$acestep_pid" \
  "$qwen3_tts_pid" \
  ...
```

---

# 25. README 服务总览更新

增加：

| 服务 | 端口 | 主要用途 | 主要路由 |
|---|---:|---|---|
| ACE-Step 1.5 XL Turbo | 8313 | 有声小说 BGM / OST | `/v1/aceStep/bgm` |

服务职责明确写：

```text
Stable Audio 3 Medium：
文本生成音乐或声效，当前主要承担 ambience / texture / 辅助声音。

MOSS-SoundEffect：
短时非语言事件音效。

ACE-Step：
有声小说配乐、主题音乐、场景 underscore、情绪 cue。
```

---

# 26. 后端 smoke test

先不要改 WebUI。

确认后端独立可用。

启动：

```bash
HOST=127.0.0.1 \
PORT=8313 \
ACESTEP_MODEL_DIR="$HOME/hf-mirror/ACE-Step/acestep-v15-xl-turbo-diffusers" \
ACESTEP_OFFLOAD=model \
uv run \
  --project ace_step_1_5 \
  python ace_step_1_5/main.py
```

health：

```bash
curl \
  http://127.0.0.1:8313/v1/health
```

---

## 26.1 第一条 10 秒测试

```bash
curl \
  -X POST \
  http://127.0.0.1:8313/v1/aceStep/bgm \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Dark cinematic ambient underscore, sparse felt piano, low cello drone, subtle dissonance, restrained dynamics, minimal percussion, no vocals, designed underneath narration",
    "seconds": 10,
    "steps": 8,
    "bpm": 58,
    "keyscale": "D minor",
    "timesignature": "4",
    "seed": 42
  }' \
  -o test-ace-step.wav
```

检查：

```bash
ffprobe \
  -hide_banner \
  test-ace-step.wav
```

应重点确认：

```text
WAV
48000 Hz
stereo
```

---

# 27. 显存验证

请求前：

```bash
nvidia-smi
```

请求过程中另一个终端：

```bash
watch -n 1 nvidia-smi
```

请求结束后：

```bash
nvidia-smi
```

你要验证的不是：

```text
“模型能跑”
```

而是：

```text
① 峰值显存不 OOM
② worker 结束后显存回落
③ 下一次 VoxCPM2 可以正常抢到 GPU
④ GPU_LOCK_FILE 没有死锁
```

这是生产接入是否成功的真正标准。

---

# 28. 长度回归顺序

不要第一次就跑 5 分钟。

建议依次测试：

```text
10 秒
30 秒
60 秒
180 秒
```

每次记录：

```text
峰值 VRAM
系统 RAM
是否 OOM
WAV 是否完整
生成后 VRAM 是否释放
```

如果：

```text
ACESTEP_OFFLOAD=model
```

在较长时长下不稳定，再改：

```bash
export ACESTEP_OFFLOAD=sequential
```

而不是立刻重构代码。

---

# 29. 后端 no-model 单元测试

你的根测试目前遵循：

```text
不加载权重
不需要 CUDA
不联网
```

ACE-Step 应保持一致。

至少测试：

### test_health_does_not_load_model

检查：

```text
GET /v1/health
```

不会 import / instantiate：

```text
AceStepPipeline
```

---

### test_rejects_invalid_duration

例如：

```json
{
  "seconds": 601
}
```

应：

```text
422
```

---

### test_rejects_empty_prompt

```json
{
  "prompt": ""
}
```

应：

```text
422
```

---

### test_build_worker_payload

验证：

```text
model_path
offload
dtype
vae_tiling
seed
steps
```

都正确传给 worker。

---

### test_missing_model

模型目录不存在时：

```text
500
```

并返回可定位错误。

---

# 30. 前端新增 js/bgm-client.js

不要修改：

```text
js/soundeffect-client.js
```

新增：

```text
js/bgm-client.js
```

建议：

```javascript
/**
 * @fileoverview AI BGM 生成客户端。
 */
(function (global) {
    const BGM_MODELS = Object.freeze({
        'ace-step-v15-xl-turbo-diffusers': Object.freeze({
            id: 'ace-step-v15-xl-turbo-diffusers',
            label: 'ACE-Step 1.5 XL Turbo',
            endpoint: 'http://127.0.0.1:8313/v1/aceStep/bgm',
            minSeconds: 10,
            maxSeconds: 600,
            promptLanguage: 'en'
        })
    });

    const DEFAULT_MODEL =
        'ace-step-v15-xl-turbo-diffusers';

    function getBgmModel(model) {
        return (
            BGM_MODELS[model]
            || BGM_MODELS[DEFAULT_MODEL]
        );
    }

    async function generateBgmAudio({
        prompt,
        seconds,
        bpm,
        keyscale,
        timesignature,
        seed = -1,
        steps = 8,
        model = DEFAULT_MODEL,
        signal,
        endpoint
    }) {
        const config =
            getBgmModel(model);

        const normalizedPrompt =
            String(prompt || '').trim();

        if (!normalizedPrompt) {
            throw new Error(
                `${config.label} Prompt 不能为空。`
            );
        }

        const payload = {
            prompt: normalizedPrompt,
            seconds: Number(seconds),
            steps: Number(steps),
            seed: Number(seed)
        };

        if (bpm !== '' && bpm != null) {
            payload.bpm = Number(bpm);
        }

        if (keyscale) {
            payload.keyscale =
                String(keyscale).trim();
        }

        if (timesignature) {
            payload.timesignature =
                String(timesignature).trim();
        }

        const response = await fetch(
            endpoint || config.endpoint,
            {
                method: 'POST',
                headers: {
                    'Content-Type':
                        'application/json'
                },
                body: JSON.stringify(payload),
                signal
            }
        );

        if (!response.ok) {
            const detail =
                await response.text();

            throw new Error(
                `${config.label} 生成失败 ` +
                `(HTTP ${response.status})：` +
                detail.slice(0, 500)
            );
        }

        const contentType =
            response.headers.get(
                'content-type'
            ) || '';

        if (
            !contentType
                .toLowerCase()
                .startsWith('audio/')
        ) {
            throw new Error(
                `${config.label} 未返回音频`
            );
        }

        const blob =
            await response.blob();

        if (!blob.size) {
            throw new Error(
                `${config.label} 返回空音频`
            );
        }

        return blob;
    }

    global.UnitaleBgmClient = {
        BGM_MODELS,
        DEFAULT_MODEL,
        getBgmModel,
        generateBgmAudio
    };
}(window));
```

---

# 31. index.html 引入脚本

在：

```html
<script src="./js/soundeffect-client.js"></script>
```

附近增加：

```html
<script src="./js/bgm-client.js"></script>
```

不要把 BGM 请求代码继续堆到 `index.html`。

---

# 32. BGM UI 改造

当前页面：

```text
背景音乐管理 (BGM)
```

保留本地上传。

在上方增加：

```text
AI 生成 BGM
```

推荐第一阶段 UI：

```text
┌──────────────────────────────────────┐
│ AI 生成 BGM                          │
│                                      │
│ Prompt                               │
│ [................................]   │
│                                      │
│ 时长        BPM                      │
│ [60]        [58]                     │
│                                      │
│ Key         Time Signature           │
│ [D minor]   [4]                      │
│                                      │
│ Seed        Steps                    │
│ [-1]        [8]                      │
│                                      │
│ [生成 BGM] [停止]                    │
└──────────────────────────────────────┘
```

第一阶段不需要：

```text
reference audio
cover
repaint
lego
extract
complete
```

先把 Text-to-Music 主链路跑稳。

---

# 33. Vue 状态

增加：

```javascript
const bgmGenerateForm = ref({
    model:
        'ace-step-v15-xl-turbo-diffusers',

    prompt: '',

    seconds: 60,

    bpm: '',

    keyscale: '',

    timesignature: '4',

    seed: -1,

    steps: 8,

    name: '',

    description: ''
});

const isGeneratingBgm =
    ref(false);

const bgmGenerationError =
    ref('');

let bgmAbortController =
    null;
```

---

# 34. 生成完成后直接进入现有 BGM Library

核心函数：

```javascript
const generateBgm = async () => {
    if (isGeneratingBgm.value) {
        return;
    }

    const prompt =
        bgmGenerateForm.value.prompt.trim();

    if (!prompt) {
        return alert('请输入 BGM Prompt');
    }

    isGeneratingBgm.value = true;
    bgmGenerationError.value = '';

    bgmAbortController =
        new AbortController();

    try {
        const blob =
            await window
                .UnitaleBgmClient
                .generateBgmAudio({
                    prompt,

                    seconds:
                        bgmGenerateForm
                            .value
                            .seconds,

                    bpm:
                        bgmGenerateForm
                            .value
                            .bpm,

                    keyscale:
                        bgmGenerateForm
                            .value
                            .keyscale,

                    timesignature:
                        bgmGenerateForm
                            .value
                            .timesignature,

                    seed:
                        bgmGenerateForm
                            .value
                            .seed,

                    steps:
                        bgmGenerateForm
                            .value
                            .steps,

                    signal:
                        bgmAbortController.signal
                });

        const now =
            Date.now();

        const filename =
            `ace-step-bgm-${now}.wav`;

        const assetKey =
            createAssetKey(
                'bgm',
                filename
            );

        const bgm = {
            id: String(now),

            name:
                bgmGenerateForm
                    .value
                    .name
                || `ACE-Step BGM ${new Date().toLocaleTimeString()}`,

            description:
                bgmGenerateForm
                    .value
                    .description
                || prompt,

            filename,

            assetKey,

            trimStart: 0,
            trimEnd: 1,

            volume: 0.3,

            enabled: true,

            source:
                'ace-step-v15-xl-turbo-diffusers',

            generation: {
                prompt,

                seconds:
                    Number(
                        bgmGenerateForm
                            .value
                            .seconds
                    ),

                bpm:
                    bgmGenerateForm
                        .value
                        .bpm
                    || null,

                keyscale:
                    bgmGenerateForm
                        .value
                        .keyscale
                    || null,

                timesignature:
                    bgmGenerateForm
                        .value
                        .timesignature
                    || null,

                seed:
                    Number(
                        bgmGenerateForm
                            .value
                            .seed
                    ),

                steps:
                    Number(
                        bgmGenerateForm
                            .value
                            .steps
                    )
            }
        };

        await saveAssetToDB(
            assetKey,
            blob
        );

        registerLocalAsset(
            bgm,
            blob,
            filename
        );

        bgmLibrary.value.push(
            bgm
        );

        await loadAudioBuffer(
            bgm
        );

        triggerAutoSave();

    } catch (error) {
        if (
            error?.name ===
            'AbortError'
        ) {
            return;
        }

        bgmGenerationError.value =
            error?.message
            || String(error);

        alert(
            `BGM 生成失败：` +
            bgmGenerationError.value
        );

    } finally {
        isGeneratingBgm.value =
            false;

        bgmAbortController =
            null;
    }
};
```

---

# 35. 停止生成

```javascript
const stopBgmGeneration = () => {
    if (bgmAbortController) {
        bgmAbortController.abort();
    }
};
```

注意：

浏览器取消 HTTP 请求**不等于立即杀死后台 worker**。

为了真正取消 GPU 推理，第二阶段可以让后端把 client disconnect / cancel 映射到 worker 进程组终止。

第一阶段先保证：

```text
前端不再等待响应
```

即可。

不要把“取消传播到 worker”与核心接入一起做，避免一次改动范围过大。

---

# 36. 为什么第一阶段不需要改 project-storage schema

当前 `normalizeLibraryItem()` 的处理方式本质上是：

```javascript
const normalized = {
    ...source
};
```

然后补齐：

```text
id
assetKey
name
description
filename
trimStart
trimEnd
volume
enabled
```

因此额外字段：

```json
{
  "source": "ace-step-v15-xl-turbo-diffusers",
  "generation": {
    "prompt": "...",
    "seconds": 60,
    "bpm": 58,
    "keyscale": "D minor",
    "timesignature": "4",
    "seed": 42,
    "steps": 8
  }
}
```

不会因为 normalize 被主动删除。

所以第一阶段可以继续：

```text
schemaVersion = 4
```

不用为了 ACE-Step 单独升到 5。

---

# 37. 工程导出不需要新增一套逻辑

当前完整工程导出已经：

```text
processLibrary(
    exportEnvelope.libraries.bgm,
    'filename'
)
```

并通过：

```text
assetKey
```

取得 Blob。

只要 ACE-Step 生成成功后执行：

```javascript
await saveAssetToDB(
    assetKey,
    blob
);
```

它就会自动进入完整工程导出。

导入时：

```text
libraries.bgm
```

也已经有恢复逻辑。

因此：

```text
AI 生成 BGM
```

应该复用当前 BGM asset pipeline，而不是再造：

```text
generatedBgmAssets
```

---

# 38. 时间轴也不需要改

ACE-Step 最终仍然只是一个：

```text
BGM Library Item
```

时间轴继续：

```json
{
  "type": "bgm",
  "action": "play",
  "bgmName": "地下室悬疑主题",
  "volume": 0.6
}
```

停止：

```json
{
  "type": "bgm",
  "action": "stop"
}
```

现有 BGM segment 计算：

```text
play
↓
currentBgm
↓
stop / next play
↓
bgmSegments
↓
loop
↓
fade in / fade out
↓
OfflineAudioContext
```

全部可以原样复用。

---

# 39. 当前脚本分析 Prompt 与 ACE-Step 的关系

现在 LLM 的 BGM 逻辑是：

```text
只从已经存在的 bgmLibrary
选择精确 BGM 名称
```

也就是：

```text
小说分析前
↓
必须先存在 BGM 素材
↓
LLM 才能输出 play / stop
```

这套规则第一阶段**继续保持**。

推荐工作流：

```text
小说
↓
人工 / LLM 判断整体风格
↓
ACE-Step 生成 3~6 条候选 BGM
↓
挑选并保存到 bgmLibrary
↓
运行脚本深度分析
↓
LLM 在已有 BGM Library 中选择
↓
输出 BGM play / stop
```

这是当前 WebUI 改动最少、最可靠的流程。

---

# 40. 推荐你第一阶段生成的 BGM 资源结构

一部悬疑短篇不要生成十几首完全不同的音乐。

建议先做：

```text
01_theme_main
主悬疑主题

02_investigation
调查 / 探索铺底

03_tension_low
轻度危险

04_tension_high
强危险 / 高潮

05_aftershock
高潮后压抑

06_outro
结尾
```

然后让当前 LLM 只从这 6 个名字中选择。

这比：

```text
每一个 dialogue
重新生成一首 BGM
```

更专业，也更省 GPU。

---

# 41. Prompt 最佳实践

ACE-Step 官方 Diffusers Pipeline 重点依赖描述式 Prompt。

你的 BGM Prompt 建议始终包含：

```text
① 音乐类型
② 情绪
③ 主要乐器
④ 配器密度
⑤ 节奏
⑥ 动态
⑦ 是否抢人声
⑧ 明确 no vocals
```

例如：

```text
Dark cinematic ambient underscore for a suspense audiobook,
sparse felt piano,
low cello and sub-bass drones,
subtle dissonant strings,
occasional distant metallic textures,
very restrained percussion,
slow evolving tension,
wide atmospheric stereo field,
low dynamic density,
leave generous space for spoken narration,
no vocals,
no choir,
no lead melody dominating the dialogue.
```

不建议：

```text
scary music
```

信息太少。

---

# 42. 有声小说专用 Prompt 约束

可以建立固定后缀：

```text
instrumental cinematic underscore,
designed to sit beneath spoken narration,
restrained dynamics,
no vocals,
no choir,
no spoken words,
no dominant lead melody,
avoid dense percussion.
```

然后让 LLM 只生成前面的：

```text
情绪
乐器
场景
节奏
```

这样可以显著降低 BGM 抢对白的问题。

---

# 43. 第二阶段：自动生成 BGM Plan

第一阶段跑通后，再考虑自动化：

```text
小说
↓
LLM BGM Director
↓
bgm_generation_plan
↓
ACE-Step
↓
bgmLibrary
↓
脚本 BGM play / stop
```

不要立即修改现有：

```text
type: bgm
```

对象去承担生成参数。

推荐将：

```text
“素材”
```

与：

```text
“时间轴控制”
```

继续分开。

---

## 43.1 未来 BGM Plan 示例

```json
{
  "id": "bgm_plan_tension_01",
  "name": "地下室低压悬疑",
  "purpose": "tension_low",
  "prompt_en": "Dark cinematic ambient underscore...",
  "duration_seconds": 180,
  "bpm": 56,
  "keyscale": "D minor",
  "timesignature": "4",
  "seed": -1
}
```

生成后变成：

```text
bgmLibrary item
```

而时间轴仍然只引用：

```json
{
  "type": "bgm",
  "action": "play",
  "bgmName": "地下室低压悬疑"
}
```

这个边界非常重要。

---

# 44. 第三阶段：Reference Audio

ACE-Step Diffusers Pipeline 支持：

```text
reference_audio
```

但不建议第一阶段做。

未来可以增加：

```text
主题音乐
↓
Reference Audio
↓
同主题变奏
```

例如：

```text
Theme A
├── Theme A - Investigation
├── Theme A - Tension
├── Theme A - Climax
└── Theme A - Outro
```

这会比每次纯 Prompt 生成更容易保持整部小说的 OST 一致性。

---

# 45. Reference Audio 后端接口建议

未来不要直接把 Blob 塞到 JSON。

可以新增：

```http
POST /v1/aceStep/upload_reference
```

Multipart：

```text
audio
full_path
```

然后生成：

```json
{
  "prompt": "...",
  "reference_audio_path": "theme-a.wav",
  "seconds": 120
}
```

worker：

```text
读取 WAV
↓
重采样至 48kHz
↓
转 [channels, samples]
↓
reference_audio=
```

第一阶段不要实现。

---

# 46. Stable Audio 和 ACE-Step 的最终职责

不要删 Stable Audio。

推荐：

## ACE-Step

```text
正式 BGM
主题音乐
悬疑 underscore
情绪配乐
高潮音乐
片头
片尾
OST 变奏
```

---

## Stable Audio 3 Medium

```text
dark drone
room tone
rain ambience
industrial ambience
wind texture
forest night ambience
transition texture
```

---

## MOSS-SoundEffect

```text
敲门
脚步
玻璃破碎
开门
关门
物体掉落
机械动作
明确环境事件
```

这样三个模型职责不会重叠。

---

# 47. 前端测试清单

## 47.1 AI BGM 生成

- [ ] 8313 不启动时页面给出明确错误
- [ ] 空 Prompt 不发请求
- [ ] 生成中按钮 disabled
- [ ] 停止按钮可 Abort fetch
- [ ] 返回非 audio MIME 时提示错误
- [ ] 返回空 Blob 时提示错误
- [ ] 成功生成后自动加入 BGM Library
- [ ] 新 BGM 可以立即试听

---

## 47.2 IndexedDB

- [ ] 生成后刷新页面素材仍存在
- [ ] `assetKey` 稳定
- [ ] `filename` 正常
- [ ] `registerLocalAsset` 能命中
- [ ] `loadAudioBuffer` 正常

---

## 47.3 完整工程

- [ ] 生成 ACE-Step BGM
- [ ] 导出完整工程 JSON
- [ ] 清空站点
- [ ] 导入工程
- [ ] BGM Library 恢复
- [ ] BGM 可以试听
- [ ] generation metadata 仍存在

---

## 47.4 时间轴

- [ ] 插入 BGM play
- [ ] 插入 BGM stop
- [ ] 顺序播放正确
- [ ] BGM loop 正常
- [ ] trimStart 正常
- [ ] trimEnd 正常
- [ ] 控制块 volume 正常
- [ ] Library volume 正常

---

## 47.5 WAV 导出

- [ ] Dialogue 正常
- [ ] SoundEffect 正常
- [ ] ACE-Step BGM 正常
- [ ] BGM 淡入正常
- [ ] BGM 淡出正常
- [ ] 切换 BGM 不重叠异常
- [ ] 最终 WAV 无截断
- [ ] 不出现明显爆音

---

# 48. 后端 GPU 回归测试

这是最重要的生产测试。

执行顺序：

```text
ACE-Step
↓
VoxCPM2
↓
MOSS SoundEffect
↓
ACE-Step
↓
Stable Audio
↓
VoxCPM2
```

每次确认：

```bash
nvidia-smi
```

目标：

```text
所有模型都能依次使用同一张 GPU
且前一个 worker 退出后不会持续占显存
```

如果出现：

```text
ACE-Step 完成
↓
VoxCPM2 OOM
```

说明接入不合格。

优先检查：

```text
worker 是否真正退出
是否有子进程残留
GPU_LOCK_FILE 是否提前释放
API 进程是否错误 import 了 torch/diffusers 并创建 CUDA context
```

---

# 49. 建议增加显存日志

worker 加载前：

```python
print(
    "[ACE-Step] before load:",
    torch.cuda.mem_get_info()
)
```

生成后：

```python
print(
    "[ACE-Step] after generation:",
    torch.cuda.mem_get_info()
)
```

API health 继续使用：

```text
nvidia-smi
```

而不是 API 进程调用：

```python
torch.cuda.*
```

避免 health 自己创建 CUDA context。

---

# 50. 不建议做的事情

## 50.1 不要模型常驻

错误：

```python
PIPE = AceStepPipeline.from_pretrained(...)
```

放在：

```text
main.py module global
```

这样会长期占 GPU / RAM，并破坏你当前生命周期。

---

## 50.2 不要复用 Stable Audio worker

不要：

```text
stable_audio_3_medium/worker.py

if model == stable:
    ...
elif model == ace_step:
    ...
```

模型依赖、显存策略、API 参数完全不同。

---

## 50.3 不要塞进 soundeffect-client.js

ACE-Step 是 BGM 模型。

应该：

```text
bgm-client.js
```

---

## 50.4 不要一开始做全自动 BGM Director

先完成：

```text
手动 Prompt
↓
ACE-Step
↓
BGM Library
↓
Timeline
↓
Mix
```

这个链路。

然后再做：

```text
LLM 自动生成 BGM Prompt
```

否则调试时你无法判断问题来自：

```text
LLM
ACE-Step
WebUI
Timeline
Mix
```

中的哪一层。

---

## 50.5 不要一开始上 reference / repaint / cover

第一阶段只实现：

```text
task_type = text2music
```

稳定之后再扩。

---

# 51. 推荐开发顺序

## Phase 0：本地模型验证

```text
□ 检查模型目录
□ uv 环境
□ import AceStepPipeline
□ 本地 10 秒 Python smoke test
```

---

## Phase 1：后端

```text
□ ace_step_1_5/
□ main.py
□ runtime.py
□ worker.py
□ /v1/health
□ /v1/aceStep/bgm
□ GPU lock
□ worker exit
□ storage/bgm
□ start.sh
□ README
□ no-model tests
```

---

## Phase 2：前端最小接入

```text
□ js/bgm-client.js
□ index.html 引入
□ AI BGM Form
□ generateBgm()
□ saveAssetToDB()
□ registerLocalAsset()
□ bgmLibrary
□ 试听
```

---

## Phase 3：工程回归

```text
□ timeline play
□ timeline stop
□ complete project export
□ complete project import
□ WAV mix
```

---

## Phase 4：导演自动化

```text
□ BGM Director Prompt
□ bgm_generation_plan
□ 批量生成
□ 自动命名
□ 自动写 bgmLibrary
□ 自动插 play/stop
```

---

## Phase 5：OST 一致性

```text
□ Reference Audio
□ Theme
□ Variation
□ Climax
□ Outro
```

---

# 52. 推荐的第一版产品交互

背景音乐页最终第一版：

```text
背景音乐管理

┌ AI 生成 ────────────────────────────────┐
│ 名称：地下室悬疑                       │
│                                        │
│ 描述：调查进入地下区域时使用           │
│                                        │
│ Prompt：                               │
│ Dark cinematic ambient...              │
│                                        │
│ Duration  60                           │
│ BPM       58                           │
│ Key       D minor                      │
│ Time      4                            │
│ Seed      -1                           │
│                                        │
│ [生成 BGM]  [停止]                     │
└────────────────────────────────────────┘

┌ 本地导入 ───────────────────────────────┐
│ 当前已有上传 BGM 表单                  │
└────────────────────────────────────────┘

BGM Library

[地下室悬疑]
ACE-Step 1.5 XL Turbo
▶ Preview
Volume
Trim
Edit
Delete
```

---

# 53. BGM Library 卡片建议显示生成来源

对于 AI BGM：

```text
地下室悬疑
ACE-Step 1.5 XL Turbo
60s · 58 BPM · D minor
Seed 42
```

对于上传：

```text
用户导入
```

字段：

```javascript
source:
    'ace-step-v15-xl-turbo-diffusers'
```

即可判断。

---

# 54. Seed 的产品价值

一定要保存 seed。

因为当用户发现：

```text
“这首风格很好，但长度/Prompt 想稍微改”
```

seed 是后续可复现和迭代的重要信息。

第一阶段后端如果收到：

```json
{
  "seed": -1
}
```

应该生成实际 seed，并建议未来通过响应头回传：

```http
X-ACE-Step-Seed: 12345678
```

前端读取：

```javascript
response.headers.get(
    'X-ACE-Step-Seed'
)
```

然后把真实 seed 保存进：

```text
generation.seed
```

这是比只保存 `-1` 更好的生产实现。

建议第一版就做。

---

# 55. 建议增加响应头

成功响应：

```http
Content-Type: audio/wav

X-ACE-Step-Model:
acestep-v15-xl-turbo-diffusers

X-ACE-Step-Seed:
123456

X-ACE-Step-Sample-Rate:
48000
```

后端：

```python
return Response(
    content=audio,
    media_type="audio/wav",
    headers={
        "X-ACE-Step-Model":
            "acestep-v15-xl-turbo-diffusers",

        "X-ACE-Step-Seed":
            str(resolved_seed),

        "X-ACE-Step-Sample-Rate":
            "48000",
    },
)
```

这要求 worker 将：

```text
resolved_seed
```

通过 sidecar JSON 或 stdout protocol 回给 API。

如果第一阶段想保持 runtime 极简，可以暂时不做 header，后续补。

---

# 56. 建议的 BGM Prompt Director 输出格式

等 Phase 4 自动化时，让 LLM 输出：

```json
{
  "name": "地下室低压悬疑",
  "description": "主角进入地下室后至发现异常之前使用",
  "prompt_en": "Dark cinematic ambient underscore, sparse felt piano, low cello drone, subtle dissonant strings, distant metallic textures, restrained dynamics, minimal percussion, designed underneath spoken narration, no vocals, no choir, no spoken words.",
  "duration_seconds": 180,
  "bpm": 56,
  "keyscale": "D minor",
  "timesignature": "4"
}
```

不要让 LLM 输出：

```text
steps
offload
dtype
guidance
CUDA
```

这些是运行时参数，不属于音乐导演语义。

---

# 57. Prompt 职责分层

推荐最终架构：

```text
LLM Director
负责：
场景
情绪
音乐风格
乐器
节奏
BPM
Key
时长

ACE-Step Client
负责：
HTTP 请求

ACE-Step Backend
负责：
steps
dtype
offload
VAE tiling
GPU lock
worker lifecycle

ACE-Step Pipeline
负责：
music generation
```

不要把四层职责混在一个 Prompt 里。

---

# 58. 16GB 卡推荐最终环境配置

你的机器默认建议：

```bash
export ACESTEP_MODEL_DIR="$HOME/hf-mirror/ACE-Step/acestep-v15-xl-turbo-diffusers"

export ACESTEP_DTYPE=bfloat16

export ACESTEP_OFFLOAD=model

export ACESTEP_VAE_TILING=1

export ACESTEP_DEFAULT_STEPS=8

export ACESTEP_DEFAULT_SECONDS=60

export ACESTEP_REQUEST_TIMEOUT=1800

export ACESTEP_PORT=8313

export LOCAL_FILES_ONLY=1
```

不要默认：

```bash
ACESTEP_OFFLOAD=none
```

---

# 59. 如果 XL 实际生产性能不理想

你已经下载 XL，因此第一步就按本文接入。

如果实测发现：

```text
CPU offload 太慢
或
长 BGM OOM
```

**不要推倒 API。**

只需要以后下载：

```text
ACE-Step/acestep-v15-turbo
```

然后：

```bash
export ACESTEP_MODEL_DIR=...
```

8313 API、前端 `bgm-client.js`、BGM Library、工程存储和时间轴完全无需重构。

这也是本文把：

```text
服务名 = ace_step_1_5
```

而不是：

```text
服务名 = xl_turbo_diffusers
```

的原因。

---

# 60. 验收标准

这次接入只有同时达到以下条件才算完成。

## 后端

- [ ] `8313 /v1/health` 正常
- [ ] health 不加载模型
- [ ] 本地权重完全离线运行
- [ ] 10 秒生成成功
- [ ] 30 秒生成成功
- [ ] 60 秒生成成功
- [ ] BF16 生效
- [ ] CPU model offload 生效
- [ ] VAE tiling 生效
- [ ] GPU lock 生效
- [ ] worker 退出后显存释放
- [ ] 不影响 VoxCPM2
- [ ] 不影响 MOSS
- [ ] 不影响 Stable Audio

---

## 前端

- [ ] BGM 页面能生成
- [ ] 生成 WAV 自动进入 bgmLibrary
- [ ] 可以试听
- [ ] 可以调整音量
- [ ] 可以调整 trim
- [ ] 可以插入时间轴
- [ ] play 正常
- [ ] stop 正常
- [ ] 完整工程可导出
- [ ] 完整工程可恢复
- [ ] 最终 WAV 混音正常

---

# 61. 最终建议

第一版的核心原则只有一句：

> **ACE-Step 只负责“生成一个标准 BGM WAV”，后续资产管理、时间轴和混音全部复用当前 WebUI。**

不要同时重构：

```text
BGM 数据模型
项目 schema
Timeline
Mixer
Script Prompt
SoundEffect
```

最小改动路线：

```text
后端新增 8313
+
前端新增 bgm-client.js
+
生成 Blob 写入现有 bgmLibrary
```

即可打通完整链路。

等这一版稳定之后，再做真正高价值的下一层：

```text
小说
↓
悬疑导演 LLM
↓
BGM Generation Plan
↓
ACE-Step
↓
主题 / 调查 / 紧张 / 高潮 / 结尾
↓
自动加入 BGM Library
↓
LLM 自动安排 play / stop
↓
VoxCPM2 + MOSS + ACE-Step
↓
完整有声小说
```

这才是最终适合 Unitale 的 BGM Director 工作流。

---

# 62. 官方参考资料

- [ACE-Step 1.5 官方 GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [ACE-Step XL Turbo Diffusers 模型](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers)
- [Hugging Face Diffusers ACE-Step Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/ace_step)
- [Diffusers 内存优化 / CPU Offload](https://huggingface.co/docs/diffusers/optimization/memory)
- [ACE-Step GPU Compatibility](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md)

---

# 63. 当前仓库对应改动文件总表

## TTS-and-VoiceDesign

```text
新增：
ace_step_1_5/.python-version
ace_step_1_5/main.py
ace_step_1_5/runtime.py
ace_step_1_5/worker.py
ace_step_1_5/pyproject.toml
ace_step_1_5/uv.lock
ace_step_1_5/README.md
ace_step_1_5/tests/test_api.py
ace_step_1_5/tests/test_runtime.py

修改：
start.sh
README.md
.gitignore（如果需要）
tests/（可选增加根级契约测试）
```

---

## TTS-Studio-WebUI

```text
新增：
js/bgm-client.js

修改：
index.html
README.md
docs/TTS-and-VoiceDesign接入.md
docs/本地开发与回归.md
```

第一阶段：

```text
js/project-storage.js
```

**原则上不需要修改。**

只有后续正式引入：

```text
bgm_generation_plan
```

并将它作为新的长期工程实体时，再考虑：

```text
schemaVersion 4 → 5
```

---

**文档结束。**
