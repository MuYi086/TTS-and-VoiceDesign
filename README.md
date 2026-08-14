# Unitale AI Local Backend

本项目是 Unitale 前端使用的本地后端，当前提供：

- Qwen3-TTS-12Hz-1.7B-Base：参考音频 + 文本合成，端口 `8305`
- VoxCPM2：参考音频 + 文本合成，端口 `8306`
- LongCat-AudioDiT-3.5B-bf16：24 kHz 参考音频声音克隆，端口 `8307`
- dots.tts-soar：48 kHz 参考音频声音克隆，端口 `8308`
- MOSS-SoundEffect v2.0：根据中英文提示词生成 48 kHz 声效，端口 `8311`
- Stable Audio 3 Medium：默认音效模型；根据英文提示词生成音乐或 44.1 kHz 立体声音效，端口 `8313`
- Qwen3-TTS VoiceDesign：根据音色描述生成参考音频，独立 uv 服务端口 `8314`，路由 `/v1/qwen/design`
- MOSS VoiceGenerator：根据音色描述生成参考音频，独立 uv 服务端口 `8315`，路由 `/v1/moss/design`
- Step-Audio-EditX：对已上传的音频按情绪、说话风格、非语言表现等进行迭代编辑；独立 uv 服务端口 `8316`，主 API `8300` 保留兼容代理，路由 `/v1/step-audio-editx/edit`
- MiMo TTS VoiceDesign：根据音色描述生成参考音频，走主 API 的 `/v1/mimo/design`
- VoxCPM2 VoiceDesign：根据音色描述生成参考音频，走主 API 的 `/v1/voxcpm2/design`

主 API、其它模型 wrapper/worker 和共享运行时模块位于 `api/`；Qwen3-TTS Base、Qwen3-TTS VoiceDesign 和 MOSS VoiceGenerator 分别位于 `qwen3_tts/`、`qwen3_voiceDesign/` 和 `moss_voiceGenerator/`，使用各自的 uv 环境。上传资源、缓存和供应商代码位于 `api/prompts/`、`api/.cache/` 与 `api/vendor/`。所有本地 TTS 模型成功合成的 WAV 都会额外保留在 `api/tempAudio/`，文件名前缀用于区分模型；可通过 `TTS_OUTPUT_DIR` 统一覆盖，也可通过对应模型的 `*_OUTPUT_DIR` 覆盖。不要把生成音频或模型权重提交到 Git。

## 本地环境

主 API 和其它轻量 wrapper 默认使用 `moss-soundEffect` Conda 环境；Qwen3-TTS 8305、Qwen3-TTS VoiceDesign 8314 和 MOSS VoiceGenerator 8315 服务使用各自目录内的 uv 环境，由 `uv run` 启动。如果部署环境另有共享 wrapper 环境，可通过 `CONDA_ENV` 覆盖：

```bash
conda activate moss-soundEffect
```

Qwen3-TTS、MOSS VoiceGenerator 和 Step-Audio-EditX 在请求期间分别拉起一次性 worker；三者使用各自 uv 项目的 Python，其它模型使用对应 Conda 环境。模型在请求结束后由 worker 退出释放显存；主 API、各包装器和 worker 共享 `GPU_LOCK_FILE`，避免并发抢占 GPU。迁移期间可设置 `STEP_AUDIO_EDITX_RUNTIME=conda` 回退到旧 `api/step_audio_editx_worker.py`。

```bash
uv run --project qwen3_tts python qwen3_tts/worker.py ...
conda run -n voxcpm2 python api/voxcpm2_worker.py ...
uv run --project qwen3_voiceDesign python qwen3_voiceDesign/worker.py ...
uv run --project moss_voiceGenerator python moss_voiceGenerator/worker.py ...
uv run --project Step_Audio_EditX python Step_Audio_EditX/worker.py ...
conda run -n LongCat-AudioDiT-3.5B-bf16 python api/longcat_audiodit_worker.py ...
conda run -n dots_tts_soar python api/dots_tts_soar_worker.py ...
```

MOSS-SoundEffect 使用独立的 `moss-soundEffect` 环境；Stable Audio 3 Medium 使用独立的
`stable_audio_3_medium` 环境，并通过本机 `stable-audio-3` 官方源码运行。MiMo 是云端 API，须通过环境变量提供密钥：

```bash
export MIMO_API_KEY=...
```

MiMo 是云端服务，运行后端的机器必须能连接 `https://api.xiaomimimo.com:443`。若网络需要代理，请在启动 `start.sh` 前设置标准代理环境变量，例如 `HTTPS_PROXY=http://127.0.0.1:7890`（并按需设置 `NO_PROXY=127.0.0.1,localhost`）。网络不可达时 `/v1/mimo/design` 会返回 `503` 及可操作的错误说明；这不会影响本地 Qwen 音色设计或其他本地 TTS 接口。

## 模型路径

默认读取以下本地目录；可在启动前用同名环境变量覆盖：

```text
/home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-VoiceGenerator
/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-Audio-Tokenizer
/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0
/home/muyi086/hf-mirror/stabilityai/stable-audio-3-medium
/home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-Base
/home/muyi086/hf-mirror/openbmb/VoxCPM2
/home/muyi086/hf-mirror/drbaph/LongCat-AudioDiT-3.5B-bf16
/home/muyi086/hf-mirror/rednote-hilab/dots.tts-soar
/home/muyi086/hf-mirror/google/umt5-base
/home/muyi086/tts-depency/LongCat-AudioDiT
/home/muyi086/tts-depency/stable-audio-3
/home/muyi086/github/TTS-and-VoiceDesign/api/voxcpm2_helpers.py
/home/muyi086/hf-mirror/stepfun-ai/Step-Audio-EditX
/home/muyi086/hf-mirror/stepfun-ai/Step-Audio-Tokenizer
/home/muyi086/tts-depency/Step-Audio-EditX
```

## 启动与健康检查

```bash
bash start.sh
curl http://127.0.0.1:8300/v1/health
curl http://127.0.0.1:8305/v1/health
curl http://127.0.0.1:8314/v1/health
curl http://127.0.0.1:8315/v1/health
curl http://127.0.0.1:8316/v1/health
curl http://127.0.0.1:8306/v1/health
curl http://127.0.0.1:8307/v1/health
curl http://127.0.0.1:8308/v1/health
curl http://127.0.0.1:8311/v1/health
curl http://127.0.0.1:8313/v1/health
```

默认服务地址：

```text
http://127.0.0.1:8300  MiMo/VoxCPM2 音色设计与 Step-Audio-EditX 编辑
http://127.0.0.1:8305  Qwen3-TTS-12Hz-1.7B-Base
http://127.0.0.1:8314  Qwen3-TTS VoiceDesign
http://127.0.0.1:8315  MOSS VoiceGenerator
http://127.0.0.1:8316  Step-Audio-EditX uv 服务
http://127.0.0.1:8306  VoxCPM2
http://127.0.0.1:8307  LongCat-AudioDiT-3.5B-bf16
http://127.0.0.1:8308  dots.tts-soar
http://127.0.0.1:8311  MOSS-SoundEffect v2.0
http://127.0.0.1:8313  Stable Audio 3 Medium
```

## 音效生成接口

MOSS-SoundEffect 和默认的 Stable Audio 3 Medium 都接受同样的基础请求结构：

```bash
curl -X POST http://127.0.0.1:8311/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"雨夜中木门被轻敲三下，近距离，无可辨认说话声","seconds":3}' \
  -o moss-sfx.wav

curl -X POST http://127.0.0.1:8313/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A wooden door is knocked three times in a quiet room, close perspective, no speech. TrackType: SFX","seconds":3}' \
  -o stable-medium-sfx.wav
```

`8313` Medium 支持 `steps`（默认 `8`）、`cfg_scale`（默认 `1.0`）和 `seed`（默认 `-1`，每次随机），最大时长为 380 秒。它要求 CUDA GPU、Ampere 或更新架构和 Flash Attention 2，不能退回 CPU；输出为 44.1 kHz、32-bit float、立体声 WAV，可生成音乐和音效，不用于语音或声音克隆。
官方模型以英文描述训练，英文提示词效果最佳；WebUI 会在剧本分析时同时生成中文 MOSS 提示词和英文
Stable Audio 提示词。短而具体的音效应使用与实际声音相符的短时长；`TrackType: SFX` 通常可帮助模型
保持音效语义。

`8313` Medium 也采用一请求一个 worker；worker 在 `stable_audio_3_medium` 环境中载入本地
`stabilityai/stable-audio-3-medium` 权重，完成后显式清理 CUDA allocator 并退出。可通过
`STABLE_AUDIO_3_MEDIUM_CONDA_ENV`、`STABLE_AUDIO_3_MEDIUM_MODEL_DIR`、`STABLE_AUDIO_3_MEDIUM_DEVICE`、
`STABLE_AUDIO_3_MEDIUM_DTYPE`、`STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS`、
`STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS`、`STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE`、
`STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED` 和 `STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT` 覆盖配置；
输出目录可由 `STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR` 或 `TTS_OUTPUT_DIR` 覆盖。

## 语音合成接口

四个语音合成服务均支持以下流程：

1. `POST /v1/upload_audio` 上传参考音频。
2. `GET /v1/check/audio?file_name=...` 确认后端已保存。
3. `POST /v2/synthesize` 生成目标音频。

所有 `/v2/synthesize` 仅做参考音频克隆，不接受 `style_prompt`；音色或风格应在生成参考音频阶段通过 Qwen 或 MiMo 的音色设计接口确定。

| 服务 | `prompt_text` 处理 |
| --- | --- |
| `8305` Qwen3-TTS Base | 映射为官方 `ref_text`；缺失时回退到仅参考音频的克隆。 |
| `8306` VoxCPM2 | `clone_mode="ultimate"` 有准确 `prompt_text` 时走 Ultimate Cloning；`clone_mode="controllable"` 只接受 `control_instruction`，不接受 `prompt_text`，并将指令写入目标文本前；未指定模式时保留旧的参考文本 / 仅参考音频兼容路径。 |
| `8307` LongCat-AudioDiT-3.5B-bf16 | 必须提供参考音频准确逐字的 `prompt_text`；worker 按官方接口拼接 `prompt_text + text`，把参考音频重采样为 24 kHz 单声道，并使用模型配置的 `max_wav_duration` 限制总时长。 |
| `8308` dots.tts-soar | 推荐提供参考音频准确逐字的 `prompt_text`，用于 continuation voice cloning；没有参考文本时使用官方支持的 x-vector-only cloning。仅支持 CUDA，输出 48 kHz 单声道。 |

LongCat-AudioDiT 的默认参数与官方声音克隆示例一致：16 步 ODE、`guidance_strength=4.0`、`guidance_method="apg"`、VAE 使用 float16。它只支持 CUDA；参考音频必须是获得授权的单说话人语音，`prompt_text` 必须与实际朗读内容准确一致。长文本按中文/英文标点分块，每块会重新带入参考音频并在片段间插入停顿；单段总时长不能超过模型配置的上限（本机 3.5B 权重为 60 秒）。这些限制来自 [LongCat-AudioDiT 官方仓库](https://github.com/meituan-longcat/LongCat-AudioDiT) 的 Python/CLI 推理示例。

LongCat 服务采用“一次请求一个 worker”生命周期；worker 完成或报错时显式执行 CUDA 同步、`empty_cache`、`ipc_collect`，随后进程退出，以释放模型显存。可通过以下环境变量覆盖默认配置：`LONGCAT_AUDIODIT_CONDA_ENV`、`LONGCAT_AUDIODIT_MODEL_DIR`、`LONGCAT_AUDIODIT_REPO_PATH`、`LONGCAT_AUDIODIT_TOKENIZER_PATH`、`LONGCAT_AUDIODIT_NFE`、`LONGCAT_AUDIODIT_GUIDANCE_STRENGTH`、`LONGCAT_AUDIODIT_GUIDANCE_METHOD`、`LONGCAT_AUDIODIT_MAX_CHARS_PER_CHUNK`、`LONGCAT_AUDIODIT_PAUSE_MS`、`LONGCAT_AUDIODIT_VAE_DTYPE` 和 `LONGCAT_AUDIODIT_REQUEST_TIMEOUT`。

LongCat、dots.tts-soar 和 Qwen3-TTS 的克隆调试默认值都集中在对应服务入口顶部，并带有中文说明；直接修改 `*_DEFAULT` 常量后重启服务即可生效。`start.sh` 只负责启动路由、环境和模型路径，不再覆盖这些服务入口内的合成默认值；部署时显式设置的同名环境变量仍然优先。

LongCat-AudioDiT 和 dots.tts-soar 的 worker 会在每个生成分段拼接前裁掉明显的前导静音，并在完整音频拼接后再次兜底检查；裁剪保留 40 毫秒起音保护，分段之间通过 `pause_ms` 配置的停顿仍会保留。

dots.tts-soar 同样采用“一次请求一个 worker”生命周期；worker 完成或报错时显式清理 CUDA allocator，随后进程退出释放模型显存。可通过 `DOTS_TTS_SOAR_CONDA_ENV`、`DOTS_TTS_SOAR_MODEL_DIR`、`DOTS_TTS_SOAR_PRECISION`、`DOTS_TTS_SOAR_LANGUAGE`、`DOTS_TTS_SOAR_NUM_STEPS`、`DOTS_TTS_SOAR_GUIDANCE_SCALE`、`DOTS_TTS_SOAR_SPEAKER_SCALE`、`DOTS_TTS_SOAR_MAX_GENERATE_LENGTH`、`DOTS_TTS_SOAR_MAX_CHARS_PER_CHUNK`、`DOTS_TTS_SOAR_PAUSE_MS`、`DOTS_TTS_SOAR_SEED` 和 `DOTS_TTS_SOAR_REQUEST_TIMEOUT` 覆盖默认配置。SOAR 的 continuation cloning 要求 `prompt_text` 与参考音频实际内容一致；省略时才使用 x-vector-only 模式。

VoxCPM2 的 `ultimate` 与 `controllable` 请求路径严格互斥：前者用于最大化复刻参考音频细节，后者用于按短控制指令调整表演节奏和情绪。所有 VoxCPM2 克隆与音色设计请求未显式传 `cfg_value` 时统一使用顶部全局配置 `VOXCPM2_CFG_VALUE`（官方 Demo 默认 `2.0`）；需要单次覆盖时仍可在请求中显式传 `cfg_value`。默认 `seed=-1`，与官方在线推理一样不固定随机种子，重新生成会得到不同候选；需要精确复现时才显式传非负 `seed`。`control_instruction` 不是响度参数；成片响度应在合成后检测和统一归一化。

`8306` 还支持 `nonverbal_tags`（数组，最多一个）。仅接受官方标签 `laughing`、`sigh`、`Uhm`、`Shh`、`Question-ah`、`Question-ei`、`Question-en`、`Question-oh`、`Surprise-wa`、`Surprise-yo`、`Dissatisfaction-hnn`，且只能配合 `clone_mode="controllable"` 使用。worker 会把最终目标文本拼为 `(control_instruction)[tag]正文`（无控制或标签时省略相应前缀），并在每个文本分片调用模型前向终端打印该最终文本、分片序号和克隆模式；不会打印参考音频转写。

参考音频上传按内容 `sha256` 校验，不再只按文件名判断是否存在；同名音频更新后会自动覆盖服务端旧缓存。`GET /v1/check/audio` 返回 `sha256` 和 `size_bytes`，供 WebUI 判断是否需要重新上传。

VoxCPM2 可直接在 [`api/voxcpm2_api.py`](api/voxcpm2_api.py) 顶部修改集中默认值：`cfg_value`、`inference_timesteps`、`normalize`、`denoise`、`retry_badcase`、`load_denoiser`、`optimize`、`device`、`seed`、分片长度、分片停顿和超时。`start.sh` 不再写入这些默认值；如启动前显式设置同名 `VOXCPM2_*` 环境变量，环境变量仍会覆盖代码默认值。`denoise=true` 时会自动启用 `load_denoiser`。

每个本地 TTS 与 Stable Audio 3 Medium 端点成功后都会保留一份原始 WAV 到输出目录，接口响应内容不变。默认目录为 `api/tempAudio/`，文件名前缀示例为 `qwen3_tts`、`voxcpm2`、`longcat_audiodit`、`dots_tts_soar` 或 `stable_audio_3_medium`。此目录不会自动清理，完成后请按需要转移或删除文件；`VOXCPM2_OUTPUT_DIR` 继续兼容旧版 VoxCPM2 专用配置。

## Step-Audio-EditX 编辑接口

Step-Audio-EditX 使用上传到 `8300` 的音频作为 prompt。先通过 `POST /v1/upload_audio` 上传音频，再调用：

```bash
curl -X POST http://127.0.0.1:8300/v1/step-audio-editx/edit \
  -H 'Content-Type: application/json' \
  -d '{"prompt_audio":"step-audio-editx/line-1.wav","prompt_text":"这是一条台词。","generated_text":"这是一条台词。","edit_type":"emotion","edit_info":"coldness"}' \
  -o edited.wav
```

请求字段 `edit_type`、`edit_info` 分别映射官方命令行的 `--edit-type`、`--edit-info`。`emotion`、`style` 与 `speed` 需要非空 `edit_info`；`paralinguistic` 使用目标文本中的官方标签；`denoise` 与 `vad` 不要求文本。`start.sh` 默认启动 `Step_Audio_EditX/` 的 uv 服务到 `8316`，主 API `8300` 的同路径会代理到该服务，因此 WebUI 无需修改地址；启动脚本使用 `uv run --no-sync`，请先完成一次 `uv sync --project Step_Audio_EditX --locked`。模型、tokenizer、官方源码和推理参数可分别通过 `STEP_AUDIO_EDITX_MODEL_DIR`、`STEP_AUDIO_TOKENIZER_PATH`、`STEP_AUDIO_EDITX_CODE_PATH`、`STEP_AUDIO_EDITX_*` 覆盖。单次音频建议不超过 30 秒。

Step-Audio-EditX 当前不要求安装 `flash_attn`：本机 uv/Conda 环境均未安装该模块，上游推理路径固定使用 `VLLM_ATTENTION_BACKEND=TRITON_ATTN`，并默认 `enforce_eager=1`；`/v1/health` 会报告 `flash_attn` 状态和该策略。`/home/muyi086/tts-depency/flash-attention` 是源码仓库，不等于已安装且可加载的扩展；只有后续实测需要 FlashAttention 性能优化时，才应针对当前 Torch/CUDA/Python 编译并单独做 canary。

```bash
curl -X POST http://127.0.0.1:8306/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"唉，还是晚了一步。","audio_path":"reference.wav","clone_mode":"controllable","control_instruction":"自然、清晰地表达，保留必要的非语言反应，吐字清晰","nonverbal_tags":["sigh"]}' \
  -o synth.wav

curl -X POST http://127.0.0.1:8307/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"今天晴暖转阴雨。","audio_path":"reference.wav","prompt_text":"这是一句参考音频转写。"}' \
  -o longcat_synth.wav

curl -X POST http://127.0.0.1:8308/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"今天晴暖转阴雨。","audio_path":"reference.wav","prompt_text":"这是一句参考音频转写。"}' \
  -o dots_tts_soar_synth.wav
```

音色设计端点：

Qwen3-TTS VoiceDesign 已完全迁移到独立的 uv 服务，直接访问 8314；主 API 8300 不再承载该模型的旧 Conda 逻辑。

```bash
curl -X POST http://127.0.0.1:8314/v1/qwen/design \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o qwen_voicedesign_reference.wav

curl -X POST http://127.0.0.1:8315/v1/moss/design \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，温柔、清晰，语速中等。","text":"你好。"}' \
  -o moss_reference.wav

curl -X POST http://127.0.0.1:8300/v1/voxcpm2/design \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o voxcpm2_reference.wav

curl -X POST http://127.0.0.1:8300/v1/mimo/design \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o mimo_reference.wav
```

WebUI 的“脚本制作”页默认选择 `stable-audio-3-medium`（8313），也可切换至 `moss-soundEffect-v2`（8311）。剧本分析会为每项 SoundEffect 计划同时保存中文 `prompt` 和英文
`prompt_en`；选择 MOSS 时发送中文字段，选择 Medium 时发送英文字段。下方“生成全部 SoundEffect
音效”会顺序调用当前下拉框选中的模型；生成结果由页面存入工程资产库，Stable Audio 服务端也会同步保留
原始 WAV 到 `api/tempAudio/`。

MOSS VoiceGenerator 必须搭配官方的 **MOSS-Audio-Tokenizer（v1，24 kHz、单声道）**；`MOSS-Audio-Tokenizer-v2` 是 48 kHz 双声道 codec，不能用于当前 1.7B VoiceGenerator，否则会产生非语音噪声。可通过 `MOSS_AUDIO_TOKENIZER_PATH` 覆盖默认路径。

如果 `/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-Audio-Tokenizer` 只是空目录、只有部分文件，或目录中没有 `config.json` 与模型权重，MOSS 请求会在加载阶段失败。使用 hf-mirror 下载完整的 v1 codec（不要下载 `MOSS-Audio-Tokenizer-v2`）：

```bash
MOSS_CODEC_DIR="${MOSS_CODEC_DIR:-$HOME/hf-mirror/OpenMOSS-Team/MOSS-Audio-Tokenizer}"
HF_ENDPOINT=https://hf-mirror.com hf download OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --local-dir "$MOSS_CODEC_DIR"
```

下载完成后，`GET http://127.0.0.1:8315/v1/health` 中的 `available.moss_audio_tokenizer` 应为 `true`，再重启 `bash start.sh`。worker 现在会在加载 Transformers 前检查 codec 的 `model_type`、24 kHz 单声道配置和权重完整性，并对不完整目录给出明确错误。

VoxCPM2 音色设计由独立的 `api/voxcpm2_voice_design.py` 和 `api/voxcpm2_voice_design_worker.py` 处理，不与克隆 worker 或 Qwen / MiMo 逻辑混用。它按照官方文档将音色描述编码为 `(音色描述)正文` 后调用 `model.generate()`。官方示例中的 `seed=42` 是可复现示例值，不是质量专用值；本项目克隆与音色设计默认不固定随机种子，需要复现实例时可通过请求显式传入 `seed=42`。

## 本地回归测试

测试不会下载权重、调用外部服务或加载 TTS 模型：

```bash
uv run --project moss_voiceGenerator python -m unittest discover -s tests -v
```

当前测试覆盖 Qwen3-TTS Base 与 VoiceDesign 的路由、健康契约、音频上传和 prompt sidecar、请求校验，以及两个 worker 使用 uv 解释器的一次性启动约束；测试不会下载权重、调用外部服务或加载 TTS 模型。
