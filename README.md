# Unitale AI Local Backend

Unitale 前端使用的本地语音后端，提供参考音频克隆、音色设计、语音编辑和
SoundEffect 生成。仓库采用“一个服务一个 uv 项目”的边界：HTTP 控制面不加载
重型模型，模型推理由对应目录中的一次性 worker 完成。

## 服务总览

| 服务 | 端口 | 主要用途 | 主要路由 |
| --- | ---: | --- | --- |
| 控制面 | 8300 | 控制面、共享上传/检查、MiMo 兼容代理 | `/v1/control` |
| Qwen3-TTS VoiceDesign | 8301 | 本地音色设计 | `/v1/qwen/timbre` |
| MOSS VoiceGenerator | 8302 | 本地音色设计 | `/v1/moss/timbre` |
| MiMo TTS VoiceDesign | 8303 | 云端音色设计 | `/v1/mimo/timbre` |
| Stable Audio 3 Medium | 8311 | 文本生成音乐或声效 | `/v1/stableAudio/soundEffect` |
| MOSS-SoundEffect v2 | 8312 | 文本生成声效 | `/v1/moss/soundEffect` |
| Qwen3-TTS Base | 8321 | 参考音频语音克隆 | `/v1/qwen/clone` |
| VoxCPM2 | 8322 | 语音克隆、音色设计 | `/v1/voxcpm2/clone` |
| LongCat-AudioDiT-3.5B | 8323 | 参考音频语音克隆 | `/v1/longCat/clone` |
| dots.tts-soar | 8324 | 参考音频语音克隆 | `/v2/dotsTTS/clone` |
| Step-Audio-EditX | 8331 | 语音编辑 | `/v1/stepAudioEditx/edit` |

每个服务都提供 `GET /v1/health`。除控制面和 MiMo 外，服务还保留本机访问的
`POST /internal/unload_all` 兼容路由；该路由只返回“一次性 worker 已退出”的状态，
不会加载常驻模型。成功生成接口返回 `audio/wav`，并在服务端保存一份 WAV。

## 目录与运行数据

```text
main/                         8300 控制面，不包含模型推理
qwen3_tts/                    Qwen3-TTS Base 的 HTTP 服务和 worker
voxcpm2/                      VoxCPM2 的 HTTP 服务、克隆和音色设计 worker
LongCat_AudioDiT_3.5B_bf16/  LongCat-AudioDiT 服务和 worker
dots_tts_soar/                dots.tts-soar 服务和 worker
moss_soundEffect/             MOSS-SoundEffect v2 服务和 worker
stable_audio_3_medium/        Stable Audio 3 Medium 服务和 worker
qwen3_voiceDesign/            Qwen VoiceDesign 服务和 worker
moss_voiceGenerator/          MOSS VoiceGenerator 服务和 worker
mimo_tts/                     MiMo 云端编排服务
Step_Audio_EditX/             Step-Audio-EditX 服务和 worker
tests/                        根目录无模型回归测试
soundEffect/                  MOSS GPU 示例和提示词说明
storage/                      上传音频、生成音频、sidecar、缓存和 GPU 锁
```

默认运行数据目录为：

| 目录 | 内容 | 覆盖变量 |
| --- | --- | --- |
| `storage/timbre/` | Qwen、MOSS、VoxCPM2、MiMo 生成的音色参考音频 | `TIMBRE_STORAGE_DIR` |
| `storage/soundEffect/` | MOSS 和 Stable Audio 生成的声效 | `SOUNDEFFECT_STORAGE_DIR`、`STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR` |
| `storage/clone/` | 参考音频、克隆结果和 Step 编辑结果 | `CLONE_STORAGE_DIR`、各服务的 `*_OUTPUT_DIR` |
| `storage/.cache/runtime/` | worker 临时文件、库缓存和共享 GPU 锁 | `RUNTIME_CACHE_DIR`、`GPU_LOCK_FILE` |

如果上传音频的内容与 `storage/timbre/` 中已有的设计音色一致，Qwen3-TTS、VoxCPM2、
LongCat 和 dots.tts-soar 会在 `storage/timbre/.references/` 保存引用映射，不再把同一 WAV
复制到 `storage/clone/`；普通用户上传的参考音频仍保存到 `storage/clone/`。这些目录是运行
数据，不要提交到 Git。

## 安装与启动

运行要求：Python `3.12.13`、`uv`、可用的 CUDA/NVIDIA 驱动（本地模型服务），以及
下方列出的模型权重和外部源码目录。权重与第三方源码不放进本仓库。

先为需要的服务同步锁定依赖；部署全部服务时可以执行：

```bash
for project in qwen3_tts mimo_tts voxcpm2 LongCat_AudioDiT_3.5B_bf16 \
  dots_tts_soar moss_soundEffect stable_audio_3_medium \
  qwen3_voiceDesign moss_voiceGenerator Step_Audio_EditX; do
  uv sync --project "$project" --locked
done
```

Step-Audio-EditX、MOSS-SoundEffect 和 Stable Audio 3 依赖外部源码或系统命令；
先准备对应路径，再执行 `uv sync`。启动前应完成依赖同步，不要把 `start.sh` 当作依赖
安装流程；使用 `--no-sync` 的服务尤其要求对应环境已经准备好。

MiMo 是云端服务，必须配置密钥：

```bash
export MIMO_API_KEY='...'
```

默认情况下 `LOCAL_FILES_ONLY=1`，本地 worker 不会从 Hugging Face 下载权重。确认模型、
Tokenizer 和上游源码就绪后启动全部服务：

```bash
bash start.sh
```

`start.sh` 会启动 8300、8301、8302、8303、8311、8312、8321、8322、8323、8324 和
8331 共 11 个进程；8300 使用 `qwen3_tts` uv 项目中的轻量 HTTP 依赖，其余服务使用
各自的 uv 项目，本地 GPU 服务通过 `GPU_LOCK_FILE` 串行访问 GPU。
任一子进程退出时脚本会终止其余进程组并清理 worker。

健康检查：

```bash
for port in 8300 8301 8302 8303 8311 8312 8321 8322 8323 8324 8331; do
  curl -fsS "http://127.0.0.1:${port}/v1/health" >/dev/null && echo "${port}: ok"
done
```

单独调试服务时，从仓库根目录执行，例如：

```bash
HOST=127.0.0.1 PORT=8321 \
  uv run --project qwen3_tts python qwen3_tts/main.py
```

## 模型路径与主要配置

`start.sh` 默认使用 `HF_MIRROR_DIR`（默认为 `$HOME/hf-mirror`）和 `$HOME/tts-depency`；
所有路径都可在启动前用环境变量覆盖。

| 服务 | 默认权重 | 其他必需路径 |
| --- | --- | --- |
| Qwen3-TTS Base | `$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `QWEN3_TTS_MODEL_DIR` |
| Qwen VoiceDesign | `$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `QWEN_VOICEDESIGN_MODEL_DIR` |
| MOSS VoiceGenerator | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator` | `MOSS_VOICEGENERATOR_MODEL_DIR`、`MOSS_AUDIO_TOKENIZER_PATH` |
| MOSS-SoundEffect | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0` | `MOSS_SOUNDEFFECT_CODE_PATH`、`MOSS_SOUNDEFFECT_MODEL_DIR` |
| Stable Audio 3 Medium | `$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium` | `STABLE_AUDIO_3_REPO_PATH`、`STABLE_AUDIO_3_MEDIUM_MODEL_DIR` |
| VoxCPM2 | `$HF_MIRROR_DIR/openbmb/VoxCPM2` | `VOXCPM2_MODEL_DIR`、仓库内 `voxcpm2/voxcpm2_helpers.py` |
| LongCat-AudioDiT | `$HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16` | `LONGCAT_AUDIODIT_REPO_PATH`、`LONGCAT_AUDIODIT_TOKENIZER_PATH` |
| dots.tts-soar | `$HF_MIRROR_DIR/rednote-hilab/dots.tts-soar` | `DOTS_TTS_SOAR_MODEL_DIR` |
| Step-Audio-EditX | `$HF_MIRROR_DIR/stepfun-ai/Step-Audio-EditX` | `STEP_AUDIO_TOKENIZER_PATH`、`STEP_AUDIO_EDITX_CODE_PATH` |

通用配置包括 `HOST`、`PORT`、`STORAGE_DIR`、`PROMPTS_DIR`、`RUNTIME_CACHE_DIR`、
`GPU_LOCK_FILE`、`LOCAL_FILES_ONLY` 和 `CUDA_RELEASE_DELAY`。服务专用配置使用对应
前缀，例如 `QWEN3_TTS_*`、`VOXCPM2_*`、`LONGCAT_AUDIODIT_*`、`DOTS_TTS_SOAR_*`、
`MOSS_SOUNDEFFECT_*`、`STABLE_AUDIO_3_MEDIUM_*`、`STEP_AUDIO_EDITX_*`、
`QWEN_VOICEDESIGN_*` 和 `MOSS_VOICEGENERATOR_*`。每个服务的 `/v1/health` 会报告
生效的路径、运行时和可用性。

Stable Audio 3 默认允许上游的 flex-attention/SDPA 回退；只有需要严格检查
FlashAttention 时才设置 `STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN=1`。VoxCPM2、
LongCat、dots.tts-soar 和 Step-Audio-EditX 的默认项目路径不要求安装 `flash_attn`。
不要把本机某个 FlashAttention 源码 checkout 当作已安装的 Python 扩展。

## 参考音频克隆

Qwen3-TTS、VoxCPM2、LongCat 和 dots.tts-soar 使用相同的三步 WebUI 流程：

1. `POST /v1/upload_audio`，表单字段为 `audio`、`full_path`；Qwen、VoxCPM2、LongCat
   和 dots.tts-soar 还接受可选的 `prompt_text`。
2. `GET /v1/check/audio?file_name=...` 检查服务自己的存储状态。
3. 调用当前模型的克隆路由，请求中的 `audio_path` 使用上传时的 `full_path` 文件名。

上传示例（以 Qwen3-TTS 8321 为例）：

```bash
curl -X POST http://127.0.0.1:8321/v1/upload_audio \
  -F 'audio=@reference.wav' \
  -F 'full_path=reference.wav' \
  -F 'prompt_text=这是一句参考音频转写。'

curl 'http://127.0.0.1:8321/v1/check/audio?file_name=reference.wav'

curl -X POST http://127.0.0.1:8321/v1/qwen/clone \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，欢迎使用。","audio_path":"reference.wav","prompt_text":"这是一句参考音频转写。"}' \
  -o qwen-clone.wav
```

各模型的 `prompt_text` 语义不同：

| 服务 | 行为 |
| --- | --- |
| Qwen3-TTS Base | 有准确参考文本时映射为官方 `ref_text`；也可使用仅音色向量克隆。 |
| VoxCPM2 | `clone_mode=ultimate` 使用参考文本；`clone_mode=controllable` 改用 `control_instruction`，二者互斥。`nonverbal_tags` 最多一个，且只能用于可控模式。 |
| LongCat-AudioDiT | 推荐提供与参考音频逐字一致的文本；参考音频会按官方流程重采样为 24 kHz 单声道。 |
| dots.tts-soar | 有参考文本时使用 continuation cloning；省略时保留官方 x-vector-only cloning。输出为 48 kHz 单声道。 |

所有这些克隆请求都拒绝 `style_prompt`；声音风格应通过音色设计或
VoxCPM2 的 `control_instruction` 表达。`text` 中的 Markdown 标题和列表标记会按服务
兼容逻辑清理，不能依赖它们传递控制指令。

## 音色设计

所有音色设计路由都接收 `voice_description` 和可选的 `text`，成功返回 `audio/wav`，
并把结果写入 `storage/timbre/`：

```bash
curl -X POST http://127.0.0.1:8301/v1/qwen/timbre \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o qwen-voice.wav

curl -X POST http://127.0.0.1:8302/v1/moss/timbre \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，温柔、清晰，语速中等。","text":"你好。"}' \
  -o moss-voice.wav

curl -X POST http://127.0.0.1:8303/v1/mimo/timbre \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o mimo-voice.wav
```

MiMo 的 8303 服务只做云端请求编排、重试、分段和本地音色缓存；8300 控制面保留兼容
代理。后端无法连接 MiMo API 时，独立服务和代理会返回 `503`，请检查
`MIMO_API_KEY`、`MIMO_BASE_URL`、DNS、HTTPS 出网和 `HTTPS_PROXY`。

MOSS VoiceGenerator 必须使用 **MOSS-Audio-Tokenizer v1**（24 kHz、单声道）。
不要把 48 kHz 双声道的 v2 codec 作为该服务的 tokenizer；8302 的健康检查中
`available.moss_audio_tokenizer` 应为 `true`。

## SoundEffect 生成

MOSS-SoundEffect v2 接受中英文非语言声效提示词，输出 48 kHz 单声道 WAV，`seconds`
范围为 `(0, 30]`。默认字段为 `num_inference_steps=100`、`cfg_scale=4.0`、
`sigma_shift=5.0`、`seed=0`：

```bash
curl -X POST http://127.0.0.1:8312/v1/moss/soundEffect \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"雨夜中木门被轻敲三下，近距离，无可辨认说话声","seconds":3}' \
  -o moss-sfx.wav
```

Stable Audio 3 Medium 接受英文提示词，可生成音乐或声效，输出为 44.1 kHz 立体声 WAV。
`seconds` 和官方别名 `duration` 必须一致，最大时长为 380 秒；默认
`seconds=7`、`steps=8`、`cfg_scale=1.0`、`seed=-1`：

```bash
curl -X POST http://127.0.0.1:8311/v1/stableAudio/soundEffect \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A wooden door is knocked three times in a quiet room. TrackType: SFX","seconds":3}' \
  -o stable-audio-sfx.wav
```

两个服务都保留旧请求格式的兼容别名。脚本制作场景的提示词规范、
`prompt_en` 约束和 GPU 示例见 [`soundEffect/README.md`](soundEffect/README.md) 与
[`soundEffect/声效提示词说明.md`](soundEffect/声效提示词说明.md)。

## Step-Audio-EditX 编辑

8331 自己负责 prompt 音频的上传、检查和编辑，不经过 8300 代理。先上传，再调用编辑：

```bash
curl -X POST http://127.0.0.1:8331/v1/upload_audio \
  -F 'audio=@line-1.wav' \
  -F 'full_path=step-audio-editx/line-1.wav'

curl -X POST http://127.0.0.1:8331/v1/stepAudioEditx/edit \
  -H 'Content-Type: application/json' \
  -d '{"prompt_audio":"step-audio-editx/line-1.wav","prompt_text":"这是一条台词。","generated_text":"这是一条台词。","edit_type":"emotion","edit_info":"coldness"}' \
  -o edited.wav
```

`edit_type` 可为 `emotion`、`style`、`paralinguistic`、`denoise`、`vad` 或 `speed`。
`emotion`、`style` 和 `speed` 需要非空 `edit_info`；`denoise` 与 `vad` 不要求文本；
其他编辑类型需要与 prompt 音频匹配的 `prompt_text`。请求字段映射到上游命令的同名
编辑语义，输出保存到 `STEP_AUDIO_EDITX_OUTPUT_DIR`（默认 `storage/clone/`）。

## 健康检查与错误语义

- 健康检查不会加载模型；`available`、`paths`、`runtime` 和 `last_errors` 用于区分依赖、
  权重、CUDA、worker 和配置问题。
- 请求校验失败通常返回 `422`；上传后找不到参考音频通常返回 `404`。
- `POST /internal/unload_all` 只允许本机访问，外部请求返回 `403`。
- 模型 worker 失败或超时会清理临时文件和进程组，再返回服务错误；共享 GPU 锁在
  `finally` 中释放。
- 生成接口响应体是 WAV，同时会写入语义对应的输出目录；这些文件不会自动清理。

## 测试与开发

根目录回归测试不下载权重、不调用外部服务、不需要 CUDA：

```bash
bash -n start.sh
uv run --project qwen3_tts python -m unittest discover -s tests -v
```

Stable Audio 的服务内测试需要从它自己的目录运行，否则 `test_migration.py` 无法解析
同目录的 `runtime.py`：

```bash
(cd stable_audio_3_medium && uv run --project . python -m unittest discover -s tests -v)
```

MOSS 的真实 CUDA/权重 smoke test 是独立流程：

```bash
bash soundEffect/run_moss_soundeffect_v2.sh
```

修改请求契约、路由、存储解析或 worker 生命周期时，应同步添加/更新对应的 no-model
测试，并在 README 中更新兼容字段。不要提交模型权重、上传音频、生成 WAV、缓存、虚拟
环境、密钥或机器专用绝对路径。

各服务的依赖、单服务启动和模型专用配置可继续参考目录内 README：
`mimo_tts/README.md`、`Step_Audio_EditX/README.md`、`LongCat_AudioDiT_3.5B_bf16/README.md`、
`dots_tts_soar/README.md`、`moss_soundEffect/README.md` 和 `stable_audio_3_medium/README.md`。
