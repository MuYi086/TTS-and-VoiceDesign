# Unitale AI Local Backend

本项目是 Unitale 前端使用的本地后端，当前提供：

- IndexTTS2：参考音频 + 文本合成，端口 `8300`
- Qwen3-TTS-12Hz-1.7B-Base：参考音频 + 文本合成，端口 `8305`
- VoxCPM2：参考音频 + 文本合成，端口 `8306`
- MOSS-SoundEffect v2.0：根据中英文提示词生成 48 kHz 声效，端口 `8311`
- Qwen3-TTS VoiceDesign：根据音色描述生成参考音频，走主 API 的 `/v1/qwen/design`
- MiMo TTS VoiceDesign：根据音色描述生成参考音频，走主 API 的 `/v1/mimo/design`

运行时 API、各模型 worker 和共享音频处理模块统一位于 `api/`；上传资源、缓存和供应商代码位于 `api/prompts/`、`api/.cache/` 与 `api/vendor/`。VoxCPM2 的成功合成结果会额外保留在 `api/tempAudio/`，可通过 `VOXCPM2_OUTPUT_DIR` 覆盖。不要把生成音频或模型权重提交到 Git。

## 本地环境

主 API 与 IndexTTS2 worker 使用：

```bash
conda activate unitale-tts-local
```

Qwen3-TTS 和 VoxCPM2 各自在请求期间由对应 Conda 环境拉起一次性 worker。模型在请求结束后由 worker 退出释放显存；主 API、各包装器和 worker 共享 `GPU_LOCK_FILE`，避免并发抢占 GPU。

```bash
conda run -n unitale-tts-local python api/indextts_worker.py ...
conda run -n qwen3-tts python api/qwen3_tts_worker.py ...
conda run -n voxcpm2 python api/voxcpm2_worker.py ...
```

MOSS-SoundEffect 使用独立的 `moss-soundEffect` 环境。MiMo 是云端 API，须通过环境变量提供密钥：

```bash
export MIMO_API_KEY=...
```

MiMo 是云端服务，运行后端的机器必须能连接 `https://api.xiaomimimo.com:443`。若网络需要代理，请在启动 `start.sh` 前设置标准代理环境变量，例如 `HTTPS_PROXY=http://127.0.0.1:7890`（并按需设置 `NO_PROXY=127.0.0.1,localhost`）。网络不可达时 `/v1/mimo/design` 会返回 `503` 及可操作的错误说明；这不会影响本地 Qwen 音色设计或其他本地 TTS 接口。

## 模型路径

默认读取以下本地目录；可在启动前用同名环境变量覆盖：

```text
/home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
/home/muyi086/hf-mirror/IndexTeam/IndexTTS-2
/home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/hf_cache
/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0
/home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-Base
/home/muyi086/hf-mirror/openbmb/VoxCPM2
/home/muyi086/github/TTS-and-VoiceDesign/api/voxcpm2_helpers.py
```

`hf_cache` 内包含 IndexTTS2 辅助模型：`w2v-bert-2.0`、`semantic_codec`、`campplus` 和 `bigvgan`。

## 启动与健康检查

```bash
bash start.sh
curl http://127.0.0.1:8300/v1/health
curl http://127.0.0.1:8305/v1/health
curl http://127.0.0.1:8306/v1/health
curl http://127.0.0.1:8311/v1/health
```

默认服务地址：

```text
http://127.0.0.1:8300  IndexTTS2 与音色设计
http://127.0.0.1:8305  Qwen3-TTS-12Hz-1.7B-Base
http://127.0.0.1:8306  VoxCPM2
http://127.0.0.1:8311  MOSS-SoundEffect v2.0
```

## 语音合成接口

三个语音合成服务均支持以下流程：

1. `POST /v1/upload_audio` 上传参考音频。
2. `GET /v1/check/audio?file_name=...` 确认后端已保存。
3. `POST /v2/synthesize` 生成目标音频。

所有 `/v2/synthesize` 仅做参考音频克隆，不接受 `style_prompt`；音色或风格应在生成参考音频阶段通过 Qwen 或 MiMo 的音色设计接口确定。

| 服务 | `prompt_text` 处理 |
| --- | --- |
| `8300` IndexTTS2 | 不使用参考转写，只接收参考音频与情绪向量。 |
| `8305` Qwen3-TTS Base | 映射为官方 `ref_text`；缺失时回退到仅参考音频的克隆。 |
| `8306` VoxCPM2 | `clone_mode="ultimate"` 有准确 `prompt_text` 时走 Ultimate Cloning；`clone_mode="controllable"` 只接受 `control_instruction`，不接受 `prompt_text`，并将指令写入目标文本前；未指定模式时保留旧的参考文本 / 仅参考音频兼容路径。 |

VoxCPM2 的 `ultimate` 与 `controllable` 请求路径严格互斥：前者用于最大化复刻参考音频细节，后者用于按短控制指令调整表演节奏和情绪。`control_instruction` 不是响度参数；成片响度应在合成后检测和统一归一化。

`8306` 还支持 `nonverbal_tags`（数组，最多一个）。仅接受官方标签 `laughing`、`sigh`、`Uhm`、`Shh`、`Question-ah`、`Question-ei`、`Question-en`、`Question-oh`、`Surprise-wa`、`Surprise-yo`、`Dissatisfaction-hnn`，且只能配合 `clone_mode="controllable"` 使用。worker 会把最终目标文本拼为 `(control_instruction)[tag]正文`（无控制或标签时省略相应前缀），并在每个文本分片调用模型前向终端打印该最终文本、分片序号和克隆模式；不会打印参考音频转写。

VoxCPM2 可直接在 [`api/voxcpm2_api.py`](api/voxcpm2_api.py) 顶部修改集中默认值：`cfg_value`、`inference_timesteps`、`normalize`、`denoise`、`retry_badcase`、`load_denoiser`、`optimize`、`device`、`seed`、分片长度、分片停顿和超时。`start.sh` 不再写入这些默认值；如启动前显式设置同名 `VOXCPM2_*` 环境变量，环境变量仍会覆盖代码默认值。`denoise=true` 时会自动启用 `load_denoiser`。

`8306` 每次合成成功后都会保留一份原始 WAV 到 `api/tempAudio/`，文件名形如 `voxcpm2_20260730_120000_xxxxx.wav`；接口响应内容不变。此目录不会自动清理，完成后请按需要转移或删除文件。

```bash
curl -X POST http://127.0.0.1:8306/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"唉，还是晚了一步。","audio_path":"reference.wav","clone_mode":"controllable","control_instruction":"自然、清晰地表达，保留必要的非语言反应，吐字清晰","nonverbal_tags":["sigh"]}' \
  -o synth.wav
```

音色设计端点：

```bash
curl -X POST http://127.0.0.1:8300/v1/qwen/design \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o qwen_reference.wav

curl -X POST http://127.0.0.1:8300/v1/mimo/design \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o mimo_reference.wav
```

`8311` 是独立的声效接口，不属于 WebUI 当前自动调用的 TTS 流程；生成的音频可手动导入前端 SFX 素材库。

## 本地回归测试

测试不会下载权重、调用外部服务或加载 TTS 模型：

```bash
conda run -n unitale-tts-local python -m unittest discover -s tests -v
```

当前测试覆盖共享音频处理、GPU worker 生命周期、参考文本接口契约，以及 IndexTTS2、Qwen3-TTS、VoxCPM2 与 SoundEffect 的共享运行时约束。
