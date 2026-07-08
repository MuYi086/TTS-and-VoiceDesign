# Unitale AI Local Backend

本项目是 Unitale 前端使用的本地后端，整合：

- IndexTTS2：参考音频 + 文本合成
- dots.tts-base：参考音频 + 文本合成，独立暴露 `8301`
- LongCat-AudioDiT-1B：参考音频 + 文本合成，独立暴露 `8302`
- MOSS-TTS-Local-Transformer-v1.5：参考音频 + 文本合成，独立暴露 `8303`
- Qwen3-TTS VoiceDesign：根据音色描述生成参考音频
- MiMo TTS VoiceDesign：根据音色描述生成参考音频，走主 API 的 `/v1/mimo/design`

## 本地环境

当前本机已创建专用 conda 环境：

```bash
conda activate unitale-tts-local
```

主 API、`8301` 的 dots HTTP 包装器、`8302` 的 LongCat HTTP 包装器、`8303` 的 MOSS HTTP 包装器、IndexTTS2 和 Qwen 子进程使用同一个 conda 环境启动。由于 IndexTTS2 需要
`transformers==4.52.1/tokenizers==0.21.0`，而 Qwen3-TTS 需要更新版本，Qwen 依赖被侧载到：

```text
vendor/qwen_libs
vendor/LongCat-AudioDiT
```

该目录只会在 Qwen 子进程中加入 `sys.path`，不会污染 IndexTTS2 主进程。Qwen 和 IndexTTS2 都是请求到来时加载，请求结束后卸载。

`dots.tts-base` 的真实推理不在 `unitale-tts-local` 里执行，而是由 `8301` 服务按请求调用：

```bash
conda run -n dots_tts python dots_tts_worker.py ...
```

因此 `dots_tts` 环境至少需要安装 `rednote-hilab/dots.tts`、`torch`、`numpy`、`soundfile`；不要求安装 `fastapi`。

`LongCat-AudioDiT-1B` 的真实推理不在 `unitale-tts-local` 里执行，而是由 `8302` 服务按请求调用：

```bash
conda run -n longcat_audiodit python longcat_audiodit_worker.py ...
```

因此 `longcat_audiodit` 环境至少需要安装 LongCat 运行时依赖：`torch`、`numpy`、`soundfile`、`librosa`、`transformers`、`funasr`。
`audiodit` 源码默认从当前项目的 `vendor/LongCat-AudioDiT` 读取；只有你想覆盖默认实现时，才需要额外设置 `LONGCAT_REPO_PATH` 或 `PYTHONPATH`。若 WebUI 只上传参考音频而不提供 `prompt_text`，`8302` 会自动调用本地 `SenseVoiceSmall` 离线生成转写 sidecar，再交给 LongCat 做克隆。
不要求在该环境里安装 `fastapi`，也不再依赖别的项目目录。

`MOSS-TTS-Local-Transformer-v1.5` 的真实推理不在 `unitale-tts-local` 里执行，而是由 `8303` 服务按请求调用：

```bash
conda run -n moss-tts-py310 python moss_tts_worker.py ...
```

因此 `moss-tts-py310` 环境至少需要安装 OpenMOSS/MOSS-TTS 官方本地运行依赖：`torch`、`torchaudio`、`transformers`。`8303` 的 worker 会复用 `~/github/timbre-design/scripts/tts_local_moss_tts_local_transformer.py` 里已经验证过的本地 helper，因此该脚本需要存在，且其依赖版本要与 `moss-tts-py310` 环境匹配。不要求在该环境里安装 `fastapi`。

```bash
export MIMO_API_KEY=...
```

MiMo 是云端 API，不加载本地模型；默认使用 `https://api.xiaomimimo.com/v1` 和 `mimo-v2.5-tts-voicedesign`。

## 模型路径

默认使用以下本地模型目录：

```text
/home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
/home/muyi086/hf-mirror/IndexTeam/IndexTTS-2
/home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/hf_cache
/home/muyi086/hf-mirror/rednote-hilab/dots.tts-base
/home/muyi086/hf-mirror/meituan-longcat/LongCat-AudioDiT-1B
/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2
/home/muyi086/hf-mirror/google/umt5-base
/home/muyi086/hf-mirror/FunAudioLLM/SenseVoiceSmall
/home/muyi086/github/TTS-and-VoiceDesign/vendor/LongCat-AudioDiT
/home/muyi086/github/timbre-design/scripts/tts_local_moss_tts_local_transformer.py
```

`hf_cache` 内包含 IndexTTS2 辅助模型：`w2v-bert-2.0`、`semantic_codec`、`campplus`、`bigvgan`。

## 启动

```bash
bash start.sh
```

默认监听：

```text
http://127.0.0.1:8300
http://127.0.0.1:8301
http://127.0.0.1:8302
http://127.0.0.1:8303
```

健康检查：

```bash
curl http://127.0.0.1:8300/v1/health
curl http://127.0.0.1:8301/v1/health
curl http://127.0.0.1:8302/v1/health
curl http://127.0.0.1:8303/v1/health
```

`indextts_ready=true` 且 `missing.indextts_main=[]`、`missing.indextts_aux=[]` 表示本地文件完整。
`8302` 的健康检查还会返回 `longcat_repo_path`、`longcat_asr_model_dir` 和自动转写参数。正常情况下 `longcat_repo_path` 应指向当前项目的 `vendor/LongCat-AudioDiT`，`longcat_asr_model_dir` 应指向本地 `SenseVoiceSmall`；如果这里为空，再检查 `vendor` 或 `hf-mirror` 是否完整。
`8303` 的健康检查会返回 `moss_helper_script`、`moss_model_dir` 和 `moss_codec_path`。若 `moss_helper_script` 或 `moss_model_dir` 不可用，先检查 `~/github/timbre-design` 和本地 `hf-mirror`。

## 常用接口

生成参考音色：

```bash
curl -X POST http://127.0.0.1:8300/v1/qwen/design \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o qwen_test.wav
```

```bash
curl -X POST http://127.0.0.1:8300/v1/mimo/design \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o mimo_test.wav
```

上传参考音频：

```bash
curl -X POST http://127.0.0.1:8300/v1/upload_audio \
  -F "full_path=qwen_test.wav" \
  -F "audio=@qwen_test.wav"
```

合成音频：

```bash
curl -X POST http://127.0.0.1:8300/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"这是一次本地合成测试。","audio_path":"qwen_test.wav"}' \
  -o synth.wav
```

`8301` 的 `dots.tts-base` 复用同一套 WebUI TTS 协议：

```bash
curl -X POST http://127.0.0.1:8301/v1/upload_audio \
  -F "full_path=qwen_test.wav" \
  -F "audio=@qwen_test.wav" \
  -F "prompt_text=这是参考音频的准确转写，可选但建议提供"
```

```bash
curl -X POST http://127.0.0.1:8301/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"这是一次 dots.tts 本地合成测试。","audio_path":"qwen_test.wav"}' \
  -o dots_synth.wav
```

如果未提供 `prompt_text`，`dots.tts-base` 仍会执行基于参考音频的克隆，但通常比“参考音频 + 准确转写”质量更弱。当前 WebUI 现有 TTS 配置只会自动上传音频，因此 `8301` 默认走这个兼容降级路径；若你后续愿意扩展 WebUI，可在上传时额外提交 `prompt_text`。

`8302` 的 `LongCat-AudioDiT-1B` 复用同一套 WebUI TTS 协议：

```bash
curl -X POST http://127.0.0.1:8302/v1/upload_audio \
  -F "full_path=qwen_test.wav" \
  -F "audio=@qwen_test.wav" \
  -F "prompt_text=这是参考音频的准确转写，可选但建议提供"
```

```bash
curl -X POST http://127.0.0.1:8302/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"这是一次 LongCat 本地合成测试。","audio_path":"qwen_test.wav"}' \
  -o longcat_synth.wav
```

如果未提供 `prompt_text`，`8302` 会先对参考音频做一次本地离线自动转写，并把结果保存为 sidecar；后续同名音频再次合成时会复用该转写，不再重复跑 ASR。若你手头已有更准确的人工转写，仍然建议在上传时显式传 `prompt_text` 覆盖自动结果。

`8303` 的 `MOSS-TTS-Local-Transformer-v1.5` 复用同一套 WebUI TTS 协议：

```bash
curl -X POST http://127.0.0.1:8303/v1/upload_audio \
  -F "full_path=qwen_test.wav" \
  -F "audio=@qwen_test.wav"
```

```bash
curl -X POST http://127.0.0.1:8303/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"这是一次 MOSS 本地合成测试。","audio_path":"qwen_test.wav"}' \
  -o moss_synth.wav
```

`8303` 的 MOSS 克隆只依赖参考音频，不强制要求 `prompt_text`。如果你希望覆盖默认推理参数，也可以在 `v2/synthesize` 请求里附带 `language`、`instruction`、`quality`、`tokens`、`max_new_tokens` 等可选字段。

## 运行策略

- `8300` 的 Qwen 和 IndexTTS2 不同时驻留显存。
- `8300 /v1/qwen/design` 请求到来时加载 Qwen，返回音频前卸载 Qwen。
- `8300 /v1/mimo/design` 走 MiMo 云 API，请求前会先卸载已驻留的 Qwen / IndexTTS2。
- `8300` 内部通过共享 GPU 锁串行执行 Qwen / MiMo / IndexTTS2，避免本地模型并发占显存。
- `8300 /v2/synthesize` 会先卸载 Qwen，再按需加载 IndexTTS2，合成结束后卸载 IndexTTS2。
- `8301 /v2/synthesize` 是轻量 HTTP 包装器；每个请求都会临时拉起 `dots_tts` 环境里的 worker，worker 退出即释放模型和显存。
- `8302 /v2/synthesize` 是轻量 HTTP 包装器；每个请求都会临时拉起 `longcat_audiodit` 环境里的 worker，worker 退出即释放模型和显存。
- `8303 /v2/synthesize` 是轻量 HTTP 包装器；每个请求都会临时拉起 `moss-tts-py310` 环境里的 worker，worker 退出即释放 MOSS 模型、codec 和显存。
- `8300`、`8301`、`8302`、`8303` 共享同一个 `GPU_LOCK_FILE`，因此 Qwen / MiMo / IndexTTS2 / dots.tts / LongCat / MOSS 不会并发抢占显存。
- 默认离线加载模型：`LOCAL_FILES_ONLY=1`。
- 不再执行云端脚本里的 apt 改源、`/app` 代码同步或清理所有 Python 进程。
