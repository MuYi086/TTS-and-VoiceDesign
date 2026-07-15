# Unitale AI Local Backend

本项目是 Unitale 前端使用的本地后端，整合：

- IndexTTS2：参考音频 + 文本合成
- dots.tts-base：参考音频 + 文本合成，独立暴露 `8301`
- LongCat-AudioDiT-1B：参考音频 + 文本合成，独立暴露 `8302`
- MOSS-TTS-Local-Transformer-v1.5：参考音频 + 文本合成，独立暴露 `8303`
- MOSS-SoundEffect v2.0：根据中英文提示词生成 48 kHz 声效，独立暴露 `8311`
- OmniVoice：参考音频 + 文本合成，独立暴露 `8304`
- Qwen3-TTS-12Hz-1.7B-Base：参考音频 + 文本合成，独立暴露 `8305`
- VoxCPM2：参考音频 + 文本合成，独立暴露 `8306`
- Qwen3-TTS VoiceDesign：根据音色描述生成参考音频
- MiMo TTS VoiceDesign：根据音色描述生成参考音频，走主 API 的 `/v1/mimo/design`

运行时 API、各模型 worker 和共享音频处理模块统一放在 `api/`；运行资源也随之归档在 `api/prompts/`、`api/.cache/`、`api/vendor/`。根目录只保留启动入口 `start.sh`、文档、测试和其他项目内容；`start.sh` 仍从根目录启动，所有端口和接口保持不变。

## 本地环境

当前本机已创建专用 conda 环境：

```bash
conda activate unitale-tts-local
```

主 API、`8301` 的 dots HTTP 包装器、`8302` 的 LongCat HTTP 包装器、`8303` 的 MOSS HTTP 包装器、`8304` 的 OmniVoice HTTP 包装器、`8305` 的 Qwen3-TTS HTTP 包装器、`8306` 的 VoxCPM2 HTTP 包装器、IndexTTS2 和 Qwen 子进程使用同一个 conda 环境启动。由于 IndexTTS2 需要
`transformers==4.52.1/tokenizers==0.21.0`，而 Qwen3-TTS 需要更新版本，Qwen 依赖被侧载到：

```text
api/vendor/qwen_libs
api/vendor/LongCat-AudioDiT
```

该目录只会在 Qwen 子进程中加入 `sys.path`，不会污染 IndexTTS2 主进程。Qwen 和 IndexTTS2 都是请求到来时加载，请求结束后卸载。

`dots.tts-base` 的真实推理不在 `unitale-tts-local` 里执行，而是由 `8301` 服务按请求调用：

```bash
conda run -n dots_tts python api/dots_tts_worker.py ...
```

因此 `dots_tts` 环境至少需要安装 `rednote-hilab/dots.tts`、`torch`、`numpy`、`soundfile`；不要求安装 `fastapi`。

`LongCat-AudioDiT-1B` 的真实推理不在 `unitale-tts-local` 里执行，而是由 `8302` 服务按请求调用：

```bash
conda run -n longcat_audiodit python api/longcat_audiodit_worker.py ...
```

因此 `longcat_audiodit` 环境至少需要安装 LongCat 运行时依赖：`torch`、`numpy`、`soundfile`、`librosa`、`transformers`、`funasr`。
`audiodit` 源码默认从当前项目的 `api/vendor/LongCat-AudioDiT` 读取；只有你想覆盖默认实现时，才需要额外设置 `LONGCAT_REPO_PATH` 或 `PYTHONPATH`。若 WebUI 只上传参考音频而不提供 `prompt_text`，`8302` 会自动调用本地 `SenseVoiceSmall` 离线生成转写 sidecar，再交给 LongCat 做克隆。
不要求在该环境里安装 `fastapi`，也不再依赖别的项目目录。

`MOSS-TTS-Local-Transformer-v1.5` 的真实推理不在 `unitale-tts-local` 里执行，而是由 `8303` 服务按请求调用：

```bash
conda run -n moss-tts-py310 python api/moss_tts_worker.py ...
```

因此 `moss-tts-py310` 环境至少需要安装 OpenMOSS/MOSS-TTS 官方本地运行依赖：`torch`、`torchaudio`、`transformers`。`8303` 的 worker 会复用 `~/github/timbre-design/modelScript/tts_local_moss_tts_local_transformer.py` 里已经验证过的本地 helper，因此该脚本需要存在，且其依赖版本要与 `moss-tts-py310` 环境匹配。不要求在该环境里安装 `fastapi`。

`OmniVoice` 的真实推理不在 `unitale-tts-local` 里执行，而是由 `8304` 服务按请求调用：

```bash
conda run -n omnivoice python api/omnivoice_tts_worker.py ...
```

因此 `omnivoice` 环境至少需要安装 OmniVoice 官方运行时依赖：`omnivoice`、`torch`、`numpy`、`soundfile`。若上传参考音频时没有同时提供 `prompt_text`，`8304` 会让 OmniVoice 在 worker 内部对参考音频执行一次自动转写；该转写相关模块同样只会在请求期间加载，worker 退出即释放。

`Qwen3-TTS-12Hz-1.7B-Base` 的真实推理不在 `unitale-tts-local` 里执行，而是由 `8305` 服务按请求调用：

```bash
conda run -n qwen3-tts python api/qwen3_tts_worker.py ...
```

因此 `qwen3-tts` 环境至少需要安装 `qwen-tts`、`torch`、`numpy`、`soundfile`。它使用参考脚本同一套克隆方式：有 `prompt_text` 时走 reference transcript 克隆；没有时退回 `x-vector-only` 模式，只依赖参考音频本身，不会额外加载 ASR。

`VoxCPM2` 的真实推理不在 `unitale-tts-local` 里执行，而是由 `8306` 服务按请求调用：

```bash
conda run -n voxcpm2 python api/voxcpm2_worker.py ...
```

因此 `voxcpm2` 环境至少需要安装 `voxcpm`、`torch`、`numpy`、`soundfile`。`8306` 的 worker 会复用 `~/github/timbre-design/modelScript/tts_local_voxcpm2.py` 里已经验证过的本地 helper，因此该脚本需要存在，且其依赖版本要与 `voxcpm2` 环境匹配。它同样满足“真实用到才加载，请求结束即卸载”：模型只在 worker 进程内按请求加载，worker 退出后显存立即清理。若未提供 `prompt_text`，`8306` 会走仅参考音频的克隆模式，不会额外加载 ASR。

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
/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0
/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2
/home/muyi086/hf-mirror/k2-fsa/OmniVoice
/home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-Base
/home/muyi086/hf-mirror/openbmb/VoxCPM2
/home/muyi086/hf-mirror/google/umt5-base
/home/muyi086/hf-mirror/FunAudioLLM/SenseVoiceSmall
/home/muyi086/github/TTS-and-VoiceDesign/api/vendor/LongCat-AudioDiT
/home/muyi086/github/timbre-design/modelScript/tts_local_moss_tts_local_transformer.py
/home/muyi086/github/timbre-design/modelScript/tts_local_voxcpm2.py
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
http://127.0.0.1:8311
http://127.0.0.1:8304
http://127.0.0.1:8305
http://127.0.0.1:8306
```

健康检查：

```bash
curl http://127.0.0.1:8300/v1/health
curl http://127.0.0.1:8301/v1/health
curl http://127.0.0.1:8302/v1/health
curl http://127.0.0.1:8303/v1/health
curl http://127.0.0.1:8311/v1/health
curl http://127.0.0.1:8304/v1/health
curl http://127.0.0.1:8305/v1/health
curl http://127.0.0.1:8306/v1/health
```

`indextts_ready=true` 且 `missing.indextts_main=[]`、`missing.indextts_aux=[]` 表示本地文件完整。
`8302` 的健康检查还会返回 `longcat_repo_path`、`longcat_asr_model_dir` 和自动转写参数。正常情况下 `longcat_repo_path` 应指向当前项目的 `api/vendor/LongCat-AudioDiT`，`longcat_asr_model_dir` 应指向本地 `SenseVoiceSmall`；如果这里为空，再检查 `api/vendor` 或 `hf-mirror` 是否完整。
`8303` 的健康检查会返回 `moss_helper_script`、`moss_model_dir` 和 `moss_codec_path`。若 `moss_helper_script` 或 `moss_model_dir` 不可用，先检查 `~/github/timbre-design` 和本地 `hf-mirror`。
`8304` 的健康检查会返回 `omnivoice_model_dir`、`device_map`、`dtype` 和 `prompt_text_fallback`。若 `omnivoice_model_dir` 不可用，先检查本地 `hf-mirror/k2-fsa/OmniVoice`。
`8305` 的健康检查会返回 `qwen3_tts_model_dir`、`device_map`、`dtype`、`attn_implementation` 和 `prompt_text_fallback`。若 `qwen3_tts_model_dir` 不可用，先检查本地 `hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-Base`。
`8306` 的健康检查会返回 `voxcpm2_model_dir`、`voxcpm2_helper_script`、`device` 和 `prompt_text_fallback`。若 `voxcpm2_model_dir` 或 `voxcpm2_helper_script` 不可用，先检查本地 `hf-mirror/openbmb/VoxCPM2` 与 `~/github/timbre-design/modelScript/tts_local_voxcpm2.py`。

## 本地回归测试

测试依赖主运行环境中的 `numpy`、`torch`、FastAPI 和各 API 的验证模型，但不会下载权重、调用外部服务或加载 TTS 模型。请从项目根目录执行：

```bash
conda run -n unitale-tts-local python -m unittest discover -s tests -v
```

当前测试覆盖共享前导静音裁剪逻辑，以及所有语音克隆服务拒绝 `style_prompt` 的 API 契约。若使用了不同的主环境名称，请将命令中的 `unitale-tts-local` 替换为 `CONDA_ENV` 的值。

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

所有 `POST /v2/synthesize` 都是参考音频克隆接口，不接受 `style_prompt`（字段出现即返回 `422`，包括值为 `null` 的情况）。音色/风格应在生成参考音频阶段通过 `/v1/qwen/design` 或 `/v1/mimo/design` 的 `voice_description` 决定；合成阶段只朗读 `text`。

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

`8304` 的 `OmniVoice` 复用同一套 WebUI TTS 协议：

```bash
curl -X POST http://127.0.0.1:8304/v1/upload_audio \
  -F "full_path=qwen_test.wav" \
  -F "audio=@qwen_test.wav" \
  -F "prompt_text=这是参考音频的准确转写，可选但建议提供"
```

```bash
curl -X POST http://127.0.0.1:8304/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"这是一次 OmniVoice 本地合成测试。","audio_path":"qwen_test.wav"}' \
  -o omnivoice_synth.wav
```

如果未提供 `prompt_text`，`8304` 会在 worker 内部调用 OmniVoice 的参考音频自动转写流程，再继续做克隆。这仍然满足“真实用到才加载、请求结束即卸载”的约束，只是首轮请求通常比显式提供转写更慢。

`8305` 的 `Qwen3-TTS-12Hz-1.7B-Base` 也复用同一套 WebUI TTS 协议：

```bash
curl -X POST http://127.0.0.1:8305/v1/upload_audio \
  -F "full_path=qwen_test.wav" \
  -F "audio=@qwen_test.wav" \
  -F "prompt_text=这是参考音频的准确转写，可选但建议提供"
```

```bash
curl -X POST http://127.0.0.1:8305/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"这是一次 Qwen3-TTS 本地合成测试。","audio_path":"qwen_test.wav"}' \
  -o qwen3_tts_synth.wav
```

如果未提供 `prompt_text`，`8305` 会退回 `x-vector-only` 克隆模式，不需要额外模型做自动转写；通常速度更稳定，但音色一致性一般不如“参考音频 + 准确转写”。默认还会裁掉生成结果前导静音，避免开头先空几秒再出声。

`8306` 的 `VoxCPM2` 也复用同一套 WebUI TTS 协议：

```bash
curl -X POST http://127.0.0.1:8306/v1/upload_audio \
  -F "full_path=qwen_test.wav" \
  -F "audio=@qwen_test.wav" \
  -F "prompt_text=这是参考音频的准确转写，可选但建议提供"
```

```bash
curl -X POST http://127.0.0.1:8306/v2/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"这是一次 VoxCPM2 本地合成测试。","audio_path":"qwen_test.wav"}' \
  -o voxcpm2_synth.wav
```

如果未提供 `prompt_text`，`8306` 会直接走仅参考音频的克隆模式，不需要额外转写模型；若你需要更稳定的音色一致性，仍然建议在上传时同时提交参考音频的准确转写。你可以在 `v2/synthesize` 请求里附带 `cfg_value`、`inference_timesteps`、`load_denoiser`、`optimize`、`seed`、`device` 等可选字段覆盖默认参数。`8306` 默认使用 `VOXCPM2_CONDA_ENV=voxcpm2` 启动 API 和 worker，默认 `device=cuda`，与 `step_3_tts_local_voxcpm2.py` 的运行环境和核心参数保持一致。

## 运行策略

- `8300` 的 Qwen 和 IndexTTS2 不同时驻留显存。
- `8300 /v1/qwen/design` 请求到来时加载 Qwen，返回音频前卸载 Qwen。
- `8300 /v1/mimo/design` 走 MiMo 云 API，请求前会先卸载已驻留的 Qwen / IndexTTS2。
- `8300` 内部通过共享 GPU 锁串行执行 Qwen / MiMo / IndexTTS2，避免本地模型并发占显存。
- `8300 /v2/synthesize` 会先卸载 Qwen，再按需加载 IndexTTS2，合成结束后卸载 IndexTTS2。
- `8301 /v2/synthesize` 是轻量 HTTP 包装器；每个请求都会临时拉起 `dots_tts` 环境里的 worker，worker 退出即释放模型和显存。
- `8302 /v2/synthesize` 是轻量 HTTP 包装器；每个请求都会临时拉起 `longcat_audiodit` 环境里的 worker，worker 退出即释放模型和显存。
- `8303 /v2/synthesize` 是轻量 HTTP 包装器；每个请求都会临时拉起 `moss-tts-py310` 环境里的 worker，worker 退出即释放 MOSS 模型、codec 和显存。
- `8311 /v1/generate` 是 MOSS-SoundEffect v2.0 的轻量 HTTP 包装器；每个请求都会在 `moss-soundEffect` 环境中启动独立 worker，worker 退出才向调用方返回音频，因此模型、CUDA 上下文和显存不会在 8311 常驻。
- `8304 /v2/synthesize` 是轻量 HTTP 包装器；每个请求都会临时拉起 `omnivoice` 环境里的 worker，worker 退出即释放 OmniVoice 模型、参考音色 prompt 和显存。
- `8305 /v2/synthesize` 是轻量 HTTP 包装器；每个请求都会临时拉起 `qwen3-tts` 环境里的 worker，worker 退出即释放 Qwen3-TTS Base 模型、voice clone prompt 和显存。
- `8306 /v2/synthesize` 是轻量 HTTP 包装器；每个请求都会临时拉起 `voxcpm2` 环境里的 worker，worker 退出即释放 VoxCPM2 模型和显存。
- `8300`、`8301`、`8302`、`8303`、`8304`、`8305`、`8306`、`8311` 共享同一个 `GPU_LOCK_FILE`，因此 Qwen / MiMo / IndexTTS2 / dots.tts / LongCat / MOSS / MOSS-SoundEffect / OmniVoice / Qwen3-TTS Base / VoxCPM2 不会并发抢占显存。
- 默认离线加载模型：`LOCAL_FILES_ONLY=1`。
- 不再执行云端脚本里的 apt 改源、`/app` 代码同步或清理所有 Python 进程。


## MOSS-SoundEffect v2.0 API

启动 `bash start.sh` 后，声效服务默认在 `8311` 监听。它只接受描述非语言声效的 `prompt`，不依赖参考音频：

```bash
curl -X POST http://127.0.0.1:8311/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"深夜的旧木门被缓慢推开，门轴发出低沉、略带生锈的连续吱呀声，安静室内近距离收音。","seconds":6}' \
  -o door_creak.wav
```

可选参数：`seconds`（大于 0 且不超过 30，默认 10）、`num_inference_steps`（默认 100）、`cfg_scale`（默认 4.0）、`sigma_shift`（默认 5.0）、`seed`、`device` 与 `torch_dtype`。为兼容本项目既有的合成调用命名，`POST /v2/synthesize` 是同一请求模型的别名；新接入优先使用 `/v1/generate`。

默认使用 `MOSS_SOUNDEFFECT_CONDA_ENV=moss-soundEffect` 和本地权重目录 `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0`。可通过 `MOSS_SOUNDEFFECT_*` 环境变量覆盖模型路径、默认参数、请求超时、设备和精度。模型只存在于每个请求创建的 worker 进程中；worker 退出后才释放共享 GPU 锁，确保显存已释放后其他 TTS/声效任务才会进入。
