# Unitale AI Local Backend

本项目是 Unitale 前端使用的本地后端，整合：

- IndexTTS2：参考音频 + 文本合成
- Qwen3-TTS VoiceDesign：根据音色描述生成参考音频

## 本地环境

当前本机已创建专用 conda 环境：

```bash
conda activate unitale-tts-local
```

主 API、IndexTTS2 和 Qwen 守护进程使用同一个 conda 环境启动。由于 IndexTTS2 需要
`transformers==4.52.1/tokenizers==0.21.0`，而 Qwen3-TTS 需要更新版本，Qwen 依赖被侧载到：

```text
vendor/qwen_libs
```

该目录只会在 Qwen 子进程中加入 `sys.path`，不会污染 IndexTTS2 主进程。

## 模型路径

默认使用以下本地模型目录：

```text
/home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
/home/muyi086/hf-mirror/IndexTeam/IndexTTS-2
/home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/hf_cache
```

`hf_cache` 内包含 IndexTTS2 辅助模型：`w2v-bert-2.0`、`semantic_codec`、`campplus`、`bigvgan`。

## 启动

```bash
bash start.sh
```

默认监听：

```text
http://127.0.0.1:8300
```

健康检查：

```bash
curl http://127.0.0.1:8300/v1/health
```

`indextts_ready=true` 且 `missing.indextts_main=[]`、`missing.indextts_aux=[]` 表示本地文件完整。

## 常用接口

生成参考音色：

```bash
curl -X POST http://127.0.0.1:8300/v1/qwen/design \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o qwen_test.wav
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

## 运行策略

- Qwen 和 IndexTTS2 不同时驻留显存。
- 调用 `/v1/qwen/design` 前会卸载 IndexTTS2。
- 调用 `/v2/synthesize` 前会卸载 Qwen，再按需加载 IndexTTS2。
- 默认离线加载模型：`LOCAL_FILES_ONLY=1`。
- 不再执行云端脚本里的 apt 改源、`/app` 代码同步或清理所有 Python 进程。
