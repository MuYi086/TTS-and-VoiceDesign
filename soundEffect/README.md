# MOSS-SoundEffect v2.0 本地测试

这里使用的是 `OpenMOSS-Team/MOSS-SoundEffect-v2.0`，不是显存需求更高的 v1。v2 使用 1.3B DiT + Flow Matching，支持中英文声效提示词、48 kHz 输出和最长 30 秒的单次生成。

## 当前环境

```text
Conda 环境：moss-soundEffect
Python：3.12
模型包：moss-soundeffect-v2 0.1.0
Torch：2.9.0+cu128
Torchaudio：2.9.0+cu128
Transformers：4.57.1
Diffusers：0.37.1
```

环境已与上游推荐版本对齐：Torch 2.9 / CUDA 12.8。MOSS v2 管线导入、依赖一致性和宿主 CUDA 可见性均已验证。

## 运行

先在 `test_moss_soundeffect_v2.py` 顶部修改提示词、时长、步数和输出路径，然后直接执行：

```bash
bash soundEffect/run_moss_soundeffect_v2.sh
```

启动脚本会自动使用 `/home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0` 的本地完整权重，**不会**访问 Hugging Face 或重复下载。它也会自动进入 `moss-soundEffect` Conda 环境；默认输出文件为 `soundEffect/outputs/dog_barking_park.wav`。

如需换一套本地权重，可在执行前临时指定目录：

```bash
MOSS_SOUNDEFFECT_MODEL_DIR=/path/to/MOSS-SoundEffect-v2.0 \
  bash soundEffect/run_moss_soundeffect_v2.sh
```

上游大模型如何从小说提取声效并生成可直接传给 `PROMPT` 的文本，见 `声效提示词说明.md`。


## HTTP API（端口 8311）

`bash start.sh` 会同时启动 MOSS-SoundEffect v2.0 HTTP 服务。默认地址：

```text
http://127.0.0.1:8311
```

健康检查：

```bash
curl http://127.0.0.1:8311/v1/health
```

生成声效：

```bash
curl -X POST http://127.0.0.1:8311/v1/generate \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt":"清晨安静的林间小径，近处零星鸟鸣清脆，远处有轻微树叶在微风中沙沙作响。","seconds":8}' \\
  -o birds.wav
```

请求字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `prompt` | 必填 | 中英文非语言声效描述。 |
| `seconds` | `10` | 时长，范围 `(0, 30]`。 |
| `num_inference_steps` | `100` | 推理步数，越高通常越慢。 |
| `cfg_scale` | `4.0` | 提示词引导强度。 |
| `sigma_shift` | `5.0` | Flow Matching 采样参数。 |
| `seed` | `0` | 随机种子。 |
| `device` | `cuda` | 可按请求覆盖默认设备。 |
| `torch_dtype` | `bfloat16` | 可按请求覆盖默认精度。 |

`POST /v2/synthesize` 是相同请求格式的兼容别名。接口不接受参考音频，也不生成台词；请仅传递声效提示词。

### 显存生命周期

8311 本身不导入 MOSS-SoundEffect 模型。每次生成都在 `moss-soundEffect` 环境中启动一个独立 worker：加载模型、生成 WAV、写入临时文件、退出进程。HTTP 包装器确认 worker 已退出并等待短暂的 CUDA 释放间隔后，才释放和其他 TTS 服务共用的 `GPU_LOCK_FILE`。因此模型、CUDA 上下文和显存不会跨请求常驻，也不会与现有 TTS worker 并发抢占显存。

可选环境变量包括 `MOSS_SOUNDEFFECT_CONDA_ENV`、`MOSS_SOUNDEFFECT_MODEL_DIR`、`MOSS_SOUNDEFFECT_DEVICE`、`MOSS_SOUNDEFFECT_DTYPE`、`MOSS_SOUNDEFFECT_REQUEST_TIMEOUT` 与所有 `MOSS_SOUNDEFFECT_DEFAULT_*` 参数。
