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
