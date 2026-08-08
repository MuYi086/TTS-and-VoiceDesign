还是有问题，可控克隆的音色和参考音频完全不一致。
这是旁白的音频文件: `api/prompts/7e83843c63bc87e3be21385485f14d6f.wav`
但是还有一个一摸一样转写文案的音频`api/prompts/8fa94e78daa3821dc331dfec173d6d2a.wav`,这个可能是遗留的上一次生成的克隆音色，实际再用的应该是'7e83843c63bc87e3be21385485f14d6f.wav'
然后这是浏览器请求信息:
curl --url 'http://127.0.0.1:8306/v2/synthesize' \
  -H 'Accept: */*' \
  -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8' \
  -H 'Connection: keep-alive' \
  -H 'Content-Type: application/json' \
  -H 'Origin: http://localhost:5502' \
  -H 'Referer: http://localhost:5502/' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Site: cross-site' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Windows"' \
  --data-raw '{"text":"我跑了。穿过房子时，门像枪声一样砰砰地关合。前门打不开了。","audio_path":"moss_voicegenerator_旁白_1786169289374.wav","backend":"voxcpm2","clone_mode":"controllable","nonverbal_tags":[],"control_instruction":"语速急促，呼吸发紧，带着匆忙逃命的紧张"}'

  返回的结果音频我在代码里写了同步保存,这是那份复制品：`api/tempAudio/voxcpm2_20260808_234019_nxo_3bry.wav`，可以明显听出音色完全不一致。

  这是服务端记录的日志信息:
  INFO:     Started server process [504363]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8300 (Press CTRL+C to quit)
INFO:     127.0.0.1:52836 - "GET /v1/check/audio?file_name=moss_voicegenerator_%E6%97%81%E7%99%BD_1786169289374.wav HTTP/1.1" 200 OK
INFO:     127.0.0.1:52836 - "OPTIONS /v2/synthesize HTTP/1.1" 200 OK
[GPU 锁] 等待进入: voxcpm2/synthesize
[GPU 锁] 已进入: voxcpm2/synthesize
[VoxCPM2] 启动 worker: env=voxcpm2
[VoxCPM2 worker] 模型目录: /home/muyi086/hf-mirror/openbmb/VoxCPM2
[VoxCPM2 worker] operation=clone
[VoxCPM2 worker] cfg_value=2.0, inference_timesteps=10
[VoxCPM2 worker] seed=20260614, normalize=False, denoise=False, retry_badcase=True, load_denoiser=False, optimize=False, local_files_only=True, device=cuda
[VoxCPM2 worker] 参考音频: /home/muyi086/github/TTS-and-VoiceDesign/api/prompts/7e83843c63bc87e3be21385485f14d6f.wav
[VoxCPM2 worker] 克隆模式: controllable
[VoxCPM2 worker] 参考文本: not provided; reference-only cloning mode
[VoxCPM2 worker] 控制指令: provided
[VoxCPM2 worker] 非语言标签: none
[VoxCPM2 worker] 文本长度: 29 字, chunks=1
[VoxCPM2 worker] 最终模型文本 chunk 1/1 clone_mode=controllable: (语速急促，呼吸发紧，带着匆忙逃命的紧张)我跑了。穿过房子时，门像枪声一样砰砰地关合。前门打不开了。
[VoxCPM2 worker] 完成: sample_rate=48000, elapsed=16.96s, output=/home/muyi086/github/TTS-and-VoiceDesign/api/.cache/runtime/voxcpm2_worker/voxcpm2_out_d17tts5o.wav
voxcpm_model_path: /home/muyi086/hf-mirror/openbmb/VoxCPM2, zipenhancer_model_path: None, enable_denoiser: False
/home/muyi086/miniconda3/envs/voxcpm2/lib/python3.10/site-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
Loading AudioVAE from pytorch: /home/muyi086/hf-mirror/openbmb/VoxCPM2/audiovae.pth
Running on device: cuda, dtype: bfloat16
Loading model from safetensors: /home/muyi086/hf-mirror/openbmb/VoxCPM2/model.safetensors
Loaded VoxCPM2Model
[VoxCPM2] worker 退出码=0，耗时 20.22s
[VoxCPM2] 已保存生成音频: /home/muyi086/github/TTS-and-VoiceDesign/api/tempAudio/voxcpm2_20260808_234019_nxo_3bry.wav
[CUDA] 等待 2.0s 释放显存: after 8306 worker
[GPU 锁] 已退出: voxcpm2/synthesize
INFO:     127.0.0.1:52836 - "POST /v2/synthesize HTTP/1.1" 200 OK

问题出在哪里，帮我深入分析找出问题，然后修复它