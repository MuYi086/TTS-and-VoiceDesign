bash start.sh 执行报错了:
(base) muyi086@DESKTOP-KMJK7K0:~/github/TTS-and-VoiceDesign$ bash start.sh 
==================================================
   Unitale AI local backend
==================================================
Main conda env:      unitale-tts-local
Qwen model:          /home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
IndexTTS2 model:     /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2
IndexTTS2 code:      /home/muyi086/github/TTS-and-VoiceDesign/api/vendor/index-tts
SoundEffect env:     moss-soundEffect
SoundEffect model:   /home/muyi086/hf-mirror/OpenMOSS-Team/MOSS-SoundEffect-v2.0
SoundEffect device:  cuda (bfloat16)
Qwen3-TTS worker env: qwen3-tts
Qwen3-TTS model:     /home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-Base
VoxCPM2 worker env:  voxcpm2
VoxCPM2 model:       /home/muyi086/hf-mirror/openbmb/VoxCPM2
VoxCPM2 config:      managed by api/voxcpm2_api.py
Qwen3-TTS trim lead: 1
Qwen3-TTS trim thres:-42 dB
Qwen3-TTS trim min:  120 ms
Qwen sidecar libs:   /home/muyi086/github/TTS-and-VoiceDesign/api/vendor/qwen_libs
MiMo base URL:       https://api.xiaomimimo.com/v1
MiMo model:          mimo-v2.5-tts-voicedesign
MiMo API key:        configured
Prompts dir:         /home/muyi086/github/TTS-and-VoiceDesign/api/prompts
HF modules cache:    /home/muyi086/github/TTS-and-VoiceDesign/api/.cache/runtime/hf_modules
GPU lock file:       /home/muyi086/github/TTS-and-VoiceDesign/api/.cache/runtime/gpu-runtime.lock
IndexTTS2 device:    auto
IndexTTS2 fp16:      1
IndexTTS2 beams:     1
CUDA kernel:         0
IndexTTS2 timeout:    600 s
IndexTTS2 CUDA retry: 1
IndexTTS2 segment:    80 tokens
IndexTTS2 max mel:    1200 tokens
Main API:            http://0.0.0.0:8300
Main health:         http://127.0.0.1:8300/v1/health
SoundEffect API:     http://0.0.0.0:8311
SoundEffect health:  http://127.0.0.1:8311/v1/health
Qwen3-TTS API:       http://0.0.0.0:8305
Qwen3-TTS health:    http://127.0.0.1:8305/v1/health
VoxCPM2 API:         http://0.0.0.0:8306
VoxCPM2 health:      http://127.0.0.1:8306/v1/health
Qwen design route:   http://127.0.0.1:8300/v1/qwen/design
VoxCPM2 design route: http://127.0.0.1:8300/v1/voxcpm2/design
MiMo design route:   http://127.0.0.1:8300/v1/mimo/design
SoundEffect route:   http://127.0.0.1:8311/v1/generate
Qwen3-TTS synth:     http://127.0.0.1:8305/v2/synthesize
VoxCPM2 synth:       http://127.0.0.1:8306/v2/synthesize
==================================================

EnvironmentLocationNotFound: Not a conda environment: /home/muyi086/miniconda3/envs/unitale-tts-local

EnvironmentLocationNotFound: Not a conda environment: /home/muyi086/miniconda3/envs/unitale-tts-local



EnvironmentLocationNotFound: Not a conda environment: /home/muyi086/miniconda3/envs/unitale-tts-local

现在音色设计的conda环境有
qwen3-voiceDesign        /home/muyi086/miniconda3/envs/qwen3-voiceDesign
voxcpm2                  /home/muyi086/miniconda3/envs/voxcpm2
moss-voiceGenerator      /home/muyi086/miniconda3/envs/moss-voiceGenerator
Ming-omni-tts-0.5B       /home/muyi086/miniconda3/envs/Ming-omni-tts-0.5B
还有已经支持的云端mimo音色设计接口
需要对应补齐
http://127.0.0.1:8300/v1/qwen/design 使用qwen3-voiceDesign环境
http://127.0.0.1:8300/v1/moss/design 使用moss-voiceGenerator环境
http://127.0.0.1:8300/v1/Ming/design 使用Ming-omni-tts-0.5B环境
音频合成的conda环境有
voxcpm2                  /home/muyi086/miniconda3/envs/voxcpm2
qwen3-tts                /home/muyi086/miniconda3/envs/qwen3-tts
Ming-omni-tts-0.5B       /home/muyi086/miniconda3/envs/Ming-omni-tts-0.5B
需要对应补齐
http://127.0.0.1:8305/v2/synthesize 使用qwen3-tts环境
http://127.0.0.1:8306/v2/synthesize 使用voxcpm2环境
http://127.0.0.1:8306/v2/synthesize 使用Ming-omni-tts-0.5B环境

以上任务需要检查并优化`~/github/TTS-Studio-WebUI`的index.html中<!-- 左侧：角色与音色绑定栏 -->下音色模型列表选择和相关逻辑；<!-- 配音/生成/播放 -->下“配音与播放”模型选择和相关逻辑
以及`~/github/TTS-and-VoiceDesign`下api目录内已有api的使用和新增api的调试和处理。具体tts模型的使用可以参考`~/github/scoring-for-TTS`下modelScript内的安装指南和调试脚本