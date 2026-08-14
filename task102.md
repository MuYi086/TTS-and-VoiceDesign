还是报错，这是服务端错误：
INFO:     127.0.0.1:41552 - "POST /v1/upload_audio HTTP/1.1" 200 OK
INFO:     127.0.0.1:41552 - "OPTIONS /v1/step-audio-editx/edit HTTP/1.1" 200 OK
[GPU 锁] 等待进入: step-audio-editx/edit
[GPU 锁] 已进入: step-audio-editx/edit
[Step-Audio-EditX] 启动 worker: python=/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/bin/python3
If you want to use the speaker diarization, please `pip install hdbscan`
If you want use mossformer, please install rotary_embedding_torch by: 
 pip install -U rotary_embedding_torch
If you want use mossformer, please install rotary_embedding_torch by: 
 pip install -U rotary_embedding_torch
Please Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
If you want use mossformer, please install rotary_embedding_torch by: 
 pip install -U rotary_embedding_torch
If you want use mossformer, please install rotary_embedding_torch by: 
 pip install -U rotary_embedding_torch
2026-08-14 22:29:10,399 [INFO] new registry table has been added: preprocessor_classes
error: Step-Audio-EditX 运行时不可导入：sox。请确认 uv 环境已安装官方依赖和定制 vLLM。
Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 75, in load_upstream
    from tokenizer import StepAudioTokenizer
  File "/home/muyi086/tts-depency/Step-Audio-EditX/tokenizer.py", line 14, in <module>
    from utils import resample_audio, energy_norm_fn, trim_silence
  File "/home/muyi086/tts-depency/Step-Audio-EditX/utils.py", line 12, in <module>
    import sox
ModuleNotFoundError: No module named 'sox'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 159, in main
    synthesize(load_request(args.input_json), Path(args.output_wav).expanduser().resolve())
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 91, in synthesize
    StepAudioTokenizer, StepAudioTTS, torch, torchaudio = load_upstream(
                                                          ^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 78, in load_upstream
    raise RuntimeError(
RuntimeError: Step-Audio-EditX 运行时不可导入：sox。请确认 uv 环境已安装官方依赖和定制 vLLM。
[Step-Audio-EditX] worker 退出码=1，耗时 8.31s
Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/main.py", line 384, in step_audio_editx_edit
    audio_bytes = manager.run_worker(payload)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/main.py", line 271, in run_worker
    raise RuntimeError(worker_error_excerpt(stderr or stdout))
RuntimeError: File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 159, in main | synthesize(load_request(args.input_json), Path(args.output_wav).expanduser().resolve()) | File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 91, in synthesize | StepAudioTokenizer, StepAudioTTS, torch, torchaudio = load_upstream( | ^^^^^^^^^^^^^^ | File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 78, in load_upstream | raise RuntimeError( | RuntimeError: Step-Audio-EditX 运行时不可导入：sox。请确认 uv 环境已安装官方依赖和定制 vLLM。
[CUDA] 等待 2.0s 释放显存: after Step-Audio-EditX worker
[GPU 锁] 已退出: step-audio-editx/edit
INFO:     127.0.0.1:44754 - "POST /v1/step-audio-editx/edit HTTP/1.1" 500 Internal Server Error
Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/api/step_audio_editx.py", line 170, in run_uv_service
    with urllib.request.urlopen(
         ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 215, in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 521, in open
    response = meth(req, response)
               ^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 630, in http_response
    response = self.parent.error(
               ^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 559, in error
    return self._call_chain(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 492, in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 639, in http_error_default
    raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 500: Internal Server Error

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/api/api.py", line 575, in step_audio_editx_edit
    audio_bytes = step_audio_editx_manager.run_uv_service(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/api/step_audio_editx.py", line 186, in run_uv_service
    raise error from exc
RuntimeError: Step-Audio-EditX uv 服务返回 HTTP 500: File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 159, in main | synthesize(load_request(args.input_json), Path(args.output_wav).expanduser().resolve()) | File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 91, in synthesize | StepAudioTokenizer, StepAudioTTS, torch, torchaudio = load_upstream( | ^^^^^^^^^^^^^^ | File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 78, in load_upstream | raise RuntimeError( | RuntimeError: Step-Audio-EditX 运行时不可导入：sox。请确认 uv 环境已安装官方依赖和定制 vLLM。
INFO:     127.0.0.1:41552 - "POST /v1/step-audio-editx/edit HTTP/1.1" 500 Internal Server Error

这是前端报错:
Step-Audio-EditX 编辑失败: {"detail":"Step-Audio-EditX uv 服务返回 HTTP 500: File \"/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py\", line 159, in main | synthesize(load_request(args.input_json), Path(args.output_wav).expanduser().resolve()) | File \"/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py\", line 91, in synthesize | StepAudioTokenizer, StepAudioTTS, torch, torchaudio = load_upstream( | ^^^^^^^^^^^^^^ | File \"/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py\", line 78, in load_upstream | raise RuntimeError( | RuntimeError: Step-Audio-EditX 运行时不可导入：sox。请确认 uv 环境已安装官方依赖和定制 vLLM。"}

帮我分析并修复，如果时缺少依赖，添加进对应项目的toml中，不要安装，我会手动安装。如果时少了github项目仓库，你不要下载，将地址在终端输出，我会手动下载