服务端报错：026-08-14 22:43:57,078 [INFO] 🎤 Loading CosyVoice with dtype=bfloat16, cuda_graph=False
2026-08-14 22:43:59,631 [INFO] 🎤 CosyVoice model loaded successfully
2026-08-14 22:43:59,632 [ERROR] Edit failed: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.
error: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.
Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/lib/python3.12/site-packages/torchaudio/_torchcodec.py", line 82, in load_with_torchcodec
    from torchcodec.decoders import AudioDecoder
ModuleNotFoundError: No module named 'torchcodec'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 159, in main
    synthesize(load_request(args.input_json), Path(args.output_wav).expanduser().resolve())
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/worker.py", line 132, in synthesize
    output_audio, sample_rate = model.edit(
                                ^^^^^^^^^^^
  File "/home/muyi086/tts-depency/Step-Audio-EditX/tts.py", line 203, in edit
    self.preprocess_prompt_wav(prompt_wav_path)
  File "/home/muyi086/tts-depency/Step-Audio-EditX/tts.py", line 365, in preprocess_prompt_wav
    prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/lib/python3.12/site-packages/torchaudio/__init__.py", line 86, in load
    return load_with_torchcodec(
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/lib/python3.12/site-packages/torchaudio/_torchcodec.py", line 84, in load_with_torchcodec
    raise ImportError(
ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.
[rank0]:[W814 22:43:59.256839016 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[Step-Audio-EditX] worker 退出码=1，耗时 33.78s
Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/main.py", line 384, in step_audio_editx_edit
    audio_bytes = manager.run_worker(payload)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/main.py", line 271, in run_worker
    raise RuntimeError(worker_error_excerpt(stderr or stdout))
RuntimeError: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ | File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/lib/python3.12/site-packages/torchaudio/__init__.py", line 86, in load | return load_with_torchcodec( | ^^^^^^^^^^^^^^^^^^^^^ | File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/lib/python3.12/site-packages/torchaudio/_torchcodec.py", line 84, in load_with_torchcodec | raise ImportError( | ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function. | [rank0]:[W814 22:43:59.256839016 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[CUDA] 等待 2.0s 释放显存: after Step-Audio-EditX worker
[GPU 锁] 已退出: step-audio-editx/edit
INFO:     127.0.0.1:52720 - "POST /v1/step-audio-editx/edit HTTP/1.1" 500 Internal Server Error
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
RuntimeError: Step-Audio-EditX uv 服务返回 HTTP 500: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ | File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/lib/python3.12/site-packages/torchaudio/__init__.py", line 86, in load | return load_with_torchcodec( | ^^^^^^^^^^^^^^^^^^^^^ | File "/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/lib/python3.12/site-packages/torchaudio/_torchcodec.py", line 84, in load_with_torchcodec | raise ImportError( | ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function. | [rank0]:[W814 22:43:59.256839016 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
INFO:     127.0.0.1:35394 - "POST /v1/step-audio-editx/edit HTTP/1.1" 500 Internal Server Error

前端报错：
Step-Audio-EditX 编辑失败: {"detail":"Step-Audio-EditX uv 服务返回 HTTP 500: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ | File \"/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/lib/python3.12/site-packages/torchaudio/__init__.py\", line 86, in load | return load_with_torchcodec( | ^^^^^^^^^^^^^^^^^^^^^ | File \"/home/muyi086/github/TTS-and-VoiceDesign/Step_Audio_EditX/.venv/lib/python3.12/site-packages/torchaudio/_torchcodec.py\", line 84, in load_with_torchcodec | raise ImportError( | ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function. | [rank0]:[W814 22:43:59.256839016 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())"}

帮我分析并修复，如果时缺少依赖，添加进对应项目的toml中，不要安装，我会手动安装。如果时少了github项目仓库，你不要下载，将地址在终端输出，我会手动下载