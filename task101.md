服务端报错：
[配置] local_files_only=True, request_timeout=600.0
INFO:     Started server process [891422]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8306 (Press CTRL+C to quit)
INFO:     127.0.0.1:57532 - "OPTIONS /v1/moss/design HTTP/1.1" 200 OK
[GPU 锁] 等待进入: moss/design
[GPU 锁] 已进入: moss/design
[MOSS VoiceGenerator] 启动 worker: python=/home/muyi086/github/TTS-and-VoiceDesign/moss_voiceGenerator/.venv/bin/python3

Loading weights:   0%|          | 0/1600 [00:00<?, ?it/s]
Loading weights:   2%|▏         | 27/1600 [00:00<00:06, 249.14it/s]
Loading weights:   4%|▍         | 67/1600 [00:00<00:04, 324.56it/s]
Loading weights:   6%|▋         | 100/1600 [00:00<00:05, 266.47it/s]
Loading weights:   8%|▊         | 128/1600 [00:00<00:06, 217.88it/s]
Loading weights:  10%|▉         | 155/1600 [00:00<00:06, 224.97it/s]
Loading weights:  11%|█         | 179/1600 [00:00<00:06, 224.62it/s]
Loading weights:  13%|█▎        | 203/1600 [00:00<00:07, 198.85it/s]
Loading weights:  14%|█▍        | 224/1600 [00:01<00:07, 193.07it/s]
Loading weights:  16%|█▌        | 250/1600 [00:01<00:06, 209.93it/s]
Loading weights:  18%|█▊        | 281/1600 [00:01<00:05, 236.39it/s]
Loading weights:  20%|██        | 324/1600 [00:01<00:04, 287.31it/s]
Loading weights:  23%|██▎       | 366/1600 [00:01<00:03, 323.29it/s]
Loading weights:  25%|██▌       | 403/1600 [00:01<00:03, 334.62it/s]
Loading weights:  28%|██▊       | 442/1600 [00:01<00:03, 347.65it/s]
Loading weights:  30%|███       | 488/1600 [00:01<00:02, 373.92it/s]
Loading weights:  33%|███▎      | 528/1600 [00:01<00:02, 381.00it/s]
Loading weights:  36%|███▌      | 574/1600 [00:01<00:02, 403.92it/s]
Loading weights:  39%|███▉      | 626/1600 [00:02<00:02, 437.11it/s]
Loading weights:  42%|████▏     | 670/1600 [00:02<00:02, 433.13it/s]
Loading weights:  45%|████▍     | 714/1600 [00:02<00:02, 428.34it/s]
Loading weights:  47%|████▋     | 757/1600 [00:02<00:01, 423.41it/s]
Loading weights:  50%|█████     | 808/1600 [00:02<00:01, 447.26it/s]
Loading weights:  53%|█████▎    | 853/1600 [00:02<00:01, 431.92it/s]
Loading weights:  56%|█████▌    | 897/1600 [00:02<00:01, 427.07it/s]
Loading weights:  59%|█████▉    | 940/1600 [00:02<00:01, 349.75it/s]
Loading weights:  61%|██████    | 978/1600 [00:03<00:02, 257.91it/s]
Loading weights:  63%|██████▎   | 1009/1600 [00:03<00:02, 230.89it/s]
Loading weights:  65%|██████▍   | 1036/1600 [00:03<00:02, 226.81it/s]
Loading weights:  66%|██████▋   | 1062/1600 [00:03<00:02, 227.70it/s]
Loading weights:  68%|██████▊   | 1087/1600 [00:03<00:02, 232.64it/s]
Loading weights:  70%|██████▉   | 1112/1600 [00:03<00:02, 229.97it/s]
Loading weights:  71%|███████   | 1136/1600 [00:03<00:02, 203.83it/s]
Loading weights:  72%|███████▏  | 1158/1600 [00:04<00:02, 198.51it/s]
Loading weights:  74%|███████▎  | 1179/1600 [00:04<00:02, 189.30it/s]
Loading weights:  75%|███████▍  | 1199/1600 [00:04<00:02, 185.64it/s]
Loading weights: 100%|██████████| 1600/1600 [00:04<00:00, 367.38it/s]
error: Unexpected keyword argument local_files_only.
Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/moss_voiceGenerator/worker.py", line 202, in main
    synthesize(load_request(args.input_json), Path(args.output_wav).expanduser().resolve())
  File "/home/muyi086/github/TTS-and-VoiceDesign/moss_voiceGenerator/worker.py", line 150, in synthesize
    processor = AutoProcessor.from_pretrained(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/moss_voiceGenerator/.venv/lib/python3.12/site-packages/transformers/models/auto/processing_auto.py", line 323, in from_pretrained
    return processor_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/api/.cache/runtime/hf_modules/transformers_modules/MOSS_hyphen_VoiceGenerator/a3b9597cc6ab4bbe/processing_moss_tts.py", line 298, in from_pretrained
    return cls(
           ^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/api/.cache/runtime/hf_modules/transformers_modules/MOSS_hyphen_VoiceGenerator/a3b9597cc6ab4bbe/processing_moss_tts.py", line 234, in __init__
    super().__init__(tokenizer=tokenizer, audio_tokenizer=audio_tokenizer, **kwargs)
  File "/home/muyi086/github/TTS-and-VoiceDesign/moss_voiceGenerator/.venv/lib/python3.12/site-packages/transformers/processing_utils.py", line 625, in __init__
    raise TypeError(f"Unexpected keyword argument {key}.")
TypeError: Unexpected keyword argument local_files_only.
[MOSS VoiceGenerator] worker 退出码=1，耗时 16.10s
Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/moss_voiceGenerator/main.py", line 365, in moss_design
    content=manager.run_worker(manager.build_worker_payload(request)),
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/moss_voiceGenerator/main.py", line 282, in run_worker
    raise RuntimeError(worker_error_excerpt(stderr or stdout))
RuntimeError: File "/home/muyi086/github/TTS-and-VoiceDesign/api/.cache/runtime/hf_modules/transformers_modules/MOSS_hyphen_VoiceGenerator/a3b9597cc6ab4bbe/processing_moss_tts.py", line 298, in from_pretrained | return cls( | ^^^^ | File "/home/muyi086/github/TTS-and-VoiceDesign/api/.cache/runtime/hf_modules/transformers_modules/MOSS_hyphen_VoiceGenerator/a3b9597cc6ab4bbe/processing_moss_tts.py", line 234, in __init__ | super().__init__(tokenizer=tokenizer, audio_tokenizer=audio_tokenizer, **kwargs) | File "/home/muyi086/github/TTS-and-VoiceDesign/moss_voiceGenerator/.venv/lib/python3.12/site-packages/transformers/processing_utils.py", line 625, in __init__ | raise TypeError(f"Unexpected keyword argument {key}.") | TypeError: Unexpected keyword argument local_files_only.
[CUDA] 等待 2.0s 释放显存: after MOSS VoiceGenerator worker
[GPU 锁] 已退出: moss/design
INFO:     127.0.0.1:57532 - "POST /v1/moss/design HTTP/1.1" 500 Internal Server Error

浏览器curl： 
curl --url 'http://127.0.0.1:8315/v1/moss/design' \
  -H 'Accept: */*' \
  -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8' \
  -H 'Cache-Control: no-cache' \
  -H 'Connection: keep-alive' \
  -H 'Content-Type: application/json' \
  -H 'Origin: http://localhost:5502' \
  -H 'Pragma: no-cache' \
  -H 'Referer: http://localhost:5502/' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Site: cross-site' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Windows"' \
  --data-raw '{"voice_description":"青年女性旁白，音域居中偏低，声线清润醇和，厚度与明亮度均衡，无气声沙哑。咬字清晰圆柔，语速中等偏慢，节奏平稳，句间停顿略长，默认语气沉稳冷静，叙述耐听且不受情节情绪影响。","text":"黄昏的光线轻轻落在木桌一角，空荡的房间里只听见远处钟声回荡。"}'

浏览器弹出错误:
{
    "detail": "File \"/home/muyi086/github/TTS-and-VoiceDesign/api/.cache/runtime/hf_modules/transformers_modules/MOSS_hyphen_VoiceGenerator/a3b9597cc6ab4bbe/processing_moss_tts.py\", line 298, in from_pretrained | return cls( | ^^^^ | File \"/home/muyi086/github/TTS-and-VoiceDesign/api/.cache/runtime/hf_modules/transformers_modules/MOSS_hyphen_VoiceGenerator/a3b9597cc6ab4bbe/processing_moss_tts.py\", line 234, in __init__ | super().__init__(tokenizer=tokenizer, audio_tokenizer=audio_tokenizer, **kwargs) | File \"/home/muyi086/github/TTS-and-VoiceDesign/moss_voiceGenerator/.venv/lib/python3.12/site-packages/transformers/processing_utils.py\", line 625, in __init__ | raise TypeError(f\"Unexpected keyword argument {key}.\") | TypeError: Unexpected keyword argument local_files_only."
}