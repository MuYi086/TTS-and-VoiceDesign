浏览器报错了：Step-Audio-EditX 编辑失败: {"detail":"Step-Audio-EditX uv 服务不可达：http://127.0.0.1:8316 ([Errno 111] Connection refused)"}
服务端错误提示：
[Qwen3-TTS] 已保存生成音频: /home/muyi086/github/TTS-and-VoiceDesign/api/tempAudio/qwen3_tts_20260814_221853_i8hiwt47.wav
[CUDA] 等待 2.0s 释放显存: after qwen3_tts worker
[GPU 锁] 已退出: qwen3_tts/synthesize
INFO:     127.0.0.1:53218 - "POST /v2/synthesize HTTP/1.1" 200 OK
INFO:     127.0.0.1:34578 - "POST /v1/upload_audio HTTP/1.1" 200 OK
INFO:     127.0.0.1:34578 - "OPTIONS /v1/step-audio-editx/edit HTTP/1.1" 200 OK
Traceback (most recent call last):
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 1344, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/http/client.py", line 1358, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/http/client.py", line 1404, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/http/client.py", line 1353, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/http/client.py", line 1113, in _send_output
    self.send(msg)
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/http/client.py", line 1057, in send
    self.connect()
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/http/client.py", line 1023, in connect
    self.sock = self._create_connection(
                ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/socket.py", line 865, in create_connection
    raise exceptions[0]
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/socket.py", line 850, in create_connection
    sock.connect(sa)
ConnectionRefusedError: [Errno 111] Connection refused

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/api/step_audio_editx.py", line 170, in run_uv_service
    with urllib.request.urlopen(
         ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 215, in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 515, in open
    response = self._open(req, data)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 532, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 492, in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 1373, in http_open
    return self.do_open(http.client.HTTPConnection, req)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/miniconda3/envs/moss-soundEffect/lib/python3.12/urllib/request.py", line 1347, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 111] Connection refused>

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/api/api.py", line 575, in step_audio_editx_edit
    audio_bytes = step_audio_editx_manager.run_uv_service(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/muyi086/github/TTS-and-VoiceDesign/api/step_audio_editx.py", line 193, in run_uv_service
    raise error from exc
RuntimeError: Step-Audio-EditX uv 服务不可达：http://127.0.0.1:8316 ([Errno 111] Connection refused)
INFO:     127.0.0.1:34578 - "POST /v1/step-audio-editx/edit HTTP/1.1" 503 Service Unavailable

分析问题并修复它