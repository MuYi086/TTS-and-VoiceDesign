[MiMo] model=mimo-v2.5-tts-voicedesign, base_url=https://api.xiaomimimo.com/v1, chunks=1
[MiMo] 合成 chunk 1/1: 30 字
[GPU 锁] 已退出: mimo/design
INFO:     127.0.0.1:51652 - "POST /v1/mimo/design HTTP/1.1" 500 Internal Server Error
INFO:     127.0.0.1:55834 - "OPTIONS /v1/mimo/design HTTP/1.1" 200 OK
[GPU 锁] 等待进入: mimo/design
[GPU 锁] 已进入: mimo/design
[MiMo] model=mimo-v2.5-tts-voicedesign, base_url=https://api.xiaomimimo.com/v1, chunks=1
[MiMo] 合成 chunk 1/1: 30 字
[GPU 锁] 已退出: mimo/design
INFO:     127.0.0.1:55834 - "POST /v1/mimo/design HTTP/1.1" 500 Internal Server Error
INFO:     127.0.0.1:52274 - "OPTIONS /v1/mimo/design HTTP/1.1" 200 OK
[GPU 锁] 等待进入: mimo/design
[GPU 锁] 已进入: mimo/design
[MiMo] model=mimo-v2.5-tts-voicedesign, base_url=https://api.xiaomimimo.com/v1, chunks=1
[MiMo] 合成 chunk 1/1: 30 字
[GPU 锁] 已退出: mimo/design
INFO:     127.0.0.1:52274 - "POST /v1/mimo/design HTTP/1.1" 500 Internal Server Error

出错了，这是前端的network请求:
curl 'http://127.0.0.1:8300/v1/mimo/design' \
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
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36' \
  -H 'sec-ch-ua: "Not;A=Brand";v="8", "Chromium";v="150", "Google Chrome";v="150"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Windows"' \
  --data-raw '{"voice_description":"女性，中年，中低音区，声线醇厚沉稳，略带沙哑的温暖质感，共鸣适中，咬字清晰从容，语速偏慢，节奏平稳少有急促停顿，整体气质冷静睿智，带有学者式的平和与见多识广的从容。","text":"那些古老的文字与知识需要敬畏与智慧，更需要谨慎地一代代传承。"}'

  前端项目在`~/github/TTS-Studio-WebUI`

  帮我分析原因并修复