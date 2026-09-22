
你先阅读`moss_audio_4b_thinking`这种python项目结构和实现，以及通过start.sh调用的方式，
然后阅读`/home/muyi086/hf-mirror/netease-youdao/Confucius4-TTS`目录,
将moss_audio_4b_thinking的调用在当前项目`Confucius4_TTS`中实现，并暴露在`start.sh`中以端口8361对外暴露服务，路由地址设置为`/v1/confucius4TTS/generate`