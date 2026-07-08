你先阅读`start.sh`，理解当前项目tts对外暴露接口的方式
然后阅读`~/github/zh-public-domain-books-collection`中`demo/script/step_3_tts_local_moss_tts_local_transformer.py`克隆音频的方式,我期望也集成到`start.sh`中，longcat_audiodit暴露的端口为8302。这里要求tts和音色设计模型是真实要使用才载入，然后使用完从显存中移除。要保留start.sh原有已实现的功能，当前改动应该在原有基础上继续新增功能。
以便于我在webui绑定配置，然后可以在`模型配置`-`TTS 语音合成配置`中添加新的longcat_audiodit对应的8302服务地址