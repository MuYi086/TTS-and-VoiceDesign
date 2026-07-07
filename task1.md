你先阅读`start.sh`，理解当前项目tts对外暴露接口的方式
移除原有的moss脚本和相关逻辑
然后阅读`~/github/zh-public-domain-books-collection`中`音色设计/script/tts_voice_design_mimo.py`合成音色的方式,我期望也集成到`start.sh`中，这里mimo是适应的云api和本地qwen-tts-design不一致，需要注意实现方式。同时tts和音色设计模型是真实要使用才载入，然后使用完从显存中移除。
以便于我在webui绑定配置，然后下拉刷选为qwen还是mimo去设计音色