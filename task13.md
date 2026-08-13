你先阅读`项目升级评估.md`
我打算将`~/github/TTS-and-VoiceDesign`和`~/github/scoring-for-TTS`
改造成现代化的python项目，并且python环境版本统一为3.12.13,tts模型运行所需的conda环境要做对应处理。要求保证升级后原有功能完整可用，具体交互参考`~/github/TTS-Studio-WebUI`前端的处理

我准备第一步先将qwen3-tts的conda环境完全复刻到uv创建的qwen3_tts环境。

你帮我评估是否可以按模型建立目录，然后使用uv init qwen3_tts，然后逐个将tts模型对应的conda环境安装的软件依赖再qwen3_tts目录内安装，然后增加对应的main.py实现原来`api`目录内该模型实现的功能，最后再start.sh暴露替换原来的qwen3_tts逻辑

如果以上可行：
你帮我找出`~/github/TTS-and-VoiceDesign`中`api`下qwen3_tts_api正常运行所需要的conda环境中所有依赖软件以及可能需要的`/home/muyi086/tts-depency`内的仓库,将内容输出到`qwen3_tts迁移计划.md`,其中python包安装命令帮我使用清华，中科大或者阿里的镜像源，或者直接输出可以供uv快速安装的配置文件，这样我手动安装会比较快.tts模型依然引用`hf-mirror`内对应的模型权重