你先阅读`项目升级评估.md`
我打算将`~/github/TTS-and-VoiceDesign`和`~/github/scoring-for-TTS`
改造成现代化的python项目，并且python环境版本统一为3.12.13,tts模型运行所需的conda环境要做对应处理。要求保证升级后原有功能完整可用，具体交互参考`~/github/TTS-Studio-WebUI`前端的处理
原始tts模型名称 a = `Step-Audio-EditX`
即将创建的uv环境 b = `Step_Audio_EditX`
我准备第一步先将 ${a} 的conda环境完全复刻到uv创建的${b}环境。

你帮我评估是否可以按模型建立目录，然后使用uv init ${b}，然后逐个将tts模型对应的conda环境安装的软件依赖再${b}目录内安装，然后增加对应的main.py实现原来`api`目录内该模型实现的功能，最后再start.sh暴露替换原来的${b}逻辑

如果以上可行：
你帮我找出`~/github/TTS-and-VoiceDesign`中`api`下${b}_api正常运行所需要的conda环境中所有依赖软件以及可能需要的`/home/muyi086/tts-depency`内的仓库,将内容输出到`${b}迁移计划.md`,其中python包安装命令帮我使用清华，中科大或者阿里的镜像源，或者直接输出可以供uv快速安装的配置文件，这样我手动安装会比较快。tts模型依然引用`hf-mirror`内对应的模型权重

具体可以参考`qwen3_tts`,这是一个成功案例，已经完成将对应tts在conda的依赖软件完整在改项目内实现并且暴露到start.sh中