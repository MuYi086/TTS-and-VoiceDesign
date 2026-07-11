你先阅读`https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0`文档
然后使用conda创建moss-soundEffect环境,安装对应软件。
然后再项目soundEffect创建对应的测试脚本,将入参
audio = pipe(
    prompt="A dog barking loudly in a park.",
    seconds=10,
    num_inference_steps=100,
    cfg_scale=4.0,
) 
变量命名再顶部维护，方便我调试时可以直接更改和测试
然后再soundEffect输出一份文档`声效提示词说明.md`，这份文档用来告诉大模型，比如当我输出一段小说文本给大模型时，大模型应该如何输出一个moss-soundEffect支持的音效文本给我，以便于我通过脚本传递给moss-soundEffect-v2.0模型时，它能输出符合期望的音频