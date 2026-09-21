我的设备情况：win11+wsl、4070tiSuper、48g内存、13600kf。我测试大模型一般在wsl的ubuntu体验
你先访问
https://www.modelscope.cn/models?page=1&tabKey=task&tasks=text-to-speech&type=audio
https://www.modelscope.cn/models?page=1&tabKey=task&tasks=auto-speech-recognition&type=audio
https://www.modelscope.cn/models?page=1&tabKey=task&tasks=text-to-audio-synthesis&type=multi-modal
https://www.modelscope.cn/models?page=1&tabKey=task&tasks=audio-generation&type=audio
这是搜索所有的语音的模型。
1. 你从中根据所有的分页信息，要记录总平台总共有多少条结果。找出平台中所有满足下载量超过200的模型，和它对应的modelscope的仓库地址，最近更新时间，整理到表格a中。写入到`流行语音合成模型功能简介.md`标题"完整流行模型"下
2. 然后你遍历所有表格a的模型列表，然后依次进入仓库主页的模型介绍一栏，查看"译文"所属区域的内容，将它的简介或者顶部模型详情记录，补全到表格内。比如"ACE-Step1.5"模型的介绍就是“ACE-Step v1.5 是一款高效开源的音乐基础模型，旨在将商用级音乐生成能力带入消费级硬件。”
3. 全部完成后，需要对表格内所有模型去重，同一个模型名的不同发布以官方发布的模型为准，其余第三方的舍弃。模型名称后不同版本视为不同的模型，需要保留。