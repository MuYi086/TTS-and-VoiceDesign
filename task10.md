你先阅读`soundEffectResult`目录
其中`soundEffectResult/text.md`是小说文本,
`soundEffectResult/deepseekRequest.json`是发给deepseek的请求,
`soundEffectResult/deepseekResponse.json`deepseek响应的结果,
`soundEffectResult/resultDrama.json`是上个版本的提示词同样的deepseekRequest返回的结果

可以从结果看出,真正的sfx_plan部分只有8处，其实偏少，我认为是覆盖的音效关键词不够。实际上在文本中

1. 他把一个落满灰尘的箱子拖到吊灯下面，好让我看清上面的刻纹
  拖箱子这个也是可以做成音效的

2. “闻起来像……茉莉花？”他凑近嗅了嗅。 
  其中"嗅了嗅"也是可以做成音效插入的，

3. 我对她的警告一笑置之，但那天下午敲响布罗迪家门时，我的心沉甸甸的
  这里的敲门也是可以做成音效的

还有一些其他场景等等。
你评估我的判断合理性，如果合理，那么结合当前index.html中的v-model="customPromptTemplate"绑定的提示词，帮我进一步优化提示词。期望的效果是大模型能拆解成剧本时更全面，更专业，更准确。这样我后期合成音频和音效也能更合理与高效。
