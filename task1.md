你先阅读`api`下等模型的接入，然后阅读`~/github/scoring-for-TTS`下`modelScript/tts_local_dots_tts_soar.py`的实现和使用方式，我期望在`~/github/TTS-and-VoiceDesign`增加一个dots_tts_soar的服务，以端口8308调用，要求能和voxcpm2一样支持克隆，完成后要自动清除显存。具体的使用限制可以参考官方文档.
然后再index.html中的
<option v-for="conf in activeTtsConfigs" :key="conf.id" :value="conf.id">
                                    {{ ttsModelLabel(conf) }}
                                </option>
要增加对应模型选择，选择后合成音频使用选定的模型执行,模型合成音频后和voxcpm2一样会同步保存到`api/tempAudio`