你先阅读`api`下等模型的接入，然后阅读`~/github/scoring-for-TTS`下`modelScript/tts_local_dots_tts_soar.py`的实现和使用方式，我期望在`~/github/TTS-and-VoiceDesign`增加一个stable_audio_3_small_sfx的服务，以端口8312调用，要求能和moss_soundEffect一样支持生成音效，完成后要自动清除显存。具体的使用限制可以参考官方文档.
然后再`~/github/TTS-Studio-WebUI`项目的index.html中的
<button @click="analyzeScript"
                                :class="['w-full px-3 py-2 text-white rounded-lg text-sm font-bold transition-all flex items-center justify-center', isAnalyzingScript ? 'bg-red-500 hover:bg-red-600' : 'bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50']">
                                <span v-if="isAnalyzingScript" class="animate-spin mr-2">⏳</span>
                                {{ isAnalyzingScript ? '停止分析' : 'AI 深度分析' }}
                            </button>
                            下方增加一个下拉筛选框，支持moss-soundEffect-v2和stable-audio-3-small-sfx选择，选择了谁，下面的"生成全部 SoundEffect 音效"就使用已选择的音效模型去生成音效。模型生成后和voxcpm2一样会同步保存到`api/tempAudio`