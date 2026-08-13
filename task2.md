你先阅读`api`下等模型的接入，然后阅读`~/github/scoring-for-TTS`下`modelScript/tts_local_dots_tts_soar.py`的实现和使用方式，我期望在`~/github/TTS-and-VoiceDesign`增加一个stable_audio_3_medium的服务，以端口8313调用，要求能和moss_soundEffect一样支持生成音效，完成后要自动清除显存。具体的使用限制可以参考官方文档.
然后再`~/github/TTS-Studio-WebUI`项目的index.html中的
<select v-model="soundEffectModel"
                                :disabled="isGeneratingSoundEffects || isAnySoundEffectGenerating"
                                class="w-full px-3 py-2 border rounded-lg text-sm font-medium focus:ring-2 focus:ring-amber-500 outline-none bg-white disabled:opacity-50 disabled:cursor-wait"
                                title="选择生成 SoundEffect 音效的模型">
                                <option v-for="model in soundEffectModelOptions" :key="model.id" :value="model.id">
                                    {{ model.label }}
                                </option>
                            </select>下拉筛选框，增加stable-audio-3-medium选择，选择了谁，下面的"生成全部 SoundEffect 音效"就使用已选择的音效模型去生成音效。模型生成后和voxcpm2一样会同步保存到`api/tempAudio`