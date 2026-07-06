~/Downloads/Unitale-1.5 项目中访问index.html,配置好llm和tts模型，
再合成音频时，提示：语音合成失败: {"detail":"CUDA driver error: device not ready"}
然后当前项目控制台报错：
(base) muyi086@DESKTOP-KMJK7K0:~/github/TTS-and-VoiceDesign$ bash ./start.sh 
==================================================
   Unitale AI local backend
==================================================
Conda env:           unitale-tts-local
Qwen model:          /home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
IndexTTS2 model:     /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2
IndexTTS2 code:      /home/muyi086/github/TTS-and-VoiceDesign/vendor/index-tts
Qwen sidecar libs:   /home/muyi086/github/TTS-and-VoiceDesign/vendor/qwen_libs
Prompts dir:         /home/muyi086/github/TTS-and-VoiceDesign/prompts
Listen:              http://0.0.0.0:8300
Health:              http://127.0.0.1:8300/v1/health
==================================================
==================================================
   Unitale AI 本地后端服务 IndexTTS2 + Qwen3-TTS VoiceDesign
==================================================
[配置] Qwen 模型目录: /home/muyi086/hf-mirror/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
[配置] IndexTTS2 模型目录: /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2
[配置] IndexTTS2 配置: /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/config.yaml
[配置] prompts 目录: /home/muyi086/github/TTS-and-VoiceDesign/prompts
[配置] local_files_only=True, preload_indextts=False
INFO:     Started server process [722699]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8300 (Press CTRL+C to quit)
INFO:     127.0.0.1:36266 - "GET /v1/check/audio?file_name=2%E6%9C%889%E6%97%A5%20(1)_%5Bcut_23sec%5D.mp3 HTTP/1.1" 200 OK
INFO:     127.0.0.1:36266 - "OPTIONS /v2/synthesize HTTP/1.1" 200 OK
[IndexTTS2] 正在载入本地模型...
>> GPT weights restored from: /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/gpt.pth
GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
>> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.
TypeError("unsupported operand type(s) for +: 'NoneType' and 'str'")
>> semantic_codec weights restored from: /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/hf_cache/semantic_codec_model.safetensors
cfm loaded
length_regulator loaded
gpt_layer loaded
>> s2mel weights restored from: /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/s2mel.pth
>> campplus_model weights restored from: /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/hf_cache/campplus_cn_common.bin
Loading config.json from local directory
Loading weights from local directory
Removing weight norm...
>> bigvgan weights restored from: /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/hf_cache/bigvgan
2026-07-06 22:16:14,493 WETEXT INFO found existing fst: /home/muyi086/github/TTS-and-VoiceDesign/vendor/index-tts/indextts/utils/tagger_cache/zh_tn_tagger.fst
2026-07-06 22:16:14,493 WETEXT INFO                     /home/muyi086/github/TTS-and-VoiceDesign/vendor/index-tts/indextts/utils/tagger_cache/zh_tn_verbalizer.fst
2026-07-06 22:16:14,493 WETEXT INFO skip building fst for zh_normalizer ...
2026-07-06 22:16:14,609 WETEXT INFO found existing fst: /home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/tn/en_tn_tagger.fst
2026-07-06 22:16:14,609 WETEXT INFO                     /home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/tn/en_tn_verbalizer.fst
2026-07-06 22:16:14,609 WETEXT INFO skip building fst for en_normalizer ...
>> TextNormalizer loaded
>> bpe model loaded from: /home/muyi086/hf-mirror/IndexTeam/IndexTTS-2/bpe.model
✅ IndexTTS2 就绪。
>> starting inference...
scaled emotion vectors to 0.6x: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
Use the specified emotion vector
Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. You should pass an instance of `Cache` instead, e.g. `past_key_values=DynamicCache.from_legacy_cache(past_key_values)`.
Traceback (most recent call last):
  File "/home/muyi086/github/TTS-and-VoiceDesign/api.py", line 496, in synthesize_v2
    manager.indextts.infer(
  File "/home/muyi086/github/TTS-and-VoiceDesign/vendor/index-tts/indextts/infer_v2.py", line 380, in infer
    return list(self.infer_generator(
  File "/home/muyi086/github/TTS-and-VoiceDesign/vendor/index-tts/indextts/infer_v2.py", line 572, in infer_generator
    codes, speech_conditioning_latent = self.gpt.inference_speech(
  File "/home/muyi086/github/TTS-and-VoiceDesign/vendor/index-tts/indextts/gpt/model_v2.py", line 773, in inference_speech
    output = self.inference_model.generate(inputs,
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/home/muyi086/github/TTS-and-VoiceDesign/vendor/index-tts/indextts/gpt/transformers_generation_utils.py", line 2247, in generate
    result = self._beam_search(
  File "/home/muyi086/github/TTS-and-VoiceDesign/vendor/index-tts/indextts/gpt/transformers_generation_utils.py", line 3458, in _beam_search
    outputs = self(**model_inputs, return_dict=True)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/muyi086/github/TTS-and-VoiceDesign/vendor/index-tts/indextts/gpt/model_v2.py", line 161, in forward
    transformer_outputs = self.transformer(
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py", line 939, in forward
    outputs = block(
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py", line 403, in forward
    attn_output, self_attn_weights = self.attn(
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py", line 309, in forward
    key_states, value_states = past_key_value.update(
  File "/home/muyi086/miniconda3/envs/unitale-tts-local/lib/python3.10/site-packages/transformers/cache_utils.py", line 545, in update
    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
RuntimeError: CUDA driver error: device not ready
INFO:     127.0.0.1:36266 - "POST /v2/synthesize HTTP/1.1" 500 Internal Server Error

帮我分析并修复问题