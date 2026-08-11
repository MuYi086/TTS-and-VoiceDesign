检查`api`目录，我期望
`longcat_audiodit`
`qwen3_tts`
`dots_tts_soar`等模型的脚本中，

能和`api/voxcpm2_api.py`一样，将合成克隆相关的变量提取成全局变量放在页面顶部，这样我要调试查看克隆效果可以直接在顶部改变值调试，就像VOXCPM2_CFG_DEFAULT = 2.0这样，并且要添加中文注释