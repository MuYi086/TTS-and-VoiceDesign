# TTS 项目量化与渐进式优化落地规划

> 文档版本：v1.0
> 更新时间：2026-08-23
> 适用项目：[`TTS-and-VoiceDesign`](https://github.com/MuYi086/TTS-and-VoiceDesign)、[`TTS-Studio-WebUI`](https://github.com/MuYi086/TTS-Studio-WebUI)
> 目标：把分享对话中关于模型量化、音频质量评估和训练学习路径的结论，渐进式落实到现有 TTS 工作台，不破坏当前可用功能。

## 1. 结论先行

当前最适合你的路线不是马上把 VoxCPM2、MOSS、LongCat 等模型全部压成 INT4，而是建立一套“原精度基线 → 选择性量化 → 音频 A/B 评估 → 可回退部署”的工程闭环。

推荐顺序如下：

1. 保留现有 BF16/FP16 版本作为生产基线，先补齐显存、耗时、模型变体和音频质量记录。
2. 使用一个标准 Transformer 模型练习 bitsandbytes 4/8 bit，验证量化目录、加载和回归流程。
3. 使用 TorchAO 或 Quanto 对一个 DiT/Transformer 主干做选择性 INT8 实验，优先保留 Codec、Vocoder、Speaker Encoder 和 LayerNorm 等敏感模块的原精度。
4. 将量化版本作为独立模型变体接入后端和 WebUI，默认仍使用 BF16/FP16，实验失败可以一键回退。
5. 在有稳定 A/B 数据后，再评估 INT4、GPTQ/AWQ 或 LoRA；不要把 GPTQ/AWQ、GGUF 强行用于没有官方转换和运行后端的自定义 TTS。

分享对话中的时间估计可以作为学习节奏参考：已有 TTS 部署基础时，完成一个已有 TTS 的量化并运行约需 2–4 周；能够处理多个不同架构并保持音质基本可接受，约需 2–4 个月。这里的时间按每周 8–12 小时学习和实践估计，不等同于纯开发工时。

## 2. 输入依据与当前状态

### 2.1 分享对话形成的硬结论

| 主题 | 采用结论 |
| --- | --- |
| LLM/标准 Transformer | 优先学习 bitsandbytes 4/8 bit，再考虑 GPTQModel/AWQ；GGUF 只用于有官方支持的 LLM |
| Diffusion/DiT/音效模型 | 优先考虑 bitsandbytes、TorchAO、Quanto；不优先使用 GPTQ、AWQ、GGUF |
| 自定义 TTS | 采用分模块量化，不做“一键全模型量化” |
| 模块策略 | Transformer/DiT 主干可先试 INT8；Codec、Acoustic Decoder、Vocoder、Speaker Encoder、LayerNorm 和 Embedding 默认保留 BF16/FP16 |
| 质量判断 | 固定参考音频、文本、Seed、采样参数，比较音色相似度、清晰度、情绪表现、音色漂移和失败率 |
| 训练方向 | 先完成量化和音频分类/事件检测，再进入 TTS 情绪或音色 LoRA；暂不从零训练 TTS 或音效基础模型 |
| 操作系统 | Windows 11 适合日常使用；Ubuntu 24.04/WSL2 更适合 CUDA 扩展、量化和校准 |

### 2.2 TTS-and-VoiceDesign 当前主分支状态

- 采用“一个服务一个 uv 项目”的边界，每个模型有独立的 `pyproject.toml`、`uv.lock`、HTTP 入口和 worker。
- HTTP 控制面不加载重型模型；模型推理由一次性 worker 完成，worker 退出后释放显存。
- 本地 GPU 服务通过共享 `GPU_LOCK_FILE` 串行执行，支持排队超时、峰值显存、排队时间和 worker 执行时间记录。
- 已具备流式上传、SHA-256 内容校验、原子提交、sidecar、存储目录和运行时缓存等基础设施。
- 当前主要服务包括：Qwen3-TTS VoiceDesign、MOSS VoiceGenerator、MiMo VoiceDesign、FireRedTTS3 Instruct、Stable Audio 3 Medium、MOSS-SoundEffect、ACE-Step 1.5、Qwen3-TTS Base、VoxCPM2、LongCat-AudioDiT、dots.tts-soar、FireRedTTS3 Base 和 Step-Audio-EditX。
- 已有 `scripts/quality_gate.sh`，执行 shell 语法检查、锁文件检查、Ruff、格式检查和无模型测试。
- 现有启动脚本默认 `LOCAL_FILES_ONLY=1`，使用 `uv run --no-sync`，不会在运行阶段自动安装依赖或下载模型。

### 2.3 TTS-Studio-WebUI 当前主分支状态

- 当前主入口是根目录 `index.html`，采用 Vue 3 CDN 单文件实现；当前主分支没有以 `package.json` 为入口的 Vue/Vite 构建链。
- `webUI/` 已被文档标记为废弃或未来迁移方向，不能把历史 Vite 页面当成当前运行入口。
- 已有音色库、TTS 生成、VoxCPM2 可控克隆、LongCat、dots.tts-soar、FireRedTTS3、Step-Audio-EditX、MOSS-SoundEffect、Stable Audio、ACE-Step BGM、IndexedDB 工程资产、WAV/SRT/MP4 导出等能力。
- 工程协议当前为 `kind: "unitale-project"`、`schemaVersion: 4`。`audioAssetKey`、`bgmLibrary`、`project.currentState` 和 `assets` 是兼容性关键字段。
- 当前已有 `js/voice-design.js`、`js/soundeffect-client.js`、`js/bgm-client.js`、`js/project-storage.js` 等边界文件。
- 现有文档已经明确：生成 BGM 应复用 `bgmLibrary + IndexedDB assets`，第一阶段不提升工程 schema；UI 重构应先做存储可靠性和固定样例回归，再迁移到 Vue/Vite。

## 3. 必须遵守的约束

### 3.1 功能与兼容性红线

- 默认模型、默认端口和现有接口不能因量化实验被替换。
- 量化版本必须是可选变体，默认仍走已验证的 BF16/FP16 版本。
- 不修改 `schemaVersion: 4`，不重命名 `UnitaleDB`、`project.currentState`、`assets`、`audioAssetKey` 和旧 `localStorage` 键。
- 不把量化参数直接散落到 `index.html`，模型能力和变体应进入统一配置目录。
- 不让 WebUI 假定模型支持 `style_prompt`、`control_instruction` 或 `nonverbal_tags`；这些字段必须按 provider（模型提供商）能力决定。
- 不把量化模型、生成音频、上传音频、缓存、虚拟环境和 API Key 提交到 Git。

### 3.2 显存与运行时红线

- RTX 4070 Ti Super 只有 16GB 显存，多个重型模型不能同时常驻 GPU。
- `GPU_LOCK_FILE` 必须继续覆盖所有本地模型 worker；量化 worker 不能绕过现有锁。
- Ollama/Qwen 与 TTS 后端当前不共享同一个 GPU 锁。做显存基准或 A/B 测试时，应先暂停 Ollama 的 GPU 推理，或在独立时段运行，避免把 Ollama 占用误判为 TTS 量化显存。
- 量化实验必须支持 CPU offload 或一次性 worker，但不能为了“提高速度”把完整模型改成常驻服务。
- 24–32GB 权重转换和临时文件建议预留至少 100GB SSD 空间；48GB 内存可以支撑一部分转换，但不能等同于显存足够。

## 4. 总体目标架构

```text
模型注册表
    ↓
健康检查 / 运行时能力
    ↓
模型变体选择：bf16 / fp16 / int8 / int4-experimental
    ↓
统一 HTTP 契约
    ↓
一次性 worker
    ↓
GPU_LOCK_FILE 串行执行
    ↓
音频输出 + 运行指标 + A/B 评估清单
```

其中，`model variant`（模型变体）只改变 worker 的加载方式和权重目录，不改变现有业务接口。生产路径与实验路径必须能并存：

```text
生产：VoxCPM2 / BF16 → 8322 /v1/voxcpm2/clone
实验：VoxCPM2 / INT8 → 独立变体或独立实验端口
```

实验变体连续回归通过后，再考虑让它复用生产端口；在此之前不要覆盖原始权重或修改默认值。

## 5. 分阶段落地路线

### P0：基线冻结与文档校准

目标：先保证“现在能用的功能”可重复、可回退。

建议改动：

1. 统一两个仓库的服务清单、端口、接口、模型名称和实际启动数量。当前 README、AGENTS 和历史文档存在服务数量与端口描述不同步的风险，应以 `start.sh` 和实际健康检查为唯一运行事实。
2. 建立 `docs/量化与显存回归.md`，记录每个模型的权重格式、模型目录、dtype、预计显存、实际峰值、生成耗时和已知限制。
3. 固定一组中文悬疑/恐怖语音回归集：旁白、平静对白、恐惧、压低声音、急促、叹气/笑声等；每条记录参考音频 SHA-256、文本、Seed、采样参数和目标模型。
4. 固定一组 SoundEffect/BGM 回归提示词，区分 MOSS-SoundEffect、Stable Audio 3 和 ACE-Step 的职责。
5. 默认把后端监听地址收敛到本机回环地址；确需局域网访问时再显式设置 `HOST`，避免本地服务无意暴露。

验收：

- `bash scripts/quality_gate.sh` 通过。
- 所有当前服务的健康检查地址和 README/前端接入文档一致。
- 现有 WebUI 固定样例可以完成导入、刷新恢复、试听、导出 WAV/SRT/MP4。
- 在不启用量化的情况下，现有生成结果和请求字段没有行为变化。

### P1：建立量化实验与显存观测工具

目标：先学会“看懂模型”和“记录变化”，不急着改线上模型。

后端建议新增：

```text
tools/quantization/
  inspect_model.py       # 统计模块、参数量、dtype、可替换 Linear
  benchmark_runtime.py  # 记录加载、排队、推理、峰值显存
  audio_ab_compare.py    # 生成固定样本并输出比较清单
docs/量化与显存回归.md
```

工具至少记录：

| 指标 | 说明 |
| --- | --- |
| model_id / variant | 模型和变体，例如 `longcat-bf16`、`longcat-int8-exp` |
| reference_sha256 | 参考音频内容哈希 |
| text_sha256 | 输入文本哈希，避免长文本直接进入报告 |
| seed / sampling | 随机种子与采样参数 |
| dtype | BF16、FP16、INT8 等 |
| quantized_modules | 实际被量化的模块列表 |
| peak_vram_mb | 持锁期间峰值显存 |
| queue_ms / worker_ms | 排队和推理耗时 |
| output_audio | 输出 WAV 路径或报告中的相对路径 |
| status / error | 成功、OOM、加载失败、音频校验失败等 |

验收：

- 连续执行 10 次相同请求，显存不会持续增长。
- worker 成功、失败、超时后都释放 GPU 锁和临时文件。
- 同一参考音频、文本、Seed 和采样参数可以重复生成并形成两份可比较记录。
- 工具不依赖真实权重的单元测试可以进入现有质量门禁。

### P2：先完成标准 Transformer 量化，再进入自定义 TTS

目标：把量化学习拆成低风险实验。

第一步使用标准 Transformers 模型练习 bitsandbytes：

- 先做 INT8，再做 NF4/4bit 加载。
- 验证 `device_map`、CPU offload、模型保存和重新加载。
- 记录模型显存、生成速度和文本输出差异。
- 明确 NF4 不是 GGUF，不能直接用 llama.cpp。

第二步使用一个音频 DiT/Transformer 进行 TorchAO 或 Quanto 实验：

- 优先选择已经能在当前项目中以 BF16 正常运行、且有清晰模块边界的模型。
- 推荐把 `LongCat-AudioDiT-3.5B-BF16` 作为第一候选实验对象；它适合验证 DiT 主干的选择性量化。
- `VoxCPM2` 继续作为自然度和音色质量参考，不建议作为第一个被改造的模型。
- `Qwen3-TTS 1.7B` 当前体量较小，量化的主要价值应是统一运行时和实验经验，而不是单纯追求省显存。

第一轮只尝试：

```text
Linear / Transformer / DiT 主干 → INT8 或 TorchAO weight-only
Speaker Encoder                → BF16/FP16
Audio Codec / Acoustic Decoder  → BF16/FP16
Vocoder                        → BF16/FP16
LayerNorm / Embedding           → 原精度
```

只有 INT8 版本在音质、稳定性和显存上都通过，才评估 INT4。若量化后出现音色漂移、爆音、断句异常、情绪失真或长文本不稳定，应保留该实验结果但停止继续压缩。

### P3：后端增加“模型变体”而不是复制一套业务接口

目标：让 BF16 与实验量化版本可以并存、可查看、可回退。

建议新增轻量运行时配置，字段示例：

```json
{
  "model_id": "longcat-audiodit",
  "variant": "int8-exp",
  "weights_dir": "/models/LongCat-AudioDiT-3.5B-int8",
  "compute_dtype": "bfloat16",
  "quant_method": "torchao",
  "quantized_modules": ["transformer.blocks.*.linear"],
  "fallback_variant": "bf16"
}
```

实现要求：

- `main.py` 只负责校验配置、启动 worker 和返回音频，不导入量化库。
- `worker.py` 负责加载量化配置；不支持量化时必须返回明确错误，不能静默使用错误权重。
- `GET /v1/health` 增加轻量 `runtime.variant`、`runtime.dtype`、`runtime.quant_method`、`runtime.available` 和 `runtime.fallback` 信息，但健康检查不能加载模型。
- 量化实验独立使用模型目录、缓存目录和日志标识；禁止覆盖 BF16 权重。
- 继续使用 `uv sync --project <dir> --locked`，启动脚本继续使用 `uv run --no-sync`。
- 量化依赖只放在对应服务的 uv 项目中，不污染控制面和其他模型环境。

推荐的第一批文件：

```text
TTS-and-VoiceDesign/
  <candidate_service>/main.py
  <candidate_service>/runtime.py
  <candidate_service>/worker.py
  <candidate_service>/pyproject.toml
  <candidate_service>/tests/
  tools/quantization/
  docs/量化与显存回归.md
  README.md
  AGENTS.md
  start.sh
```

### P4：WebUI 增加能力识别、变体显示和 A/B 入口

目标：前端能够知道“当前调用的是哪个模型变体”，但不让用户直接接触复杂的 Python/量化细节。

建议新增：

```text
js/model-registry.js       # 统一 TTS、音色设计、音效、BGM provider 配置
js/health-client.js        # 读取 /v1/health 和可用能力
js/benchmark-client.js     # 固定样本 A/B 请求与结果记录
```

前端模型卡片应展示：

- 模型名称与用途：音色设计、克隆、编辑、SoundEffect、BGM。
- Base、INT8、INT4-experimental 等变体。
- 当前健康状态、端口、是否需要参考文案、是否支持 VoxCPM2 可控克隆。
- dtype、量化方式、已量化模块、最近一次峰值显存和失败原因。

A/B 流程必须固定：

1. 选择同一角色、同一参考音频、同一文本。
2. 固定 Seed、采样参数、速度、音量和停顿。
3. 后端按 GPU 锁串行生成两个版本。
4. 浏览器保存两个结果的 `audioAssetKey`，并提供并排播放、下载和人工评分。
5. 生成独立的 JSON/Markdown 评估报告，不写入 `schemaVersion: 4` 的工程协议。

现有 IndexedDB、`bgmLibrary`、`audioAssetKey`、导入导出和时间轴逻辑继续复用。量化 A/B 报告不应新增一套平行音频资产系统。

### P5：在行为基线稳定后再推进 Vue/Vite 工程化重构

目标：解决当前 `index.html` 单文件过重的问题，但不以重写 UI 作为量化工作的前置条件。

推荐顺序：

1. 先用固定工程样例保护 `schemaVersion: 4` 和 IndexedDB 资产恢复。
2. 抽出 `domain/project`、`domain/script` 和 `services/storage`。
3. 抽出 `services/providers`，让 TTS、音色设计、音效和 BGM 都通过模型注册表接入。
4. 抽出音频播放、裁剪、混音和导出服务。
5. 迁移低风险页面，再迁移脚本工作台和批量生成。
6. 新工程通过导入、刷新、播放、导出和真实后端回归后，再切换默认入口。

不要在 P5 之前同时进行“大规模 UI 改版、路由化、模型抽象、量化和存储协议升级”。这些任务会让问题难以定位。

## 6. 推荐的执行批次

| 批次 | 建议任务标题 | 主要仓库 | 完成标志 |
| --- | --- | --- | --- |
| Q0 | 对齐服务清单与运行文档 | 两个仓库 | 端口、路由、模型、启动数量一致 |
| Q1 | 建立固定音频和显存基线 | 后端 | 有可重复的 benchmark manifest |
| Q2 | 增加模型结构检查和峰值显存记录 | 后端 | 10 次连续请求无泄漏 |
| Q3 | 标准 Transformer bitsandbytes 实验 | 独立实验目录 | INT8/NF4 可加载、可保存、可回归 |
| Q4 | LongCat DiT 选择性 INT8 实验 | 后端 | BF16 与 INT8 A/B 报告完整 |
| Q5 | 健康检查暴露模型变体 | 后端 | WebUI 能识别 variant，不加载模型即可返回 |
| Q6 | WebUI 模型注册表与变体卡片 | 前端 | 不改变旧工程即可切换模型变体 |
| Q7 | WebUI A/B 试听与评估报告 | 前端 | 两个结果可播放、下载、评分、导出 |
| Q8 | 迁移低风险页面到 Vue/Vite | 前端 | 新旧入口并存，固定样例回归通过 |
| Q9 | TTS 情绪/音色 LoRA 可行性评估 | 两个仓库 | 仅在有数据集和模型训练代码后立项 |

建议先完成 Q0–Q4，再决定是否继续 Q5–Q9。Q4 之前不要把量化版本当作生产默认值。

## 7. 你的机器与环境建议

| 资源 | 使用建议 |
| --- | --- |
| RTX 4070 Ti Super 16GB | 单模型串行推理、INT8/部分 INT4 实验、音频 A/B；避免多个重模型常驻 |
| 48GB RAM | 模型转换、CPU offload、临时权重加载；大型转换仍要观察交换空间 |
| i5-13600KF | 运行静态前端、worker 编排、音频预处理和无模型测试；不要期待 CPU offload 等于高速度 |
| Windows 11 | 浏览器、日常 WebUI、最终工作流使用 |
| Ubuntu 24.04/WSL2 | uv 环境、CUDA 扩展、TorchAO、模型转换和 benchmark；推荐作为后端主环境 |
| Ollama + Qwen3.5 9B | 脚本分析、角色/情绪/音效计划；与 TTS GPU benchmark 分时运行 |
| 模型目录 | 统一使用本地 `hf-mirror`，启用 `LOCAL_FILES_ONLY=1`，不在运行期联网下载 |
| 临时空间 | 24–32GB 权重转换建议至少准备 100GB SSD 空间，并保留原始权重只读备份 |

推荐目录分工：

```text
~/projects/TTS-and-VoiceDesign
~/projects/TTS-Studio-WebUI
~/hf-mirror/                 # 原始模型和官方/社区权重
~/tts-quant-experiments/     # 独立量化输出，不覆盖原权重
~/tts-benchmarks/            # JSON 清单、日志、A/B 音频和报告
```

## 8. 统一验收标准

### 后端

- `bash -n start.sh` 通过。
- `bash scripts/quality_gate.sh` 通过。
- 所有无模型测试不下载权重、不调用云端 API、不需要 CUDA。
- BF16/FP16 默认路径行为不变。
- 量化变体明确报告 `variant`、`dtype`、`quant_method` 和 `quantized_modules`。
- 量化失败时返回可定位错误并回退到显式指定的 BF16 版本，不能静默混用模型目录。
- 连续 10 次生成后显存没有单调增长；成功、异常、超时都能释放锁和 worker。
- 音频输出采样率、声道、Content-Type、文件保存路径与现有 WebUI 契约一致。

### WebUI

- 不修改 `schemaVersion: 4` 和既有 IndexedDB 资产结构。
- 旧工程导入、刷新恢复、试听、裁剪、顺序播放、BGM、SoundEffect、WAV/SRT/MP4 导出全部通过回归。
- 模型变体不可用时显示原因，不把失败变体静默替换成其他模型。
- A/B 生成使用相同参考音频、文本、Seed 和采样参数。
- `audioAssetKey` 只作为 IndexedDB 查找键，不拼接为静态 URL。
- 量化报告与工程文件分离，避免导入评估报告覆盖用户的模型配置。

### 音频质量

每个量化候选至少完成以下人工和自动记录：

- 音色相似度：是否仍然像同一角色。
- 清晰度：吐字、齿音、爆音、断字和静音异常。
- 表演能力：平静、恐惧、压迫、急促、低语等场景是否仍然可用。
- 稳定性：长文本、中文标点、数字、括号控制和非语言标签是否异常。
- 音色漂移：多句连续生成后，角色是否逐渐偏离参考音色。
- 工程指标：峰值显存、排队时间、worker 时间、失败率、输出时长和文件大小。

## 9. 暂不做的事项

- 暂不开发通用“一键量化所有 TTS 模型”脚本。
- 暂不把 VoxCPM2、MOSS、Fish Speech 等自定义 TTS 强行转 GGUF。
- 暂不把所有模块都压到 INT4。
- 暂不因为量化实验重写现有 API 或工程 schema。
- 暂不从零训练 TTS 或环境音效基础模型。
- 暂不在量化、UI 重写、云端中转和多人协作之间同时开工。

## 10. 当前下一步

建议下一张开发任务只做 Q0：

1. 对比两个仓库当前 `main` 的实际端口和 README/AGENTS/接入文档。
2. 修正文档漂移，确定一份服务清单。
3. 建立固定回归音频和 JSON manifest，不改生产推理逻辑。
4. 为每个模型记录 BF16/FP16 基线，先不加入量化参数。
5. 基线通过后，再单独创建 Q1/Q2 任务进入显存和模型结构观测。

这样做可以保证后续即使某个 TorchAO、bitsandbytes 或自定义 worker 实验失败，也不会影响当前已经能完成的音色设计、语音克隆、音频编辑、SoundEffect、BGM 和成片导出流程。

## 11. 参考资料

- [模型量化学习与训练路线分享对话](https://chatgpt.com/share/6a8a2f9d-6774-83ec-8bc7-944e6c693c06)
- [TTS-and-VoiceDesign README](https://github.com/MuYi086/TTS-and-VoiceDesign/blob/main/README.md)
- [TTS-and-VoiceDesign AGENTS.md](https://github.com/MuYi086/TTS-and-VoiceDesign/blob/main/AGENTS.md)
- [TTS-and-VoiceDesign start.sh](https://github.com/MuYi086/TTS-and-VoiceDesign/blob/main/start.sh)
- [TTS-Studio-WebUI README](https://github.com/MuYi086/TTS-Studio-WebUI/blob/main/README.md)
- [TTS-Studio-WebUI AGENTS.md](https://github.com/MuYi086/TTS-Studio-WebUI/blob/main/AGENTS.md)
- [WebUI 本地开发与回归](https://github.com/MuYi086/TTS-Studio-WebUI/blob/main/docs/本地开发与回归.md)
- [WebUI 重构顺序计划](https://github.com/MuYi086/TTS-Studio-WebUI/blob/main/docs/重构顺序计划.md)

