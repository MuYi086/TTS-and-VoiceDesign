# Easy Dataset 与 LLaMA-Factory 训练操作指南

> 目标：训练 `audio-polisher-lora`，把悬疑/恐怖原稿 A 改写为能进入现有导演层的纯文本 B。  
> 目标硬件：Windows 11 + Ubuntu 24.04，RTX 4070 Ti Super 16GB，内存 48GB。  
> 推荐起点：Qwen3.5-4B、4-bit QLoRA、8K 训练长度。  
> 方案版本：1.0（2026-08-26）。不同版本界面文字可能略有变化，以官方仓库当前 README 和示例为准。

## 1. 最终要得到什么

完成本指南后，应保留以下产物：

```text
training/
├── README.md
├── prompts/
│   └── audio_polisher_system.txt
├── data/
│   └── story_audio_polish/
│       ├── dataset_info.json
│       ├── story_audio_polish_train.json
│       ├── story_audio_polish_validation.json
│       ├── story_audio_polish_test.json
│       └── manifest.json
├── configs/
│   ├── train_audio_polisher_qwen35_4b_qlora.yaml
│   ├── infer_audio_polisher_qwen35_4b.yaml
│   └── merge_audio_polisher_qwen35_4b.yaml
├── eval/
│   ├── gold_test.jsonl
│   └── eval_report.md
└── adapters/
    └── audio-polisher-qwen35-4b/
```

这些训练文件建议放进项目仓库，但以下大文件不要提交 Git：

- 原始/合并模型权重。
- Easy Dataset 数据库和上传原文。
- 受版权保护的 PDF、音频、转写和训练样本。
- checkpoint、缓存、日志中的密钥。

只提交脱敏后的配置、数据 schema、提示词、评测规则和允许分发的样本。

## 2. 开始前的硬性检查

### 2.1 权利与数据来源

每个来源至少记录：

```json
{
  "source_id": "magazine-001-story-03",
  "title": "内部标题",
  "rights": "owned|licensed|public-domain|research-only",
  "license_note": "授权文件或许可证位置",
  "allowed_training": true,
  "allowed_distribution": false
}
```

无法确认训练权限的期刊、小说和市面音频，不进入训练数据。Easy Dataset 的 AGPL-3.0 软件许可证不等于你导入内容的版权许可。

### 2.2 环境隔离

- Easy Dataset 可以运行在 Windows、Linux 或 Docker。
- LLaMA-Factory 推荐在 Ubuntu 24.04 原生环境训练；如果使用 WSL2，先确认 GPU 透传稳定。
- LLaMA-Factory、Easy Dataset、TTS-and-VoiceDesign 和 TTS-Studio-WebUI 使用不同目录和虚拟环境。
- 训练期间关闭 TTS、VoxCPM2、CosyVoice、IndexTTS2 及其他占用显存的进程。
- 检查现有 `GPU_LOCK_FILE` 对应进程，不得删除有效锁强行训练。

### 2.3 基座一致性

LoRA 只能加载到训练时完全相同的基座架构和版本。训练前冻结以下信息：

```text
model_id: Qwen/Qwen3.5-4B
model_revision: 具体 commit 或下载快照
template: qwen3_5_nothink
enable_thinking: false
llamafactory_commit: git rev-parse HEAD 的结果
```

如果最终必须使用 Qwen3.5-9B，则应重新在同版本 9B 上训练 LoRA，不能把 4B 适配器挂到 9B。

## 3. 先建立不训练的基线

在制作大规模数据前，从未参与训练的 20～30 个故事中选 80～120 个场景，建立 `gold_test.jsonl`。每条包含 A、人工 Gold B、关键事件、人物、说话人和禁止改变项。

先用基座模型和固定提示词生成 B。建议系统提示词：

```text
你是中文悬疑有声演播文本润色器。

任务：把用户提供的原稿改写为适合有声演播的纯文本。

必须遵守：
1. 保留全部事实、关键事件、人物关系、因果、时间线、线索和揭示顺序。
2. 只改善口语自然度、句子长度、停连节奏、对白归属和指代清晰度。
3. 不新增人物、动机、线索、场景、解释或结局，不删减关键内容。
4. 保持悬疑与恐怖气氛，不用夸张形容词替代实际情节。
5. 只输出改写后的中文正文；不输出分析、说明、Markdown、JSON、标题、角色标签、情绪标签、SFX、BGM 或 TTS 参数。
```

建议用户消息模板：

```text
请将以下原稿改写为适合中文悬疑有声演播的纯文本。

<原稿>
{{source_text}}
</原稿>
```

固定推理参数并保存结果。若基座已经达到《训练评估方案》的全部门槛，停止训练，直接使用提示词 + 规则层。

## 4. 安装 Easy Dataset

官方支持桌面客户端、NPM 和 Docker。以下优先选择 Docker，便于持久化和隔离；也可按官方 NPM 方式运行。

### 4.1 Docker 方式

```yaml
services:
  easy-dataset:
    image: ghcr.io/conardli/easy-dataset
    container_name: easy-dataset
    ports:
      - "1717:1717"
    volumes:
      - ./local-db:/app/local-db
      - ./prisma:/app/prisma
    restart: unless-stopped
```

```bash
docker compose up -d
```

浏览器打开 `http://localhost:1717`。`local-db` 和 `prisma` 目录需要纳入本地备份，但不要提交到公共仓库。

### 4.2 NPM 方式

```bash
git clone https://github.com/ConardLi/easy-dataset.git
cd easy-dataset
npm install
npm run build
npm run start
```

浏览器打开 `http://localhost:1717`。

## 5. 在 Easy Dataset 构造 A→B 候选

### 5.1 新建专用项目

创建项目 `suspense-audio-polisher-v1`。不要与“作者 LoRA”或普通知识问答数据混用。

模型提供方可配置为任意 OpenAI 兼容 API。生成 B 的教师模型应比目标 4B 基座更强，temperature 建议从 `0.2` 开始；生成时追求保真和稳定，不追求随机创作。

### 5.2 文档预处理

- 可直接上传 PDF、Markdown、DOCX、TXT、EPUB。
- 扫描 PDF 或复杂排版 PDF 先用 MinerU/OCR 转换，再抽查至少 10 页。
- 删除页眉、页脚、广告、目录、版权页、重复章节和 OCR 噪声。
- 保留说话人线索、引号、段落和章节边界。
- 每个文件只放一个故事；文件名包含稳定 `story_id`。

### 5.3 切分

推荐每个文本块是一段完整场景或完整叙事单元，通常 800～2,500 个汉字。切分时：

- 不在一句话、对白轮次、关键揭示或因果链中间截断。
- 同一故事可有多个块，但最终数据切分必须按整个故事分组。
- 如果一个场景的 A+B 预计超过 8K token，再缩短场景；不要单纯按固定字符数乱切。
- 切分后逐块预览，发现 OCR 错误立即返回修正。

### 5.4 创建固定问题模板

在“问题/问题模板”区域创建文本模板，问题内容使用：

```text
请在不改变任何事实、人物关系、因果、时间线、关键线索和揭示顺序的前提下，将参考原稿改写为适合中文悬疑有声演播的纯文本。改善口语自然度、句子长度、停连节奏、对白归属和指代清晰度。不得新增或删减关键内容。只输出改写后的正文，不输出分析、Markdown、JSON、标题、角色/情绪/SFX/BGM/TTS 标签。
```

使用“从模板生成问题”，让每个文本块只产生一个改写任务。不要使用默认“从文档生成若干事实问答”的流程；那会训练知识问答，而不是 A→B 变换。

### 5.5 自定义答案提示词

在项目设置的 Prompt Settings/提示词设置中，修改中文 Answer Prompt。必须保留 `{{text}}` 和 `{{question}}` 两个变量。可使用：

```text
# 角色
你是中文悬疑有声演播文本润色器。

# 原稿 A
------ 原稿 Start ------
{{text}}
------ 原稿 End ------

# 任务
{{question}}

# 硬约束
1. 逐项保留人物、地点、物件、数字、关键事件、因果、时间线、线索和揭示顺序。
2. 只处理口语自然度、过长句、停连节奏、对白归属和指代歧义。
3. 不新增人物、动机、线索、场景、解释、评价或结局；不删减关键内容。
4. 悬念来自原情节，不靠无根据的形容词和感叹号制造。
5. 只输出改写后的中文正文。
6. 禁止输出分析、思维链、Markdown、JSON、标题、角色标签、情绪标签、SFX、BGM 或 TTS 参数。
```

### 5.6 批量生成与人工确认

批量生成 B 后，每条至少完成以下对照：

- 人名、身份、地点、数字、时间全部核对。
- 关键事件列表逐项核对。
- 对白说话人和代词无歧义。
- 没有新增解释、提前泄底或改变揭示顺序。
- 输出是纯正文，无 `<think>`、代码块、JSON、标签。
- 朗读时长和文本长度合理。

不合格则人工修改或重新生成；合格后标记 Confirmed/已确认。未经确认的数据不进入训练。

注意：Easy Dataset 当前实现可能在答案没有思维链时额外合成 COT。训练数据不要包含 COT；后续装配脚本只读取 `answer`，忽略 `cot`。

## 6. 关键步骤：把 Easy Dataset 结果装配成真正的 A→B 数据

### 6.1 为什么不能直接用一键导出

当前 Easy Dataset 的 LLaMA-Factory 导出逻辑把 `question` 写入 user 消息，把 `answer` 写入 assistant 消息；生成答案时使用的原文块 A 不会被自动加入 user 消息。直接训练会丢失输入 A。

训练前随机抽查 20 条，必须看到：

```text
user = 任务说明 + 完整原稿 A
assistant = 对应演播成稿 B
```

只要 user 中没有 A，立即停止训练。

### 6.2 从 Easy Dataset 本地 SQLite 只读装配

Easy Dataset 的数据模型保留 `Datasets.questionId → Questions.chunkId → Chunks.content` 的关联。可用下列只读脚本把已确认的 B 与原始块 A 重新配对，并按完整文件/故事分组切分。

把脚本保存为 `build_audio_polisher_dataset.py`：

```python
#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

SYSTEM = (
    "你是中文悬疑有声演播文本润色器。保持全部事实、人物关系、因果、时间线、"
    "关键线索和揭示顺序不变，只输出适合演播的中文纯文本。"
)

TASK = (
    "请将以下原稿改写为适合中文悬疑有声演播的纯文本。只改善口语自然度、"
    "句子长度、停连节奏、对白归属和指代清晰度；不得新增或删减关键内容；"
    "不得输出分析、Markdown、JSON、标题、角色/情绪/SFX/BGM/TTS 标签。"
)

FORBIDDEN = ("<think>", "```", "[SFX]", "[BGM]", "Unitale", "schemaVersion")


def norm(text):
    return unicodedata.normalize("NFKC", text or "").strip()


def make_sample(source, target):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"{TASK}\n\n<原稿>\n{source}\n</原稿>",
            },
            {"role": "assistant", "content": target},
        ]
    }


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Easy Dataset 的 db.sqlite")
    ap.add_argument("--project-id", required=True, help="浏览器项目 URL 中的 projectId")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    db_path = Path(args.db).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT d.id AS pair_id,
               d.answer AS target_text,
               c.content AS source_text,
               c.fileName AS story_id,
               c.name AS chunk_name
        FROM Datasets d
        JOIN Questions q ON q.id = d.questionId
        JOIN Chunks c ON c.id = q.chunkId
        WHERE d.projectId = ? AND d.confirmed = 1
        ORDER BY c.fileName, c.createAt, d.createAt
        """,
        (args.project_id,),
    ).fetchall()

    if not rows:
        raise SystemExit("没有找到可关联的已确认数据；检查 db、projectId 和 Confirmed 状态。")

    records = []
    seen_sources = set()
    for row in rows:
        source = norm(row["source_text"])
        target = norm(row["target_text"])
        if not source or not target:
            continue
        if any(token in target for token in FORBIDDEN):
            raise SystemExit(f"样本 {row['pair_id']} 含禁用标记")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest in seen_sources:
            continue
        seen_sources.add(digest)
        records.append(
            {
                "pair_id": row["pair_id"],
                "story_id": row["story_id"] or row["chunk_name"],
                "source_sha256": digest,
                "source_chars": len(source),
                "target_chars": len(target),
                "sample": make_sample(source, target),
            }
        )

    stories = sorted({r["story_id"] for r in records})
    if len(stories) < 10:
        raise SystemExit("独立故事少于 10 个，无法可靠做故事级 train/val/test 划分。")

    rng = random.Random(args.seed)
    rng.shuffle(stories)
    n_test = max(1, round(len(stories) * 0.10))
    n_val = max(1, round(len(stories) * 0.10))
    test_ids = set(stories[:n_test])
    val_ids = set(stories[n_test:n_test + n_val])

    groups = defaultdict(list)
    manifest = []
    for r in records:
        split = "test" if r["story_id"] in test_ids else "validation" if r["story_id"] in val_ids else "train"
        groups[split].append(r["sample"])
        manifest.append({k: r[k] for k in r if k != "sample"} | {"split": split})

    for split in ("train", "validation", "test"):
        write_json(out / f"story_audio_polish_{split}.json", groups[split])

    dataset_info = {
        "story_audio_polish_train": {
            "file_name": "story_audio_polish_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "story_audio_polish_validation": {
            "file_name": "story_audio_polish_validation.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "story_audio_polish_test": {
            "file_name": "story_audio_polish_test.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
    }
    write_json(out / "dataset_info.json", dataset_info)
    write_json(out / "manifest.json", manifest)
    print({split: len(groups[split]) for split in ("train", "validation", "test")})


if __name__ == "__main__":
    main()
```

找到数据库：

```bash
find /path/to/easy-dataset -maxdepth 4 -name 'db.sqlite' -type f
```

NPM/Docker 源码安装通常使用 `prisma/db.sqlite`；桌面版位置可能不同，以实际文件为准。先停止 Easy Dataset 或复制一份数据库快照，再对快照执行只读脚本。

从浏览器 URL 取得 `projectId`，然后运行：

```bash
python build_audio_polisher_dataset.py \
  --db /absolute/path/to/db.sqlite \
  --project-id YOUR_PROJECT_ID \
  --out /absolute/path/to/LlamaFactory/data/story_audio_polish \
  --seed 42
```

如果数据是通过 Easy Dataset“导入数据集”功能导入的，其 `questionId` 可能没有对应原始 Chunks，不能使用上述 JOIN。此时在导入前就把 `question` 组装成“任务 + 完整 A”，把 `answer` 设为 B；导出后再做抽样核验。

### 6.3 必做数据检查

```bash
python -m json.tool data/story_audio_polish/dataset_info.json >/dev/null
python -m json.tool data/story_audio_polish/story_audio_polish_train.json >/dev/null
python -m json.tool data/story_audio_polish/story_audio_polish_validation.json >/dev/null
python -m json.tool data/story_audio_polish/story_audio_polish_test.json >/dev/null
```

人工随机抽查至少 20 条：

- user 中有完整 A，不是只有固定任务。
- assistant 是对应 B。
- train/validation/test 没有相同 `story_id`。
- B 无分析、COT、JSON 和导演标签。
- A/B 没有串故事。
- 测试集没有进入训练集。

建议再用 Qwen3.5 tokenizer 统计 `system + user + assistant` token；超过 `cutoff_len` 的样本会被截断，不能默默接受。

## 7. 安装 LLaMA-Factory

### 7.1 Ubuntu 24.04 环境

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与你驱动匹配的 CUDA 版 PyTorch，然后：

```bash
pip install -e .
pip install -r requirements/metrics.txt
pip install bitsandbytes
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY

llamafactory-cli version
git rev-parse HEAD
```

若 CUDA 不可用，停止训练并修复环境。不要让 pip 安装的 CPU 版 PyTorch继续跑。

### 7.2 模型准备

下载 `Qwen/Qwen3.5-4B` 原始 Transformers 权重到固定本地目录，例如：

```text
/data/models/Qwen3.5-4B/
```

要求：

- 不是 Ollama 模型目录。
- 不是 GGUF 文件。
- 保留 tokenizer、config 和 safetensors。
- 记录下载 revision/hash。
- 生产离线运行时把模型和 adapter 预先落盘，继续遵守 `LOCAL_FILES_ONLY=1`。

## 8. 编写 QLoRA 训练配置

保存为 `configs/train_audio_polisher_qwen35_4b_qlora.yaml`：

```yaml
### model
model_name_or_path: /data/models/Qwen3.5-4B
trust_remote_code: true
quantization_bit: 4
quantization_method: bnb
flash_attn: auto

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

### dataset
dataset_dir: ./data/story_audio_polish
dataset: story_audio_polish_train
eval_dataset: story_audio_polish_validation
template: qwen3_5_nothink
enable_thinking: false
cutoff_len: 8192
packing: false
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 2

### output
output_dir: ./saves/audio-polisher-qwen35-4b/lora/sft
logging_steps: 5
save_steps: 50
eval_steps: 50
eval_strategy: steps
save_total_limit: 2
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
seed: 42
data_seed: 42
ddp_timeout: 180000000
resume_from_checkpoint: null
```

说明：

- `template` 在训练和推理时必须完全相同。
- `qwen3_5_nothink + enable_thinking: false` 避免把思维链混进纯文本输出。
- `cutoff_len: 8192` 是 16GB 显存上的起始值，不是保证值。
- `gradient_accumulation_steps` 增加等效批量，不降低单次前向的显存峰值。
- 先不启用 FlashAttention、Unsloth、Liger 等额外优化；基础链路稳定后再逐项 A/B 测试。

如果 9B 是硬性生产目标，复制配置并修改基座和输出目录，起始 `cutoff_len` 建议 4096 或 6144；先做 16 条 smoke test，不要直接跑完整训练。

## 9. 先做 smoke test

```bash
cd /path/to/LlamaFactory
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  /absolute/path/to/configs/train_audio_polisher_qwen35_4b_qlora.yaml \
  max_samples=16 \
  num_train_epochs=0.1 \
  output_dir=./saves/smoke-audio-polisher
```

smoke test 必须确认：

- 数据被识别为 ShareGPT/messages。
- 日志中模板是 `qwen3_5_nothink`。
- 没有把 test 数据集加载为训练数据。
- GPU 正常使用，显存不溢出。
- 输出目录生成 adapter 文件。
- 解码一条样本时只返回纯 B。

OOM 排查顺序：

1. 确认没有其他 GPU 进程。
2. 保持 batch size 为 1。
3. `cutoff_len: 8192 → 6144 → 4096`。
4. `dataloader_num_workers: 0`，排除系统内存/Windows 多进程问题。
5. 把超长故事改为完整场景切分。
6. 若仍失败，使用 4B 而不是 9B。

不要先把量化降到 2-bit；那会给文本保真带来额外质量风险。

## 10. 正式训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  /absolute/path/to/configs/train_audio_polisher_qwen35_4b_qlora.yaml
```

训练中记录：

- LLaMA-Factory commit。
- 完整 YAML。
- 基座模型 revision。
- `manifest.json` hash。
- train/eval loss。
- GPU 型号、驱动、CUDA、PyTorch、bitsandbytes 版本。
- 开始/结束时间和异常。

判断训练是否可用不能只看 loss。若 validation loss 下降但测试文本开始漏剧情、添加细节或模板化严重，应以 Gold 结果为准。

首次建议 3 epoch；不要因为训练 loss 继续下降就无限增加 epoch。300～500 条数据很容易过拟合教师文风。

## 11. 加载 LoRA 验证

保存为 `configs/infer_audio_polisher_qwen35_4b.yaml`：

```yaml
model_name_or_path: /data/models/Qwen3.5-4B
adapter_name_or_path: ./saves/audio-polisher-qwen35-4b/lora/sft
template: qwen3_5_nothink
enable_thinking: false
infer_backend: huggingface
trust_remote_code: true
```

CLI 测试：

```bash
llamafactory-cli chat \
  /absolute/path/to/configs/infer_audio_polisher_qwen35_4b.yaml
```

WebChat：

```bash
llamafactory-cli webchat \
  /absolute/path/to/configs/infer_audio_polisher_qwen35_4b.yaml
```

OpenAI 风格 API：

```bash
llamafactory-cli api \
  /absolute/path/to/configs/infer_audio_polisher_qwen35_4b.yaml
```

先用 API 验证 LoRA 本身，不要立刻改 WebUI。API 输出通过后，再把 B 接入现有导演层。

## 12. 固定评测

### 12.1 文本层

对同一封存测试集生成两套结果：

- `baseline`：原始基座 + 同一系统提示词。
- `lora`：同一基座 + `audio-polisher-lora` + 同一提示词。

固定 temperature、top_p、seed、max_new_tokens。逐条检查：

```text
关键事件召回
新增事实
人物/身份/时间线
对白归属
指代清晰
演播自然度
悬念揭示顺序
禁用格式
```

任何剧情硬错误直接 Gold fail，不被平均分抵消。

### 12.2 导演层

把 B 原样交给现有 WebUI 导演层，不让 LoRA 生成项目 JSON。检查：

- 输出可解析。
- `kind: "unitale-project"`。
- `schemaVersion: 4`。
- `audioAssetKey`、`bgmLibrary`、`project.currentState`、`assets` 等现有字段不丢失。
- 旧项目可继续载入。

### 12.3 TTS 层

第一轮：

- 固定 TTS 后端、声线、参考音频、seed 和分段参数。
- 关闭 SFX/BGM，只比较 baseline B 与 LoRA B 的语音自然度和稳定性。

第二轮：

- 开启完整导演、SFX/BGM 和混音。
- 检查角色清晰、悬念节奏、音效位置和端到端成功率。

上线门槛使用《训练评估方案》中的 Gold 门槛；LoRA 至少在 30 条双盲样本中达到 65% 偏好率，且不得增加语义硬错误。

## 13. 合并 LoRA（通过评测后才做）

保存为 `configs/merge_audio_polisher_qwen35_4b.yaml`：

```yaml
### model
model_name_or_path: /data/models/Qwen3.5-4B
adapter_name_or_path: ./saves/audio-polisher-qwen35-4b/lora/sft
template: qwen3_5_nothink
trust_remote_code: true

### export
export_dir: ./exports/audio-polisher-qwen35-4b-merged
export_size: 5
export_device: cpu
export_legacy_format: false
```

执行：

```bash
llamafactory-cli export \
  /absolute/path/to/configs/merge_audio_polisher_qwen35_4b.yaml
```

合并时不能填写 `quantization_bit`，也不能使用 GGUF/Ollama 量化模型作为基座。48GB 系统内存通常足以在 CPU 合并 4B/9B，但仍需保留磁盘和内存余量。

先保留 adapter 版本作为可回滚源；合并模型、GGUF/Ollama 转换和最终量化属于部署阶段。每次转换后重新跑固定 Gold 测试，因为量化可能改变细节保真和输出格式。

## 14. 接入现有项目

推荐调用顺序：

```text
用户题材/梗概
  → story-writer-lora（以后再训练）
  → 原稿 A
  → audio-polisher-lora
  → 纯文本 B
  → 现有 WebUI 导演层
  → Unitale C
  → 现有 TTS worker
  → 音频 D
```

接入要求：

- Writer 与 Polisher 是独立 adapter，按顺序加载/调用。
- POC 不合并两个 LoRA，也不同时激活。
- B 经过确定性预处理和后校验后才进入导演层。
- 数字、日期、百分比、英文缩写和特殊符号的朗读规范由规则层处理。
- 训练/推理服务不要绕过现有 GPU 锁和一次性 worker 资源回收机制。
- 模型、adapter 和 tokenizer 在部署前完整下载；运行时禁止自动下载或安装。
- WebUI 项目 Schema 不因增加 LoRA 而改变。

## 15. 常见错误

### 错误 1：直接训练 Easy Dataset 的 ShareGPT 导出

症状：每条 user 都是同一句“请改写”，没有原稿 A。  
结果：模型学习固定风格生成，无法根据输入改写。  
修复：执行第 6 节成对装配；随机抽查 user 中确实有 A。

### 错误 2：把小说原文直接做 SFT 输出

症状：输入是“写一个恐怖故事”，输出是期刊段落。  
结果：只能学习风格模仿，不会 A→B。  
修复：作者 LoRA 与润色 LoRA 分开；润色集必须是显式 A/B 对。

### 错误 3：ASR 转写当作 B 直接训练

症状：只有音频转写，没有对应 A。  
结果：没有变换监督，还会学习 ASR 错字。  
修复：只在有授权、原稿和人工校对转写的情况下作为三联辅助数据。

### 错误 4：随机按段切分

症状：同一故事的前后段分别进入训练和测试。  
结果：评测被剧情泄漏污染。  
修复：始终按 `story_id` 分组切分。

### 错误 5：训练 B 时输出导演标签

症状：B 含角色、情绪、SFX/BGM 或 JSON。  
结果：与现有导演层职责重叠，接口不稳定。  
修复：B 只保留纯文本，结构化信息继续由导演层生成。

### 错误 6：只看 loss 或平均分

症状：平均指标很好，但有少量改凶手、漏线索或串角色。  
结果：最终音频不可交付。  
修复：硬错误一票否决，技术通过和 Gold 通过分开统计。

### 错误 7：训练期间与 TTS 抢显存

症状：随机 OOM、worker 崩溃、锁状态混乱。  
修复：训练离线独占 GPU，不删除有效锁，不在运行时安装依赖。

## 16. 完成检查表

- [ ] 已有 80～120 条封存 Gold 基线集。
- [ ] 已证明提示词基线仍有明确缺口。
- [ ] 至少 300 条已确认 A/B 配对，来自至少 20～30 个故事。
- [ ] 每条 user 明确包含 A，assistant 只包含 B。
- [ ] 故事级 train/validation/test 无交叉。
- [ ] 数据无 COT、JSON、导演标签和版权不明内容。
- [ ] smoke test 成功并能加载 adapter。
- [ ] 正式训练参数、版本和 hash 已记录。
- [ ] LoRA 在固定盲测中胜过基线，且无语义硬错误增长。
- [ ] B→C 的 Unitale Schema 回归 100% 通过。
- [ ] 固定 TTS 端到端成功率达到门槛。
- [ ] 合并/量化后的模型重新通过 Gold 测试。
- [ ] 生产保持离线、环境隔离、GPU 锁和现有 WebUI Schema。

## 17. 官方参考

- [Easy Dataset 官方仓库与安装说明](https://github.com/ConardLi/easy-dataset)
- [Easy Dataset LLaMA-Factory 导出实现](https://github.com/ConardLi/easy-dataset/blob/main/app/api/projects/%5BprojectId%5D/llamaFactory/generate/route.js)
- [Easy Dataset 数据库 Schema](https://github.com/ConardLi/easy-dataset/blob/main/prisma/schema.prisma)
- [Easy Dataset A 参与答案生成的实现](https://github.com/ConardLi/easy-dataset/blob/main/lib/services/datasets/index.js)
- [LLaMA-Factory 官方安装说明](https://github.com/hiyouga/LlamaFactory#installation)
- [LLaMA-Factory ShareGPT/OpenAI messages 格式](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md)
- [LLaMA-Factory LoRA/QLoRA/推理/合并命令](https://github.com/hiyouga/LlamaFactory/blob/main/examples/README.md)
- [LLaMA-Factory Qwen3.5 模板示例](https://github.com/hiyouga/LlamaFactory/blob/main/examples/ascend/qwen3_5_full_sft_fsdp2.yaml)
- [Qwen3.5-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)

