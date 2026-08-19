# Python 工程规范

本仓库的模型服务是独立的 uv 项目，而非一个可发布的统一 Python
包。每个服务的顶层 `main.py` 与 `worker.py` 都需要以该服务目录为导入
根目录运行，因此保持 `[tool.uv] package = false`；`src/*/__init__.py`
仅用作元数据包标记，不能把这些 HTTP 服务误当作 PyPI 发布物。

## 可复现环境

- 所有服务固定使用 Python `3.12.13`。`.python-version` 与
  `requires-python` 必须保持一致。
- 运行时依赖保持精确版本，以避免 CUDA、Torch 和上游模型运行时的组合
  发生意外漂移；开发工具使用宽松下限，但由每个项目提交的 `uv.lock`
  锁定精确版本。
- 通用 Python 包从官方 PyPI 解析。CUDA wheel 和模型专用包仍必须通过
  `[tool.uv.sources]` 中显式声明的来源获取。`moss_voiceGenerator` 是部署
  例外：它依赖预先准备的 `/home/muyi086/tts-depency/MOSS-TTS` editable
  checkout，因此同步前必须确认该目录存在；不要让 uv 改为联网拉取它。

从仓库根目录同步全部服务：

```bash
for project in qwen3_tts mimo_tts voxcpm2 LongCat_AudioDiT_3.5B_bf16 \
  dots_tts_soar moss_soundEffect stable_audio_3_medium ace_step_1_5 \
  qwen3_voiceDesign moss_voiceGenerator Step_Audio_EditX; do
  uv sync --project "$project" --locked
done
```

修改任一 `pyproject.toml` 后，必须在相同项目目录运行 `uv lock`，提交
`pyproject.toml` 和 `uv.lock`，再用 `uv lock --check` 验证。

## 格式化与静态检查

每个服务都在 `pyproject.toml` 中配置同一套 Ruff 基线：`E`、`F`、`I`、
`UP`、`B`，目标版本 Python 3.12，行宽 100。`B008` 是 FastAPI
`File(...)`/`Form(...)` 路由签名的框架约定；`E501` 交给 Ruff formatter
处理，因为 URL、协议文本和框架签名不应为了机械折行而损失可读性。

```bash
for project in qwen3_tts mimo_tts voxcpm2 LongCat_AudioDiT_3.5B_bf16 \
  dots_tts_soar moss_soundEffect stable_audio_3_medium ace_step_1_5 \
  qwen3_voiceDesign moss_voiceGenerator Step_Audio_EditX; do
  (cd "$project" && uv run --locked ruff check . && uv run --locked ruff format --check .)
done

uvx ruff check main tests soundEffect
uvx ruff format --check main tests soundEffect
```

修复格式时使用 `ruff format .`；只使用 `ruff check --fix` 的安全修复，
并审阅每项行为相关改动。不要使用无规则号的 `noqa`；必要的例外必须写明
原因，并限制到最小文件或行范围。

## 日志、异常和 worker 边界

- HTTP 服务代码使用 `logging.getLogger(__name__)`；在把未知异常转换为
  `HTTPException` 的边界使用 `logger.exception(...)`，以保留 traceback。
- one-shot worker 的 stdout/stderr 是父进程提取错误摘要的协议边界，保留其
  错误输出；新增 worker 诊断应写到 stderr，不应污染成功 WAV 的输出路径。
- 异常转换必须保留原因链（`raise ... from exc`）。超时后先终止进程组，
  再抛出带原始 `TimeoutExpired` 原因的业务异常。
- `main.py` 只做 HTTP 校验、存储和 worker 生命周期管理；不得在其中导入
  或加载重型模型。`worker.py` 负责模型导入和推理，并始终清理临时文件、
  进程组与 GPU 锁。

## 测试

静态检查和单元测试必须不下载权重、不依赖 CUDA，也不调用真实 MiMo。
先运行根目录迁移测试，再运行含有专属测试的服务：

```bash
bash -n start.sh
uv run --project qwen3_tts python -m unittest discover -s tests -v
(cd stable_audio_3_medium && uv run --project . python -m unittest discover -s tests -v)
(cd ace_step_1_5 && uv run --project . python -m unittest discover -s tests -v)
```
