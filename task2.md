### P1：清理与最终接口约束冲突的兼容代码

  AGENTS.md:65 明确禁止保留旧接口兼容别名，但当前仍有 10 个 /internal/unload_all 路由，README 也将其标为兼容路由，例如
  README.md:24。这些路由对 one-shot worker 实际只执行空操作。

  此外：

  - MiMo 还有未列入最终接口的 /v1/voice-design/providers。
  - VoxCPM2 保留了未注册为路由的整套音色设计代码和独立 worker，见 voxcpm2/main.py:644 和 voxcpm2/main.py:747。
  - README 仍把 VoxCPM2 描述成“语音克隆、音色设计”，见 README.md:19。

  建议先确认 WebUI 没有调用，再删除旧路由、死代码、对应测试和文档。

  ### P1：收紧请求参数并修复 0 值语义

  Qwen 注释规定 max_chars_per_chunk=0 表示不分片，见 qwen3_tts/main.py:130，但 worker 使用 value or 120，会把 0 改为
  120，见 qwen3_tts/worker.py:374。多个 worker 的 pause_ms=0 也会被改回 250。

  同时，Qwen、MOSS、VoxCPM2、dots 和 MiMo 的若干生成参数没有合理上下限，可能导致超长任务、显存溢出或异常 500。

  建议：

  - 用显式 is None 处理合法的零值。
  - 为文本长度、token、steps、temperature、top-p、分片长度、停顿等补充 Pydantic 边界。
  - 清理没有实际作用的 save_as 字段，但实施前检查 WebUI 请求。
  - 对已知兼容字段显式建模，减少无限制的 extra="ignore"。

  ### P1：统一上传和共享存储实现

  目前 6 个上传入口都执行完整 await audio.read()，例如 qwen3_tts/main.py:696，没有大小限制、格式校验或流式写入。共享目
  录中的上传、sidecar 和引用映射使用直接覆盖写，不是原子操作，多个服务同时同步参考音频时存在竞争窗口。

  音色去重还会在每次上传时重新哈希所有 timbre WAV；控制面甚至用 read_bytes() 整文件读入，见 main/main.py:147。文件越
  多，上传耗时和内存占用越明显。

  建议提取一个轻量共享存储模块，统一实现：

  - 上传大小和允许格式限制；
  - 流式计算 SHA-256；
  - 临时文件加 os.replace 原子写入；
  - 基于 sidecar/index 的内容寻址去重；
  - 相对路径引用，避免存储目录迁移后 .path 失效；
  - 并发上传和失败回滚测试。

  ### P1：GPU 队列增加准入和可观测性

  共享 flock 能保证显存安全，但当前等待没有超时。无效请求也经常先等待 GPU 锁，再检查参考文件或构建 payload。

  建议保留“一请求一 worker”和共享锁，同时：

  - 在获取 GPU 锁前完成纯 CPU/文件校验；
  - 增加锁等待超时和明确的 429/503；
  - 记录排队时间、模型加载时间、推理时间、退出码和峰值显存；
  - 客户端断开后避免继续执行尚未开始的排队任务。

  部分 worker 设置了 expandable_segments，VoxCPM2、LongCat 和 dots 却主动删除该配置，且没有解释原因。此项应通过真实 GPU
  benchmark 决定，不能直接统一开启或关闭。

  ### P1：建立真正执行的质量门禁

  当前测试只验证 Ruff 配置存在，没有实际执行 formatter，因此 9/11 个项目已经出现格式漂移，例如 qwen3_tts/main.py:239。

  建议：

  - 先执行一次纯格式化修复。
  - 新增统一 check 脚本或 CI，实际运行 Ruff、锁检查和 85 项无模型测试。
  - 建立轻量 QA 环境，避免 CI 同步全部 CUDA/Torch 模型依赖。
  - 迁移掉当前 TestClient 的弃用警告。

  ### P2：文档、依赖和运行数据治理

  - 项目升级评估.md:46 仍写着“没有 pyproject、没有 tests”和旧端口，建议归档并加历史文档警告。
  - ACE-Step 的 Diffusers Git 依赖没有在 pyproject.toml 固定 commit，见 ace_step_1_5/pyproject.toml:12；虽然当前 lock
    固定了 commit，但重新生成锁时可能漂移。

  - README 明确说明输出文件不会自动清理，见 README.md:295。建议先增加磁盘占用/剩余空间健康指标，再提供默认关闭的保留期
    限或容量清理策略。