# Pro Strategy Transfer POC

本实验只回答一个问题：启动时由 `deepseek-v4-pro` 生成的**可见领域策略**，能否让
同一 Session 中关闭思考的 `deepseek-v4-flash`，在未见过的新句子上比直接读取原始
用户说明翻译得更好。

两组 Flash 获得完全相同的 system prompt、原始启动说明、模型、测试句与参数：

- `RAW_BRIEF`：启动说明后只有固定的普通确认语。
- `PRO_PLAYBOOK`：启动说明后保存 Pro 生成的领域语义、自然句式、检查项和自创示例。

12 条正式测试句不会发送给 Pro；每组重复两次。总调用量为 1 次 Pro + 48 次 Flash。
实验不保存或回放 `reasoning_content`，因为普通无工具调用的 DeepSeek 多轮请求不会把
上一轮思维链加入后续上下文；真正可迁移的是可见 `content`。

## 正式运行

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_strategy_transfer
```

运行者只需执行命令并返回终端完整输出以及 `Results`、`Session`、`Blind review` 三个
路径，不应修改文件或评价译文。

## 离线验证

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_strategy_transfer --dry-run
```

## 盲评

盲评文件隐藏组名和重复序号。逐条填写 `semantic_fidelity_0_to_5` 与
`naturalness_0_to_5` 后执行：

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_bootstrap_review `
  --groups RAW_BRIEF,PRO_PLAYBOOK `
  --results poc\results\agent-strategy-YYYYMMDD-HHMMSS.jsonl `
  --review poc\results\agent-strategy-blind-YYYYMMDD-HHMMSS.jsonl
```

自动 `diagnostic_anchor_pass` 仅检查少量可审计语义锚点，不代表整体翻译质量。是否采用
启动 Pro，必须以盲评的语义完整度、自然度和重大语义错误数为准。
