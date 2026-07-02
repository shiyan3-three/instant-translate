# POC 6.1：带语义类型的引用槽 + 本地安全格式化

本 canary 针对 POC 6 暴露的两个问题：

1. `⟦REF_n⟧` 不携带原词含义和词性，Flash 无法稳定组织自然句法。
2. 双空格、标点和普通词误加括号属于机械格式问题，不应继续依赖模型自行检查。

## 三组对照

- `DIRECT_BASELINE`：复用 POC 6 中对应的旧 Direct 输出。
- `AGENT_V6_NORMALIZED`：复用旧 Agent 输出，只应用新的本地安全格式化。
- `AGENT_V61_TYPED`：新的带语义类型引用槽 + 相同本地安全格式化。

因此可以区分改善来自格式器，还是来自引用槽带来的语义/句法信息。

## 测试范围

只选择 8 个困难案例，每个重复两次：

- 条件关系
- 药物数量和频率
- 数据库部署关系
- 时间信息完整性
- 推荐语气与标点
- 软件部署条件句
- OCR 变量纠错
- OCR 数据库/服务器纠错

正式运行复用已有 Agent memory，不调用 Pro，只新增 16 次 Flash。

## 正式运行

```powershell
.\.venv\Scripts\python.exe poc_agent_memory_canary.py `
  --baseline logs\poc\agent-memory-20260701-185140.jsonl `
  --memory logs\poc\agent-memory-state-20260701-185140.json
```

运行者只执行一次并原样返回完整终端输出、`Results` 和 `Blind review` 路径。
不得修改文件、评分、分析、重试、修复或执行 Git 操作。

## 本地安全格式化边界

允许的自动处理只有：

- 恢复本地引用槽
- 将非授权括号拆为普通平假名，同时保留词本身
- 删除纯标点
- 将所有空白统一为两个 ASCII 空格并去除首尾空白

不会自动改写词语、数量、助词、时态或句法，因此不能伪造语义提升。

严格评估器额外检查括号是否平衡、括号词是否来自本次授权引用。

## 离线干跑

```powershell
.\.venv\Scripts\python.exe poc_agent_memory_canary.py `
  --baseline logs\poc\agent-memory-20260701-185140.jsonl `
  --memory logs\poc\agent-memory-state-20260701-185140.json `
  --dry-run
```

## 盲评

正式运行后盲评文件应包含 48 行，并隐藏组别、重复次数和格式化信息。

```powershell
.\.venv\Scripts\python.exe poc_agent_bootstrap_review.py `
  --groups DIRECT_BASELINE,AGENT_V6_NORMALIZED,AGENT_V61_TYPED `
  --results logs\poc\agent-memory-canary-YYYYMMDD-HHMMSS.jsonl `
  --review logs\poc\agent-memory-canary-blind-YYYYMMDD-HHMMSS.jsonl
```

## 通过标准

- Typed 组严格硬约束至少通过 `13/16`
- Typed 组术语遗漏为 `0`
- Typed 组 P50 不超过 `3 秒`
- `terminology_04` 和 `ocr_noise_04` 的句法/语义优于旧 Agent
- 盲评语义忠实度和自然度均优于 `AGENT_V6_NORMALIZED`

盲评前只允许判断机械约束，不能宣称整体翻译质量提升。
