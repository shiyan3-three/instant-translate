# POC 6：编译后的持久 Agent 记忆

本 POC 验证产品化 Agent 路线，而不是再次测试“Pro 复述规则”。

启动时，Pro 读取完整用户规则、确认示例和 200 条引用，生成本地持久记忆：

- 决策优先级
- 可执行翻译步骤
- 否定、条件、数量、语气和软件关系的语义策略
- OCR 决策策略
- 失败预防规则
- 3～6 组启动练习示例

后续 Flash 以非思考模式运行，接收稳定记忆、可见示例会话、当前 OCR
以及本地检索到的引用。完整 200 条引用不会在每次翻译时重复发送，
`reasoning_content` 也不会保存或回放。

## 正式运行

复用 2026-06-30 已完成的 POC 5 Direct 基线，只新增 1 次 Pro 和 48 次 Flash：

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_memory `
  --baseline poc\results\session-replay-20260630-143244.jsonl
```

运行者只负责执行一次命令并返回终端原始输出及以下三个路径：

- `Results`
- `Agent memory`
- `Blind review`

运行者不得修改脚本、配置或结果，不得执行 Git 操作，不得评分、分析、
重试或修复。发生错误时只返回完整错误。

## API 调用边界

- Pro：1 次启动编译
- Flash：24 个案例 × 2 次重复 = 48 次
- Direct：复用旧结果，不重新调用

因此不能重复运行正式命令；失败时应先把错误交回开发者判断。

## 离线干跑

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_memory `
  --baseline poc\results\session-replay-20260630-143244.jsonl `
  --dry-run
```

干跑使用确定性记忆样例，生成全部 48 个 Agent 请求载荷，但不调用 API。

## 盲评

盲评文件混合旧 Direct 和新 Agent 结果，并隐藏组别和重复次数。评分者必须
具备足够的中日翻译判断能力，只能看到盲评文件，不能查看 Results。

全部 96 行评分完成后运行：

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_bootstrap_review `
  --groups DIRECT_BASELINE,AGENT_MEMORY `
  --results poc\results\agent-memory-YYYYMMDD-HHMMSS.jsonl `
  --review poc\results\agent-memory-blind-YYYYMMDD-HHMMSS.jsonl
```

## 预先约定的通过标准

- Agent 语义忠实度均值至少比 Direct 高 `0.5/5`
- 自然度不得下降
- Agent 硬约束通过率至少 `80%`
- Agent 缺失预期术语不超过 `2`
- Agent 翻译延迟 P50 不超过 `3 秒`

未完成盲评前，机器约束数据不能替代整体翻译质量结论。
