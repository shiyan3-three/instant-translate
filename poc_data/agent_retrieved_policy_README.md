# Retrieved Typed Policy — Final Decision POC

这是进入主程序前的最终架构判断。本轮复用上一轮已经完成的 `RAW_BRIEF` 与
`PRO_PLAYBOOK` 共48条结果，只新增 `RETRIEVED_POLICY` 的24次 Flash 调用，不再调用
Pro。

`RETRIEVED_POLICY` 的处理方式：

1. 从已保存的 Pro playbook 中只保留用户启动说明明确提到的中文歧义词。
2. 删除 Pro 自创示例、通用句式和质量检查，避免全局污染。
3. 保存为本地结构化策略。
4. 每条 OCR 按中文触发词检索，最多向 Flash 注入3条相关规则。
5. 没有命中规则的普通句子使用与 `RAW_BRIEF` 相同的确认上下文。

## 正式运行

```powershell
.\.venv\Scripts\python.exe poc_agent_retrieved_policy.py
```

脚本自动读取 `logs/poc` 中最新的正式 `agent-strategy-*.jsonl` 基线。也可以显式指定：

```powershell
.\.venv\Scripts\python.exe poc_agent_retrieved_policy.py `
  --baseline logs\poc\agent-strategy-20260702-212448.jsonl
```

运行者只执行命令，返回完整终端输出以及 `Results`、`Policy state`、
`Combined blind review` 三个路径；不要修改文件或评价译文。

## 离线验证

```powershell
.\.venv\Scripts\python.exe poc_agent_retrieved_policy.py --dry-run
```

## 最终盲评

联合盲评文件包含三组共72条输出，组名和重复序号均隐藏。由具备日语判断能力的独立
评审填写两个0～5分字段后运行：

```powershell
.\.venv\Scripts\python.exe poc_agent_bootstrap_review.py `
  --groups RAW_BRIEF,PRO_PLAYBOOK,RETRIEVED_POLICY `
  --results logs\poc\agent-retrieved-policy-YYYYMMDD-HHMMSS.jsonl `
  --review logs\poc\agent-retrieved-policy-blind-YYYYMMDD-HHMMSS.jsonl
```

决策之后不再继续堆 POC：若检索策略获胜，则直接集成本地 Session、相关规则检索、
后台 Pro 刷新和快速/思考模式共享历史；若原始说明获胜，则保留相关原始反馈回放，
Pro 只用于用户主动思考或提出待确认的优化建议。
