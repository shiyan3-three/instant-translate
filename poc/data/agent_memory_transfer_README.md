# 最终方向 POC：反馈记忆迁移

这轮只验证一个问题：用户纠正过一次之后，Flash 能否把纠正迁移到一条
从未见过但相似的新句子。

它不再测试占位符、全量术语、词性标签、多模型投票或额外语义框架。
三组使用完全相同的 Flash、基础规则、20 条基础软件术语、原始 OCR 文本和
本地表面格式器，唯一变量是反馈记忆。

## 数据来源

共 8 组反馈迁移场景：

- 2 组来自本机 `FeedbackStore` 中用户已经确认的真实记忆
- 6 组来自前几轮 POC 已经观察到的明确失败，并标记为项目审查案例

每组包含：旧句、错误译文、纠正意见、接受译文，以及一条隐藏的新句。
Pro 启动编译时只能看到前四项，看不到新句和机器期望，因此不能提前写答案。

数据文件：`poc/data/agent_memory_transfer_dataset.json`

## 三组对照

- `STATELESS`：只给新句，不提供历史或记忆。
- `RAW_SESSION`：保留旧句、错误回答、用户纠正和接受译文的原始会话历史。
- `COMPILED_MEMORY`：Pro 启动时将全部确认反馈编译成规则，运行时按触发词检索。

每组 8 个场景 × 2 次重复，共 48 次 Flash；另有 1 次 Pro 启动编译。

## 正式运行

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_memory_transfer
```

运行者只能执行一次，并返回完整终端输出以及：

- `Results`
- `Compiled memory`
- `Blind review`

不得修改文件、执行 Git、评分、分析、修复或自动重试。

## 离线干跑

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_memory_transfer --dry-run
```

干跑会生成 48 个精确请求载荷，但不会调用 API。

## 盲评

正式运行后的盲评文件包含 48 行，隐藏组别、重复次数和记忆命中信息。
评分者需要同时评价：

- `semantic_fidelity_0_to_5`
- `naturalness_0_to_5`
- `preference_transfer_0_to_5`

全部评分持久化后运行：

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_memory_transfer_review `
  --results poc\results\memory-transfer-YYYYMMDD-HHMMSS.jsonl `
  --review poc\results\memory-transfer-blind-YYYYMMDD-HHMMSS.jsonl
```

## 预先约定的架构决策

- `COMPILED_MEMORY` 的偏好迁移比 `STATELESS` 至少高 `0.75/5`，且语义/自然度
  相对 `RAW_SESSION` 不下降超过 `0.25`：采用 Pro 编译的本地记忆。
- `RAW_SESSION` 明显胜出：采用持久会话历史与相关历史回放，不增加 Pro 编译层。
- 两者都未明显胜过 `STATELESS`：Flash 无法可靠继承这些偏好，复杂/反馈相关任务
  路由给 Pro。

无论哪组获胜，本地 Session 都采用事件日志作为事实源；摘要或编译记忆只是可重建
缓存，不能替代原始用户反馈。
