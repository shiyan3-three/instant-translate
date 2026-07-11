# Semantic Guard POC

本轮只回答一个问题：在 Flash 翻译链路上，相比当前 profile，静态语义守卫与
运行时检索的语义策略哪一种更能减少重大语义错误和未完成句，同时不破坏硬格式
通过率和延迟。

三组使用运行时解析出的当前生产 prompt、policy、Profile digest、Flash、原始
OCR 文本和生产 OutputNormalizer/OutputValidator，唯一变量是 memory_hints 中的
语义约束。脚本不会读取或写入真实 Agent Session。

## 本 POC 验证什么

- 静态语义守卫能否在不引入检索开销的前提下降低重大语义错误。
- 运行时检索的语义策略能否在未见过的新句上进一步提升语义忠实度。
- 两种方式是否会在硬格式通过率或 P95 延迟上带来不可接受的回退。
- 当两者仍出现明显未完成句时，是否应停止继续堆叠提示，改为轻量语义检查或
  定向路由到 Pro。

## 三组对照

- `CURRENT_PROFILE`：当前生产 profile，不注入额外语义守卫，作为基线。
- `STATIC_SEMANTIC_GUARD`：通过生产 `_wrap_source_text()` 的 memory_hints
  通道注入固定语义检查规则，对所有句子生效，不依赖检索。
- `RETRIEVED_SEMANTIC_POLICY`：Pro 启动编译的语义策略按中文触发词检索，
  运行时在固定守卫之外最多注入三条相关规则；无命中时与静态守卫组相同。

## 数据隔离

- Pro 只能看到 `training_incidents`，即已确认的训练期事件。
- held-out 与 control 来源永不进入 Pro 载荷，避免提前写答案或污染泛化评估。
- seen_regression 的成功不能替代 held-out 的泛化成功。
- 正式数据固定为 3 条真实回归、6 条对应 held-out 和 3 条过度约束 control；
  测试会校验完整原文列表，不能替换成更容易的普通句。
- 纯数字、日期、范围和小数不是技术术语，fixture 与运行输出均不得为它们添加
  方括号。

## 调用与生产边界

- 1 次 Pro 生成与当前 `TranslationAgent.digest_rules()` 等价的 visible Profile。
- 1 次 Pro 仅根据三条 training incidents 编译 typed semantic policy。
- 12 条 × 3 组 × 2 次重复，共 72 次 thinking-disabled Flash。
- 每条 Flash 只调用一次；即使 production validator 失败也不触发 Thinking retry。
- 不调用 Pro judge，机器 anchor 和句尾检查只作诊断，不宣称是语义评分。

## 正式运行

```powershell
.\.venv\Scripts\python.exe -m poc.poc_semantic_guard
```

运行者只执行一次并原样返回完整终端输出以及 `Results`、`Policy state`、
`Blind review` 三个路径；不得修改文件、评分、分析、重试、修复或执行 Git
操作。

## 离线干跑

```powershell
.\.venv\Scripts\python.exe -m poc.poc_semantic_guard --dry-run
```

干跑会生成 72 条生产形状的请求与结果记录，但使用人工 fixture 响应且不会调用
API；干跑汇总不能作为翻译质量证据。

## 盲评

盲评文件隐藏组名和重复序号。逐条填写 `semantic_fidelity_0_to_5` 与
`naturalness_0_to_5`，并记录重大语义错误数和未完成句数后运行：

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_bootstrap_review `
  --groups CURRENT_PROFILE,STATIC_SEMANTIC_GUARD,RETRIEVED_SEMANTIC_POLICY `
  --results poc\results\semantic-guard-YYYYMMDD-HHMMSS.jsonl `
  --review poc\results\semantic-guard-blind-YYYYMMDD-HHMMSS.jsonl
```

## 预先约定的决策规则

- 若 `STATIC_SEMANTIC_GUARD` 与 `RETRIEVED_SEMANTIC_POLICY` 的语义分差距
  ≤ 0.25，且重大错误数相同，则优先采用静态守卫。
- `RETRIEVED_SEMANTIC_POLICY` 仅在同时满足以下条件时才考虑进入生产：
  - held-out 语义均值比静态守卫高出 ≥ 0.4；
  - 重大错误数与未完成句数更少；
  - 硬格式通过率至多下降 1；
  - P95 延迟增幅 ≤ 30%。
- 若两者仍出现明显未完成句，不再继续堆叠提示；改为验证轻量语义检查或定向
  路由到 Pro。
- seen_regression 的成功不能替代 held-out 的泛化成功。
