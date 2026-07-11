# Runtime Profile 替换式交接 POC

## Protocol 版本记录

- `protocol v1`：2026-07-10 的正式尝试在第二次 Pro 已返回、任何 Flash
  调用开始前失败。原因是本地已经执行 `evidence_quote <= 160` 的校验，但该
  长度限制没有写进 Pro compiler 指令。该次实际调用量为 2 Pro、0 Flash，
  没有产生任何评测译文。
- `protocol v2`：明确向 Pro 传达 `instruction <= 240`、
  `evidence_quote <= 160`、非空、总计至少一条、最短连续证据和输出前自检要求；
  同时增加“HTTP 响应先写 state，之后才校验”的渐进 checkpoint。v2 正式尝试
  发出了 2 个 Pro 网络请求：CURRENT 响应有效；Projection 的 HTTP 响应
  `content` 为空。实验在 Flash 前停止，共 0 Flash、0 评测译文。
- `protocol v3`：读取 v2 的 pre-Flash state，重新验证并复用已经通过的
  CURRENT Profile，只新调用 1 次 Projection Pro。HTTP 2xx 即使 content 为空，
  也先保存状态码、message keys、content/reasoning 长度、finish reason、usage
  和 latency；不保存真实 reasoning content。

v3 复用 CURRENT 只是在避免重复调用已经成功的编译步骤，不是依据翻译结果调参；
v2 没有产生评测译文。数据集、分组、决策门槛和 Flash 请求均保持不变。

本实验验证一个边界明确的问题：启动时的 Pro 能否只基于用户已经确认的
Prompt 与 AI 优化内容，生成更紧凑的 Runtime Profile，并以“替换”而非
“追加”的方式交给 Flash。

这不是新知识层。Policy、Knowledge Reference 和反馈记忆继续由现有生产通道
管理；Pro 不能创建术语、固定译法、例句或用户没有表达的偏好。

## 两组对照

- `CURRENT_RUNTIME`：严格复现当前生产启动交接。Flash 获得完整生产 system
  prompt、Pro 当前格式清单和生产 `_wrap_source_text()` 请求；格式清单既保留为
  启动 assistant 消息，也通过当前 OCR user message 的 `<RULE_CHECKLIST>` 注入。
- `PRO_RUNTIME_PROJECTION`：Flash 不再获得完整 Compiled Prompt 和格式清单，
  只获得 `DEFAULT_BASE_PROMPT`、最小 Policy JSON、经证据校验的
  `<AGENT_PROFILE>`、语言方向、相同运行时工具规则及相同当前请求包装。

两组使用相同的语言对、Policy、Reference、反馈记忆、ReferencePlan、技术词
提取、Normalizer、Validator、Flash 模型、temperature、max_tokens 和 OCR 原文。

## 信任边界

Projection Pro 只能看到：

- 中文到日本语的语言方向；
- Compiled Prompt 中按固定标题提取的 User Constraint Layer；
- 按固定标题提取的 AI Optimization Layer；
- 去掉 `source_text` 的当前有效 Policy；
- Reference 和确认反馈会在每次请求按需注入的说明。

`<ACTIVE_POLICY>` 只保留本地白名单中的结构化规则，不包含 `model_instruction`
或 `source_text`。它是用户确认的权威输出契约；Profile 只能提供语义与风格指导，
不能覆盖 Policy。

每条生成指令必须附带原层中的连续原文证据。三个列表合计至少一条。代码会拒绝错误层、虚假证据、
额外 Schema 字段、术语表、映射、固定读法、例句、测试句、翻译结果、思维链、
Markdown 表格、评测句泄漏、超过 1800 字符的渲染结果，以及任何本应留在
Policy 的字符集、平假名、括号、空格、标点、分词或词边界要求。证据只写入 state 审计，
Flash 只收到 instruction。

正式模式先校验 `--reuse-current-profile-state` 指定的 CURRENT checkpoint，任何
Prompt、Policy、Reference、模型、Profile 内容或评测泄漏不一致都会在网络前失败。
随后从 `status=running` 的新 state 开始。Projection Pro HTTP 成功返回后，程序先
保存 payload、raw response、usage、finish reason、latency 和
`validation_status=pending`，落盘后才执行 validator。成功改为 `passed`；失败
记录阶段、组别、原异常和真实 Pro/Flash 调用数，然后原样抛出，不重试。完整
成功后 state 才变为 `completed`。

在构造 48 个 Flash job 前，每个 case 只执行一次 Reference 匹配和 Feedback
Memory 检索，并冻结 CURRENT/Projection 两份 user content、RuntimeCaseContext 和
命中 ID。之后两组及重复轮次只能读取缓存。

## 冻结数据与调用量

数据集：`poc/data/agent_runtime_projection_dataset.json`

SHA256：

```text
d0f9ca8cfd9a72793b10d4fd089c3c6b9485bc60108a539d5872279afe37154b
```

共 12 条全新中文句子：8 条 runtime regression、4 条普通 control。它们与
`semantic_profile_dataset.json`、`agent_strategy_transfer_dataset.json` 和
`semantic_guard_dataset.json` 不存在完全相同原文，也不会进入两个 Pro payload。

正式实验固定为：

- 复用 1 个已校验 CURRENT Pro Profile；
- 新调用 1 次 Projection Pro；
- 12 cases × 2 groups × 2 repetitions = 48 次 Flash；
- temperature=0.0；
- 0 次 Pro judge；
- 0 次 retry；
- 固定随机种子打乱调用顺序。

## 本轮只允许离线运行

```powershell
python -m unittest tests.test_poc_agent_runtime_projection
python -m poc.poc_agent_runtime_projection --dry-run --reuse-current-profile-state poc/results/runtime-projection-state-20260710-190955.json
python -m unittest discover -s tests
python -m compileall -q app poc tests
```

不要在本轮执行无参数命令。无参数模式预留给之后由用户明确安排的正式 API
实验。

## 正式输出预留

- `poc/results/runtime-projection-YYYYMMDD-HHMMSS.jsonl`
- `poc/results/runtime-projection-state-YYYYMMDD-HHMMSS.json`
- `poc/results/runtime-projection-blind-YYYYMMDD-HHMMSS.jsonl`

盲审文件隐藏组别和重复次数，只保留原文、审查重点、译文和人工评分字段。

## 预声明决策门槛

只有 `PRO_RUNTIME_PROJECTION` 同时满足以下条件才考虑集成：

- 盲评 semantic fidelity 至少提高 0.4；
- naturalness 至少提高 0.3；
- runtime regression 的重大语义错误至少下降 30%；
- 生产格式通过数最多下降 1 条；
- Flash P95 延迟增幅不超过 15%；
- 平均 estimated input tokens 至少下降 20%；
- Profile 没有任何无证据指令、术语映射或评测句泄漏。

否则不修改生产 Agent Profile，不把 POC Profile 写入真实 Session，后续转向
Flash 模型能力或用户主动优化路径。

`--dry-run` 只能证明实验设计、隔离和离线执行链正确，不能证明翻译质量。
