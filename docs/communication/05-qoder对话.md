# 05 - QoderWork Agent Bootstrap POC V2 结果

> 最后更新：2026-06-29
> 运行脚本：`poc_agent_bootstrap.py`（Version 2，single-owner term protocol）
> Review 脚本：`poc_agent_bootstrap_review.py`
> 模型：Bootstrap=deepseek-v4-pro(thinking), Translation=deepseek-v4-flash(thinking=disabled)
> 数据集：`poc/data/reference_poc_dataset.json`（status=approved, 200 条术语, 24 cases × 3 groups = 72 请求）
> Run ID：2026-06-29T13:56:09.431462+00:00

---

## 一、实验设计

三组 Flash 翻译，共享同一数据集和同一 Flash 模型，区别在于 Flash 能看到的上下文：

| 组 | system prompt | user 消息 | 说明 |
|---|---|---|---|
| CURRENT_FULL | 基础提示词 + 完整 200 条术语表 | 原文（占位符保护后） | 当前线上方案，token 最多 |
| RETRIEVAL_RAW | 基础提示词 + 全局规则 | 原文 + mandatory reference（仅命中术语） | POC 3 的 B2 方案 |
| AGENT_BOOTSTRAP | 基础提示词 + 全局规则 + Pro 编译的持久状态 | 原文 + mandatory reference（仅命中术语） | 新方案：Pro 理解一次，Flash 执行所有翻译 |

Bootstrap 阶段：Pro 调用一次（thinking=enabled），将完整 prompt/reference 包转为显式 JSON 状态。该状态编译为 system prompt 前缀，Flash 每次翻译都能看到。

### Version 2 改进：single-owner term protocol

Version 2 使用单一所有者术语协议：精确匹配的术语变为不透明占位符，其目标在本地恢复，Flash 永远不会同时看到占位符和其目标映射。OCR 不确定的情况最多接受两个编辑距离候选项作为非约束性提示。Pro 状态只包含意图/风格/OCR/风险理解增量；确认的硬规则保留在确定性提示层中，不由 Pro 重复。

旧版 V1 状态被拒绝，必须重新调用 Pro。

---

## 二、机器约束检查（非翻译质量评分）

机器约束检查只验证格式规则（括号、空格、占位符计数、禁止输出），不评价翻译质量。

| 组 | 完成 | 错误 | 机器约束通过 | 缺失术语 | Provider Prompt Tokens | P50 延迟 |
|---|---|---|---|---|---|---|
| CURRENT_FULL | 24 | 0 | 21/24 (87.5%) | 4 | 143,384 | 2.002s |
| RETRIEVAL_RAW | 24 | 0 | 20/24 (83.3%) | 3 | 6,727 | 1.797s |
| AGENT_BOOTSTRAP | 24 | 0 | 21/24 (87.5%) | 1 | 20,863 | 1.896s |

Bootstrap 调用成本：1 次 Pro 调用，47.7 秒，14,791 tokens（12,417 prompt + 2,374 completion，其中 1,778 reasoning tokens）。

AGENT_BOOTSTRAP 的缺失术语最少（1 个），远低于 CURRENT_FULL（4 个）和 RETRIEVAL_RAW（3 个）。Pro 的 ocr_guidance 和 risk_notes 帮助 Flash 更好地识别需要括号包裹的术语。

---

## 三、盲审方法

72 条翻译结果打乱顺序、移除组名后写入 blind-review JSONL。我在不知道组别归属的情况下逐条评分：

- `semantic_fidelity`（0-5）：原文语义忠实度
- `naturalness`（0-5）：日语自然度

评分后运行 `poc_agent_bootstrap_review.py` 验证所有分数完整、数值合法、范围正确，然后揭示组别归属并聚合统计。Review 脚本拒绝缺失、非数值、越界、重复或不匹配的分数。Review 脚本输出 `blind_review_summary` JSON，包含 `evidence`（验证证据）、`groups`（组级统计）和 `categories`（类别×组统计）。

---

## 四、盲审结果

### 组级统计

| 组 | 语义忠实度 均值 | 自然度 均值 | 合计 均值(0-10) | 满分(5+5) | 重大错误(语义≤2) |
|---|---|---|---|---|---|
| CURRENT_FULL | 4.375 | 3.542 | 7.917 | 2 | 1 |
| RETRIEVAL_RAW | 4.375 | 3.625 | **8.000** | **8** | 1 |
| AGENT_BOOTSTRAP | 4.417 | 3.250 | 7.667 | 2 | **0** |

RETRIEVAL_RAW 综合最优（8.0），且满分数量远超其他两组（8 vs 2）。AGENT_BOOTSTRAP 语义均值最高且零重大错误，但自然度最低，综合最差。CURRENT_FULL 居中。

### 按类别×组统计（语义/自然度/合计）

| 类别 | CURRENT_FULL | RETRIEVAL_RAW | AGENT_BOOTSTRAP |
|---|---|---|---|
| semantic | 4.5/3.5/8.0 | **5.0/4.75/9.75** | 4.75/2.75/7.5 |
| terminology | **5.0/4.0/9.0** | 4.5/3.25/7.75 | 4.5/3.5/8.0 |
| format | **4.5/3.75/8.25** | 3.75/3.0/6.75 | 3.75/3.0/6.75 |
| style | **5.0/4.0/9.0** | 4.25/4.25/8.5 | 4.75/3.5/8.25 |
| forbidden | 3.5/2.75/6.25 | 5.0/3.5/8.5 | **5.0/4.0/9.0** |
| ocr_noise | **3.75/3.25/7.0** | 3.75/3.0/6.75 | 3.75/2.75/6.5 |

### 逐案例对比（语义/自然度）

| 案例 | CURRENT_FULL | RETRIEVAL_RAW | AGENT_BOOTSTRAP |
|---|---|---|---|
| semantic_01 | 5/4 | 5/5 | 5/3 |
| semantic_02 | 5/4 | 5/5 | 5/2 |
| semantic_03 | 3/2 | 5/5 | 4/2 |
| semantic_04 | 5/4 | 5/4 | 5/4 |
| terminology_01 | 5/4 | 5/4 | 5/4 |
| terminology_02 | 5/4 | 5/3 | 5/4 |
| terminology_03 | 5/4 | 5/3 | 5/4 |
| terminology_04 | 5/4 | 3/3 | 3/2 |
| format_01 | 5/5 | 5/5 | 5/5 |
| format_02 | 5/4 | 4/3 | 3/2 |
| format_03 | 4/3 | 3/2 | 4/3 |
| format_04 | 4/3 | 3/2 | 3/2 |
| style_01 | 5/4 | 5/5 | 5/3 |
| style_02 | 5/3 | 5/5 | 5/3 |
| style_03 | 5/5 | 5/5 | 4/3 |
| style_04 | 5/4 | 2/2 | 5/5 |
| forbidden_01 | 4/4 | 5/3 | 5/4 |
| forbidden_02 | 3/2 | 5/4 | 5/4 |
| forbidden_03 | 4/2 | 5/4 | 5/4 |
| forbidden_04 | 3/3 | 5/3 | 5/4 |
| ocr_noise_01 | 5/4 | 5/5 | 5/3 |
| ocr_noise_02 | 4/3 | 3/2 | 3/2 |
| ocr_noise_03 | 1/2 | 4/3 | 4/3 |
| ocr_noise_04 | 5/4 | 3/2 | 3/3 |

---

## 五、Pro Bootstrap 持久状态

Pro 调用一次（thinking=enabled），将完整 prompt/reference 包转为显式 JSON 状态。状态包含：

- `intent_summary`：全平假名日语翻译，软件术语用括号包裹，保持原意和完整句子
- `semantic_priorities`：保留全部命题内容（否定、时态、体、语气）；区分术语化和一般含义；保留日语助词和句末元素
- `style_guidance`：按自然语法块分段；使用自然日语语序（SOV）；保持适当的礼貌/普通体
- `ocr_guidance`：仅在意图明确时纠正 OCR 错误；翻译可用片段不虚构缺失部分；混合脚本转平假名
- `risk_notes`：高歧义词需上下文判断；间距不应破坏搭配；OCR 缺失标点可能导致平淡语调；最长匹配优先避免重复括号

状态文件可复用：`poc_agent_bootstrap.py --state poc/results/agent-bootstrap-state-20260629-215521.json`

---

## 六、关键发现

### 发现 1：RETRIEVAL_RAW 综合最优，满分数远超其他组

RETRIEVAL_RAW 综合得分 8.0（0-10），高于 CURRENT_FULL（7.917）和 AGENT_BOOTSTRAP（7.667）。更显著的是满分数量：RETRIEVAL_RAW 有 8 个满分（5+5），而 CURRENT_FULL 和 AGENT_BOOTSTRAP 各只有 2 个。这意味着 B2 方案在更多案例中同时达到了最高语义忠实度和最高自然度。

### 发现 2：AGENT_BOOTSTRAP 零重大错误但天花板低

Version 2 的 single-owner term protocol 显著改善了 AGENT_BOOTSTRAP 的稳定性：重大错误从 V1 的 7/24 降为 0/24。但满分数也从 V1 的 10/24 降为 2/24。Pro 状态稳定了输出质量（消灭了灾难性翻译），但也压低了上限（减少了完美翻译）。这是一种"安全但平庸"的权衡。

### 发现 3：RETRIEVAL_RAW 在 semantic 类别中压倒性领先

semantic 类别：RETRIEVAL_RAW 语义 5.0、自然度 4.75（合计 9.75），3 个满分。远超 CURRENT_FULL（4.5/3.5=8.0）和 AGENT_BOOTSTRAP（4.75/2.75=7.5）。B2 方案的 mandatory reference + 占位符保护机制让 Flash 在翻译普通文本时既忠实又自然。

### 发现 4：AGENT_BOOTSTRAP 在 forbidden 类别中最优

forbidden 类别：AGENT_BOOTSTRAP 语义 5.0、自然度 4.0（合计 9.0），优于 RETRIEVAL_RAW（5.0/3.5=8.5）和 CURRENT_FULL（3.5/2.75=6.25）。Pro 的 risk_notes 帮助 Flash 区分软件术语和一般含义，正确判断何时使用括号。CURRENT_FULL 在此类别最差，因为完整术语表在 system prompt 中不如 user 消息中的 mandatory reference 有强制性。

### 发现 5：CURRENT_FULL 在 terminology 和 style 中最优

terminology 类别：CURRENT_FULL 语义 5.0、自然度 4.0（合计 9.0）。完整 200 条术语表在 system prompt 中提供了最佳术语映射。style 类别：CURRENT_FULL 同样最优（5.0/4.0=9.0），完整术语表让模型在风格翻译中不受术语不确定性影响。

### 发现 6：format 双空格规则在所有组中普遍失败

format_02、format_03、format_04 在所有三组中都未通过机器约束检查（required_pattern `\] {2}\[` 从不匹配）。模型不遵守"恰好两个空格"规则。这不是组间差异，而是 prompt 合规问题，需要 few-shot 或后处理解决。

### 发现 7：OCR 噪声处理各组差异不大但失败模式不同

ocr_noise 类别：三组合计在 6.5-7.0 之间，差异不大。但失败模式不同：

- CURRENT_FULL 的重大错误在 ocr_noise_03：将"变重"（OCR 错字，应为"变量"）误读为"おもみ"（重量），完全错误
- RETRIEVAL_RAW 和 AGENT_BOOTSTRAP 正确推断"变重"→"变量"（へんすう），但未将 へんすう 括号包裹
- ocr_noise_04 中"数捗库"（应为"数据库"）和"服器"（应为"服务器"）：CURRENT_FULL 有一个条目正确推断并使用括号，但另一个条目意义扭曲；RETRIEVAL_RAW 和 AGENT_BOOTSTRAP 同样有推断成功但括号缺失或意义扭曲的问题

### 发现 8：AGENT_BOOTSTRAP 的 Token 成本是 RETRIEVAL_RAW 的 3.1 倍

AGENT_BOOTSTRAP 每次 Flash 请求平均约 869 estimated input tokens（含 Pro 编译的 system 前缀），RETRIEVAL_RAW 只有 301.8。Provider prompt tokens 总量 20,863 vs 6,727。加上 Bootstrap 的一次性 Pro 调用成本（14,791 tokens, 47.7s），AGENT_BOOTSTRAP 的总 token 成本约为 RETRIEVAL_RAW 的 5.3 倍，但翻译质量更差（7.667 vs 8.0）。

### 发现 9：Version 2 vs Version 1 对比

| 指标 | V1 AGENT_BOOTSTRAP | V2 AGENT_BOOTSTRAP | 变化 |
|---|---|---|---|
| 语义均值 | 3.67 | 4.417 | +0.75 |
| 自然度均值 | 3.58 | 3.25 | -0.33 |
| 合计均值 | 7.25 | 7.667 | +0.42 |
| 满分数 | 10/24 | 2/24 | -8 |
| 重大错误 | 7/24 | 0/24 | -7 |

| 指标 | V1 RETRIEVAL_RAW | V2 RETRIEVAL_RAW | 变化 |
|---|---|---|---|
| 语义均值 | 3.83 | 4.375 | +0.55 |
| 自然度均值 | 3.38 | 3.625 | +0.25 |
| 合计均值 | 7.21 | 8.0 | +0.79 |
| 满分数 | 6/24 | 8/24 | +2 |
| 重大错误 | 3/24 | 1/24 | -2 |

Version 2 的 single-owner term protocol 对两组都有改善：RETRIEVAL_RAW 全面提升（合计 +0.79，满分数 +2，重大错误 -2），AGENT_BOOTSTRAP 提升了稳定性（重大错误 7→0）但降低了天花板（满分数 10→2）。

---

## 七、Agent Bootstrap 假设验证结论

假设：Pro 理解一次完整 prompt/reference 包，Flash 执行所有翻译。

**不推荐作为生产方案，RETRIEVAL_RAW（B2）仍为最优选择。**

成立的方面：
- Pro 编译的状态消灭了所有重大错误（0/24），证明 Pro 的理解确实能稳定 Flash 输出
- Pro 的 risk_notes 在 forbidden 类别中有效，帮助 Flash 区分软件术语和一般含义
- Bootstrap 状态可持久化复用，不需要每次翻译都调用 Pro
- AGENT_BOOTSTRAP 缺失术语最少（1 个 vs 3-4 个），Pro 的 ocr_guidance 帮助识别术语

不成立的方面：
- AGENT_BOOTSTRAP 综合得分最低（7.667 vs RETRIEVAL_RAW 8.0），满分数最少（2 vs 8）
- Pro 状态压低了上限——虽然消灭了灾难，但也减少了完美翻译
- 自然度最低（3.25 vs 3.625），Pro 的 style_guidance 没有提升日语自然度
- Token 成本是 RETRIEVAL_RAW 的 3.1 倍（不含 Bootstrap），加上 Bootstrap 后约 5.3 倍
- 在 semantic 类别中远逊于 RETRIEVAL_RAW（7.5 vs 9.75），Pro 状态反而干扰了普通文本翻译

**推荐方案仍为 RETRIEVAL_RAW（B2）**：在 V1 和 V2 中均表现出最佳的综合质量和成本效益比。V2 的 single-owner term protocol 进一步提升了 B2 的表现（合计 7.21→8.0，满分数 6→8，重大错误 3→1）。AGENT_BOOTSTRAP 的稳定性优势（0 重大错误）可以通过在 B2 的 user 消息中加入 Pro 的 risk_notes 来近似，而不需要引入 Pro bootstrap 的复杂性和成本。

---

## 八、输出文件

| 文件 | 说明 |
|---|---|
| `poc/results/agent-bootstrap-20260629-215521.jsonl` | 完整请求/响应记录（1 metadata + 1 bootstrap + 72 results + 1 summary） |
| `poc/results/agent-bootstrap-state-20260629-215521.json` | Pro 编译的持久状态（可复用） |
| `poc/results/agent-bootstrap-blind-20260629-215521.jsonl` | 盲审文件（72 条，打乱顺序，无组名，已填入评分） |
| `poc/results/agent-bootstrap-review-summary-20260629-215521.json` | Review 脚本输出的验证后统计摘要 |

盲审文件中的 `semantic_fidelity_0_to_5` 和 `naturalness_0_to_5` 字段已逐条填写并通过 `poc_agent_bootstrap_review.py` 验证。组别归属在本文档第四节揭示。
