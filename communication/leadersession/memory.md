# 架构师 AI 跨会话记忆

> 长期保存的关键决策和经验，跨会话有效

---

## 🎯 项目核心信息

**项目**：instant-translate
**类型**：Windows 桌面即时翻译悬浮窗工具
**技术栈**：Python 3.9 + PySide6 + PaddleOCR 2.7.3 + OpenAI 兼容 API
**仓库**：https://github.com/shiyan3-three/instant-translate

**核心功能**：
1. 鼠标框选屏幕区域
2. 每 500ms 截图 + BLAKE2b 变化检测
3. OCR 识别文字变化
4. AI 翻译
5. 悬浮窗展示结果
6. 支持 3 组独立的框选+翻译工作流

---

## 🧠 关键技术决策（已确认）

### 1. Agent 化改造方向
- **决策**：翻译管线从无状态管道升级为有记忆的 Agent
- **原因**：每次都重发 200-1600 字规则，模型重复消化，慢且浪费
- **方案**：规则消化一次（thinking 模型，5-10s），后续翻译只发原文（秒回）
- **前置条件**：DeepSeek thinking 切换 POC 必须通过
- **来源**：`plans/02-Agent化改造方案.md`、`plans/04-决策记录.md`

### 2. 穿透用 WS_EX_TRANSPARENT
- **决策**：选择框穿透用 Win32 `SetWindowLongW` 操作 `WS_EX_TRANSPARENT`
- **原因**：setMask、WindowTransparentForInput 都不够稳定
- **来源**：`plans/04-决策记录.md` 2026-06-02

### 3. PaddleOCR 降级到 2.7.3
- **决策**：PaddleOCR 3.x → 2.7.3，PaddlePaddle 3.3.1 → 2.6.2
- **原因**：3.x API 不兼容 + Windows Python 3.9 有 bug
- **来源**：`plans/04-决策记录.md` 2026-06-02

### 4. 翻译模板合并三个功能
- **决策**：侧边栏四页→三页，"翻译模板"合并原"翻译方向"+"提示词"+"会话管理"
- **原因**：语言方向、约束层、引用层、上下文是同一翻译场景的不同维度
- **来源**：`plans/04-决策记录.md` 2026-06-03

### 5. 上下文溢出：本地存储 + compact
- **决策**：上下文存本地 JSON，窗口快满时 compact 旧条目为摘要
- **原因**：纯内存重启丢失，纯 API token 爆炸
- **关键**：compact 是纯客户端操作（术语 KV + 最近 N 轮 + 文本拼接），不调 API
- **来源**：`plans/04-决策记录.md` 2026-06-03

---

## ⚠️ 已知风险和坑

### 1. thinking 切换可靠性（未验证）
- **风险**：Agent 方案完全依赖"开 thinking 消化规则 → 关 thinking 快速翻译"
- **坑**：DeepSeek API 可能不支持同一 messages 数组混用 thinking enabled/disabled
- **必须测**：
  - reasoning_content 是否能被后续非 thinking 轮读到
  - thinking tokens 计费（成本是否可接受）
  - KV cache 命中率（thinking 切换是否导致 cache miss）
- **状态**：❌ 未验证，已安排 POC（`task.md` 任务 4）

### 2. PaddleOCR 大段文本卡死（未定位）
- **现象**：选择框内文字少时正常，文字多时运行一段时间后 OCR 不更新
- **可能原因**：
  - PaddleOCR 内部状态累积
  - 去重逻辑误判（大段文本中少量变化，trigram 相似度仍高）
- **定位方法**：入口/出口加时间戳日志，区分是 PaddleOCR 慢还是去重误杀
- **状态**：🟡 已安排诊断（`task.md` 任务 2）

### 3. 编辑模式闪烁（已确认根因）
- **现象**：编辑模式下 badge/工具栏不断闪烁，按钮点不到
- **根因**：每 500ms 轮询触发 `_upsert_selection_box` → `apply_model` → show/hide
- **修复**：状态比对 + 脏标记（只在真正变化时才重绘）
- **状态**：🔴 已安排修复（`task.md` 任务 1）

---

## 📝 经验和教训（来自 2026-06-10 会话）

### 我犯过的错误

1. **compact 压缩想复杂了**
   - 我说"压缩需要调 API"
   - 实际：纯客户端操作就够了（术语 KV + 最近 N 轮 + 文本拼接）
   - 教训：不要过度设计

2. **"不关 thinking"方案没抓住核心矛盾**
   - 我提议"不关 thinking，在 prompt 里注入提醒"
   - 问题：这不能解决"慢"的核心问题
   - 教训：保守方案也要解决核心矛盾，否则没意义

3. **两个独立 Agent 是多余的**
   - 我提议"一个消化规则，一个执行翻译"
   - 问题：同一会话的两个阶段，不需要拆开
   - 教训：过度设计 = 增加复杂度 = 引入新 bug

### 正确的判断

1. **thinking 切换需要严格验证** — POC 必须测三个场景
2. **编辑模式闪烁用状态比对修** — 治本且改动小
3. **OCR 卡死需要日志定位** — 不要盲目改代码
4. **大段文本去重只对比尾部** — 字幕变化通常在底部
5. **优先级：闪烁 > OCR 诊断 > thinking 验证** — 符合项目实际

---

## 🔧 技术细节速查

### 状态比对修闪烁
```python
class SelectionBox:
    def __init__(self):
        self._last_applied_state = None

    def apply_model(self, model):
        current_state = (model.geometry, model.group_id, model.lang_pair, model.edit_mode)
        if current_state == self._last_applied_state:
            return
        self._last_applied_state = current_state
        # ... 原有逻辑
```

### OCR 诊断日志
```python
def recognize(self, img, lang):
    t0 = time.time()
    logger.debug(f"[OCR-IN] {lang} img_shape={img.shape}")
    result = self._paddle_ocr[lang].ocr(img, cls=False)
    logger.debug(f"[OCR-OUT] {lang} elapsed={time.time()-t0:.2f}s result_len={len(result[0]) if result[0] else 0}")
    return result
```

### 大段文本去重优化
```python
def _should_skip_duplicate(self, new_text, old_text):
    if len(new_text) > 500:  # 大段文本
        # 只对比最后 200 字符（字幕通常底部变化）
        return trigram_similarity(new_text[-200:], old_text[-200:]) > 0.9
    else:
        return trigram_similarity(new_text, old_text) > 0.85
```

---

## 👥 协作 AI 评估

### 执行 AI（写 huifu.md 的）
- **架构理解**：8/10 — 能看懂复杂方案并指出逻辑漏洞
- **实现细节**：6/10 — 边界情况考虑不够
- **优先级判断**：9/10 — 和项目方判断完全一致
- **风险意识**：7/10 — 能指出别人的风险，但自己也有盲区
- **沟通能力**：9/10 — 表格清晰、逻辑分层、结论明确

**适合做**：架构评审、方案讨论、优先级决策
**需要 review**：边界情况、错误处理、性能优化
**不适合做**：底层 C++/Win32、PaddleOCR 调优

**最大价值**：第三方视角指出盲区（如我的过度设计）
**局限**：判断也有偏差（如对 thinking 切换过度乐观）

---

## 📋 待办事项模板

### 当前紧急
- [ ] 修复编辑模式闪烁
- [ ] OCR 卡死诊断日志
- [ ] 翻译方向持久化
- [ ] thinking 切换 POC

### 验证阶段
- [ ] POC 结果决定 Agent 方案
- [ ] 术语索引提取精度验证

### 功能开发
- [ ] Agent 核心开发
- [ ] GUI 翻译模板重构
- [ ] OCR 冷启动预热

### 长期优化
- [ ] exe 瘦身（720MB → 300MB）
- [ ] 工具栏交换语言按钮

---

## 🔗 快速导航

**必读文档**：
- `communication/README.md` — 对话流程说明
- `communication/leadersession/context.md` — 当前会话状态
- `plans/02-Agent化改造方案.md` — 核心改造方案
- `plans/03-待解决问题清单.md` — 当前 bugs

**对话链路**：
jiaojie.md → react.md → huifu.md → task.md

**工作文件**（执行时创建）：
- `communication/progress.md` — 任务进度
- `communication/blocked.md` — 卡住的问题
- `communication/question.md` — 需要回答的疑问
