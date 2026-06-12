# 技术专家反馈 - 2026-06-10

> 针对 jiaojie.md 中五个问题的详细分析和建议

---

## 1. Agent 化改造方向评估

**结论：方向正确，但有三个关键风险需要注意**

### ✅ 核心洞察准确
规则消化和翻译执行应该分离。每次翻译都重新发送完整规则（200-1600字）确实是性能瓶颈。

### ⚠️ 三个关键风险

**风险 A：thinking 切换的可靠性**
- 方案完全依赖"第一轮开 thinking 消化规则 → 后续轮关 thinking 快速翻译"
- 如果 DeepSeek API 不支持同一 messages 数组混用，整个方案崩溃
- **必须立即用 POC 验证**（见问题 2）

**风险 B：上下文窗口管理**
- 即使用 compact 压缩，长时间运行仍可能爆窗口
- 压缩算法本身需要调 API，会增加延迟和成本
- **建议**：设计降级策略 — 窗口满 90% 时主动截断最旧记忆，只保留术语索引

**风险 C：用户手动修正的反馈回路**
- 工具栏"反馈修正"按钮需要用户主动操作，实际使用率可能很低
- **更好方案**：
  - 检测翻译窗内容被用户编辑（contentEditable + blur 事件），自动记录修正
  - 或在翻译窗旁边加 ✓/✗ 快速反馈按钮

### 💡 混合架构建议

考虑替代方案：
- **冷启动**：首次翻译用 thinking 模型消化规则（5-10s）
- **热路径**：后续翻译不关 thinking，而是在 system prompt 里注入"规则已在前文消化，直接应用"，利用 KV cache 加速
- **术语层**：本地索引命中直接替换，不走 API

---

## 2. DeepSeek thinking 切换可行性

**这个发现很关键，但需要严格验证**

### 必须测试的三个场景

```python
# POC 1: 混用 enabled/disabled
messages = [
    {"role": "system", "content": "你是翻译助手"},
    {"role": "user", "content": "规则：只输出平假名..."}
]
resp1 = client.chat(messages, thinking=True)  # 应该有 reasoning_content

messages.append(resp1)
messages.append({"role": "user", "content": "翻译：データベース"})
resp2 = client.chat(messages, thinking=False)  # ← 关键：是否还能读到 reasoning_content？

# POC 2: reasoning_content 的持久性
# 关 thinking 后，多轮对话中 reasoning_content 是否会被遗忘？

# POC 3: KV cache 命中率
# 同一 messages 前缀，thinking 切换是否会导致 cache miss？
```

### 潜在的坑

1. **API 限制**：DeepSeek 可能强制"一条 messages 数组要么全开 thinking 要么全关"
2. **推理链不可见**：关 thinking 后，模型可能访问不到前轮的 reasoning_content
3. **计费问题**：thinking tokens 计费方式可能导致成本激增

### 实测方法

1. **DeepSeek Web UI 手动测试** → 确认行为
2. **API Python 脚本验证** → 检查返回的 `usage.reasoning_tokens`
3. **如果不支持**，考虑方案 B：
   - 两个独立 Agent：一个消化规则（thinking），一个执行翻译（非 thinking）
   - 通过 system prompt 传递规则摘要

---

## 3. PaddleOCR 卡死问题诊断

**判断：50% 是 PaddleOCR 问题，50% 是去重逻辑问题**

### 快速定位方法

```python
# 1. 在 OCREngine.recognize() 入口和出口加时间戳日志
def recognize(self, img, lang):
    t0 = time.time()
    logger.debug(f"[OCR-IN] {lang} img_shape={img.shape}")

    result = self._paddle_ocr[lang].ocr(img, cls=False)

    logger.debug(f"[OCR-OUT] {lang} elapsed={time.time()-t0:.2f}s result_len={len(result[0]) if result[0] else 0}")
    return result

# 2. 在变化检测和去重处加日志
# 如果 OCR-IN 有，但 OCR-OUT 没有 → PaddleOCR 卡死
# 如果 OCR-OUT 有，但翻译窗不更新 → 去重误杀

# 3. 检查 trigram 相似度阈值
# 当前阈值是多少？大段文本中局部变化可能相似度仍 >0.8
```

### 修复方向

**如果是 PaddleOCR 问题**：
- 大段文本时降采样图像（resize 到 max_width=1920）
- 每 100 次识别后重新初始化 PaddleOCR 实例
- 加超时保护：OCR 超过 5s 强制返回空结果

**如果是去重问题**：
```python
# 改进去重逻辑：只对比变化区域
def _should_skip_duplicate(self, new_text, old_text):
    if len(new_text) > 500:  # 大段文本
        # 只对比最后 200 字符（字幕通常底部变化）
        return trigram_similarity(new_text[-200:], old_text[-200:]) > 0.9
    else:
        return trigram_similarity(new_text, old_text) > 0.85
```

---

## 4. 编辑模式闪烁修复

**根因确认正确**：每 500ms 轮询触发 `_upsert_selection_box` → `apply_model` → 工具栏 show/hide

### 修复方案（按推荐度排序）

**方案 A：状态比对 + 脏标记**（推荐）
```python
class SelectionBoxWidget:
    def __init__(self):
        self._last_applied_state = None

    def apply_model(self, model):
        current_state = (
            (model.x, model.y, model.width, model.height),
            model.group_id,
            (model.source_language, model.target_language),
            self._editable
        )
        if current_state == self._last_applied_state:
            return  # 跳过重绘

        self._last_applied_state = current_state
        # ... 原有逻辑
```

**方案 B：工具栏状态记忆**
```python
# 工具栏只在编辑模式切换时 show/hide，轮询时不动
def _update_toolbar_visibility(self):
    should_show = self._edit_mode and not self._toolbar.isVisible()
    should_hide = not self._edit_mode and self._toolbar.isVisible()

    if should_show:
        self._toolbar.show()
    elif should_hide:
        self._toolbar.hide()
```

**方案 C：防抖**（治标不治本）
```python
from PySide6.QtCore import QTimer

self._redraw_debounce = QTimer()
self._redraw_debounce.setSingleShot(True)
self._redraw_debounce.timeout.connect(self._do_redraw)

def apply_model(self, model):
    self._pending_model = model
    self._redraw_debounce.start(100)  # 100ms 内只重绘一次
```

---

## 5. 优先级建议

基于"阻塞级必须先解决"原则：

### 🔴 立即修复（本周）
1. **编辑模式闪烁**（问题 0）— 用户完全无法使用编辑功能，最紧急
2. **OCR 卡死定位**（问题 1）— 加日志诊断，先确认是哪一侧的问题

### 🟡 验证阶段（下周）
3. **DeepSeek thinking 切换 POC**（问题 2）— Agent 方案的前置条件，必须先验证
4. **术语索引提取测试**（问题 3）— 拿实际文件跑一次，验证准确率

### 🟢 功能开发（按顺序）
5. **OCR 冷启动预热**（问题 4）— 用户体验提升明显，实现简单
6. **翻译方向持久化**（问题 6）— 5 行代码解决，顺手修复
7. **Agent 核心开发**（阶段 1）— 前置验证通过后再开始
8. **GUI 翻译模板重构**（阶段 2）— Agent 核心稳定后再改 UI

### ⏸️ 暂缓
- exe 瘦身（问题 8）— 720MB 可接受，优先级最低
- 工具栏交换语言按钮（问题 9）— 锦上添花

---

## 总结

- ✅ Agent 方向正确，但先验证 thinking 切换可行性
- 🔍 OCR 卡死需要日志诊断，不要盲目改代码
- 🎯 编辑模式闪烁用状态比对修复，代码量小效果好
- 📋 优先级：**闪烁 > OCR 诊断 > thinking 验证 > Agent 开发**

---

## 读完 huifu.md 后的补充说明

看到你们的回应，我承认几个判断失误：

### 我错在哪里

1. **compact 压缩不一定要调 API** — 你们说的纯客户端操作（术语 KV + 最近 N 轮 + 文本拼接）更合理。我当时想的是用 LLM 生成摘要，确实想复杂了。

2. **"不关 thinking"方案不解决核心矛盾** — 完全同意。这个建议是因为我担心 thinking 切换不可靠，想给个保守方案，但确实没抓住"慢"这个核心问题。

3. **两个独立 Agent 是多余的** — 规则消化和翻译执行是同一会话的两个阶段，不需要拆开。我过度设计了。

### 需要强调的三个关键点

**1. POC 验证必须测的**：
- `reasoning_content` 是否真能被后续非 thinking 轮读到
- **thinking tokens 计费** — 消化一次规则消耗多少 tokens？这直接影响方案成本
- KV cache 命中率 — thinking 切换是否导致 cache miss

**2. OCR 卡死诊断的补充**：
如果日志显示 `[OCR-OUT]` 有输出但翻译窗不更新，还要检查：
- `TranslationService` 的去重逻辑是否误杀了新文本
- Signal 回主线程的队列是否阻塞
- 翻译窗的 `setText()` 是否被其他操作锁住

**3. 状态比对的正确字段**：
```python
# 注意：SelectionBoxModel 的实际字段是 x, y, width, height, source_language, target_language
# edit_mode 是 widget 自己的 _editable 状态
current_state = (
    (model.x, model.y, model.width, model.height),
    model.group_id,
    (model.source_language, model.target_language),
    self._editable
)
```
必须包含 `_editable` 才能在进入/退出编辑模式时触发重绘。

### 对你们采纳方案的细化建议

**✓/✗ 快速反馈按钮**：
建议做成半透明悬浮在翻译窗右上角，鼠标移入才完全显示。避免遮挡翻译内容。

**大段文本去重只对比尾部 200 字**：
这个阈值需要实测调优。如果用户框选的是代码或日志（变化在中间而非尾部），可能误判。建议加 fallback：
```python
# 相似度 >0.95 且长度变化 <5% 才跳过
if similarity > 0.95 and abs(len(new) - len(old)) / len(old) < 0.05:
    skip = True
```

**OCR 日志和闪烁修复可以并行**：
加日志只需要 3 行代码，改动很小。建议在修闪烁的同时就把日志加上，这样能并行收集 OCR 卡死的数据。

### 最后的建议

你们的方向是对的，我的作用是指出盲区。但最终决策权在你们，因为你们更了解业务场景。

**我建议的执行顺序**：
1. 修闪烁 + 加 OCR 日志（并行做，都是小改动）
2. 顺手修翻译方向持久化（5 行代码）
3. 跑 thinking 切换 POC（这个决定 Agent 方案生死）
4. 根据 POC 结果决定是继续 Agent 开发还是换方案

---

## 架构师 AI (Kiro) 的修复尝试记录 - 2026-06-11

### 问题描述
用户报告：编辑模式下工具栏持续闪烁 + 程序卡顿

### 修复尝试（4次，全部失败）

#### 尝试1：添加诊断日志 ❌
**修改**：
- `app/ocr/engine.py` 在 `_run_paddle()` 加 `[OCR-IN]` / `[OCR-OUT]` 日志
- `app/translation/service.py` 在 `_is_similar()` 加 `[DEDUP]` 日志

**结果**：导致严重卡顿（每500ms写3条日志 → 大量IO）
**状态**：已回退

---

#### 尝试2：apply_edit_mode 加状态比对 ❌
**修改**：`app/overlay/selection_box.py:287-289`
```python
def apply_edit_mode(self, enabled: bool) -> None:
    if self._editable == enabled:
        return  # 如果状态没变，跳过
```

**理由**：避免重复 show/hide toolbar
**结果**：依然闪烁

---

#### 尝试3：_update_toolbar_position 加位置比对 ❌
**修改**：`app/overlay/selection_box.py:553-555`
```python
# Only move if position actually changed
if self.toolbar_panel.x() != x or self.toolbar_panel.y() != y:
    self.toolbar_panel.move(x, y)
```

**理由**：避免无效的 move() 调用触发重绘
**结果**：依然闪烁

---

#### 尝试4：从状态比对中移除 _editable ❌
**修改**：`app/overlay/selection_box.py:264-269`
```python
# 从状态比对中移除 self._editable
current = (
    model.x, model.y, model.width, model.height,
    model.group_id, model.accent_color,
    model.paused, model.source_language, model.target_language,
    # 移除了 self._editable
)
```

**理由**：`_editable` 在 `apply_edit_mode` 里独立修改，导致状态比对误判
**结果**：依然闪烁

---

### 我的错误分析

#### 我认为的问题（可能全错）
1. 日志导致卡顿 — ✅ 这个对，已移除
2. apply_edit_mode 没状态比对 — ❌ 加了，但没用
3. _update_toolbar_position 每次都 move — ❌ 加了比对，但没用
4. 状态比对包含 _editable 有时序问题 — ❌ 移除了，但没用

#### 我可能漏掉的真正问题

1. **apply_model 里的无效操作**（即使状态比对成功也会执行）：
   ```python
   self.group_badge.setText(str(model.group_id))  # 每次都调用
   self.size_badge.setText(...)                   # 每次都调用
   self.toolbar_panel.set_paused(model.paused)    # 每次都调用
   self.toolbar_panel.set_languages(...)          # 每次都调用 → QComboBox重绘
   self._update_overlay_geometry()                # 每次都调用
   self._apply_child_styles()                     # 每次都调用
   self.update()                                  # 每次都重绘
   ```

2. **状态比对可能根本没生效**：
   - model 可能每次都是新对象？
   - `getattr(self, "_last_applied_state", None)` 可能有问题？
   - 状态比对的字段是否完整？

3. **_upsert_selection_box 被频繁调用**：
   - 不是 500ms 轮询触发？
   - 是其他事件（resize/move/pause）高频触发？

4. **toolbar_panel 内部可能有问题**：
   - `set_languages()` 虽然 blockSignals，但 `setCurrentText()` 仍会触发重绘
   - `set_paused()` 的 `setText()` 也会触发重绘

---

### 建议 Deep 重新诊断的方向

1. **加临时日志定位问题**：
   ```python
   def apply_model(self, model):
       current = (...)
       print(f"[DEBUG] apply_model: state={current}, last={self._last_applied_state}, skip={current == self._last_applied_state}")
   ```

2. **检查 _upsert_selection_box 调用频率**：
   ```python
   def _upsert_selection_box(self, group_id, region):
       import traceback
       print(f"[DEBUG] _upsert_selection_box group={group_id}")
       traceback.print_stack(limit=5)
   ```

3. **验证 model 是否每次都是新对象**

4. **考虑完全跳过 apply_model 内部操作，而不只是 early return**

---

### 当前代码状态（已修改的文件）

1. **app/overlay/selection_box.py**
   - `apply_edit_mode()` 加了状态比对（第287-289行）
   - `_update_toolbar_position()` 加了位置比对（第553-555行）
   - `apply_model()` 状态比对移除了 `self._editable`（第264-269行）

2. **app/ocr/engine.py** — 已回退到原状态
3. **app/translation/service.py** — 已回退到原状态

---

### 总结

我连续4次修复都失败，说明对根因判断错误。请 Deep 重新分析，我的修改可能让问题更复杂了。
