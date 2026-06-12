# 架构师 AI 方案评审报告 - 2026-06-11

> 评审对象：Deep 设计的任务1-4修复方案（task.md）

---

## ✅ 总体结论

**方向正确，优先级合理，验收标准明确**

发现 3 个细节问题，已全部修正完毕。

---

## 逐任务评审

### 任务1：修复编辑模式闪烁 ✅

**方案**：状态比对 + `_last_applied_state`

**问题**：❌ 状态字段对不上
- task.md 写 `model.geometry, model.lang_pair, model.edit_mode`
- 实际 `SelectionBoxModel` 没有这些字段

**已修正**：
```python
current_state = (
    (model.x, model.y, model.width, model.height),
    model.group_id,
    (model.source_language, model.target_language),
    self._editable
)
```

**验收标准**：合理 ✅

---

### 任务2：OCR卡死诊断日志 ✅

**方案**：入口/出口加时间戳日志

**问题**：❌ 日志位置不够精确
- Deep 写"在 `OCREngine.recognize()` 加日志"
- 实际应该在 `_run_paddle()` 内部，才能定位 PaddleOCR 本身

**已修正**：
- PaddleOCR日志：`app/ocr/engine.py` 的 `_run_paddle()` 方法
- 去重日志：`app/translation/service.py` 的 `_is_similar()` 方法

**验收标准**：合理 ✅

---

### 任务3：翻译方向持久化 ✅

**方案**：在语言选择回调调用 `AppSettings.save()`

**问题**：❌ 缺少文件路径
- Deep 没说明在哪个文件修改
- 执行AI会浪费时间找文件

**已修正**：
- 修改 `app/gui/main_window.py` 的 `LanguagePage._emit_change()` 方法
- 从 `AppSettings.load()` 读取默认值在初始化时

**验收标准**：合理 ✅

---

### 任务4：DeepSeek thinking 切换 POC ✅

**方案**：三个测试场景

**评审**：✅ 完全正确，无需调整

**关键**：必须测 `usage.reasoning_tokens` 计费

---

## 修正清单

| 问题 | 影响 | 已修正 |
|------|------|--------|
| 状态字段对不上 | 执行AI会卡住 | ✅ |
| OCR日志位置不精确 | 无法定位真正问题 | ✅ |
| 缺少文件路径 | 浪费时间找文件 | ✅ |

---

## ✅ 可以执行

task.md 和 react.md 已全部修正，可以让执行AI开始干活。
