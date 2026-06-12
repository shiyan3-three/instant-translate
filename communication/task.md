# 任务指令 - 2026-06-10

> 给执行 AI 的明确任务列表

---

## 🔴 立即执行（今天必须完成）

### 任务 1：修复编辑模式闪烁

**目标**：选择框编辑模式下，badge/工具栏不再闪烁，用户能正常点击按钮

**具体要求**：
1. 修改文件：`app/overlay/selection_box.py`
2. 在 `SelectionBoxWidget` 类的 `__init__` 中添加 `_last_applied_state = None`
3. `apply_model()` 方法开头加状态比对：
   ```python
   current_state = (
       (model.x, model.y, model.width, model.height),
       model.group_id,
       (model.source_language, model.target_language),
       self._editable
   )
   if current_state == self._last_applied_state:
       return
   self._last_applied_state = current_state
   ```
4. 测试：编辑模式下停留 10 秒，工具栏不闪烁，按钮能点击

**交付物**：
- 修改后的代码文件路径
- 测试结果截图或日志

---

### 任务 2：OCR 卡死诊断日志

**目标**：在 OCR 入口/出口加时间戳日志，定位卡在 PaddleOCR 还是去重逻辑

**具体要求**：
1. 修改文件：`app/ocr/engine.py` 的 `_run_paddle()` 方法
   - 在 `ocr.ocr(array)` 前加：
     ```python
     import time
     from app.logger import get_debug_logger
     t0 = time.time()
     get_debug_logger().debug(f"[OCR-IN] lang={lang_code} img_shape={array.shape}")
     ```
   - 在 `ocr.ocr(array)` 后加：
     ```python
     get_debug_logger().debug(f"[OCR-OUT] lang={lang_code} elapsed={time.time()-t0:.2f}s raw_len={len(raw_results) if raw_results else 0}")
     ```
2. 修改文件：`app/translation/service.py` 的 `_is_similar()` 方法
   - 在 `return overlap >= ...` 前加：
     ```python
     from app.logger import get_debug_logger
     get_debug_logger().debug(f"[DEDUP] overlap={overlap:.3f} threshold={self._SIMILARITY_THRESHOLD:.3f} skip={overlap >= self._SIMILARITY_THRESHOLD}")
     ```
3. 运行程序，框选大段文本，等待卡死现象，导出 `debug.log`

**交付物**：
- 修改后的代码
- 出现卡死时的完整 debug.log（最后 100 行）

---

### 任务 3：顺手修复翻译方向持久化

**目标**：主窗口选择的语言对重启后保存

**具体要求**：
1. 修改文件：`app/gui/main_window.py`
2. 在 `LanguagePage` 的 `_emit_change()` 方法中调用 `AppSettings` 保存
3. 在 `MainWindow` 初始化 `LanguagePage` 时从 `AppSettings.load()` 读取默认语言对
4. 测试：选中文→英文，重启程序，语言对保持不变

**交付物**：
- 修改后的代码文件路径

---

## 🟡 并行准备（不阻塞上面的任务）

### 任务 4：DeepSeek thinking 切换 POC

**目标**：验证同一 messages 数组能否混用 thinking enabled/disabled

**具体要求**：
1. 新建 `tests/poc_deepseek_thinking.py`
2. 实现三个测试场景：
   ```python
   # 场景 1: 混用 enabled/disabled
   # 场景 2: reasoning_content 持久性（多轮对话）
   # 场景 3: 检查 usage.reasoning_tokens 计费
   ```
3. 运行并记录：
   - 是否报错
   - 关 thinking 后正确率是否下降
   - thinking tokens 消耗量

**交付物**：
- POC 脚本代码
- 测试结果 markdown 报告（包含每个场景的输入/输出/结论）

---

## ⚠️ 执行规则

1. **按顺序做**：任务 1 → 任务 2 → 任务 3，完成一个再开始下一个
2. **任务 4 可以并行**：如果 API 调用需要等待，可以在等待时切到任务 1-3
3. **不要擅自改需求**：
   - 不要"顺便优化"其他代码
   - 不要"我觉得这样更好"就改方案
   - 如果发现需求有问题，先在 `communication/question.md` 里提问，等回复再动手
4. **每个任务完成后**：
   - 在 `communication/progress.md` 里更新进度
   - 写清楚改了哪些文件、测试结果如何
5. **如果卡住超过 30 分钟**：
   - 立即停止
   - 在 `communication/blocked.md` 里说明卡在哪里、尝试了什么、需要什么帮助

---

## 📋 优先级说明

- 任务 1（闪烁）是用户完全无法使用的阻塞问题，最高优先级
- 任务 2（OCR 日志）是诊断工具，不修复问题但必须先定位
- 任务 3（持久化）是 5 行代码的小问题，顺手修
- 任务 4（POC）决定 Agent 方案是否可行，但不阻塞当前开发

---

## ✅ 验收标准

**任务 1 通过条件**：
- 编辑模式下工具栏不闪烁
- 能正常点击工具栏按钮
- 代码逻辑清晰，没有引入新 bug

**任务 2 通过条件**：
- 日志能明确显示 OCR 耗时
- 能区分是 PaddleOCR 慢还是去重误杀
- 日志格式统一，易于分析

**任务 3 通过条件**：
- 重启后语言对保持不变
- 不影响现有功能

**任务 4 通过条件**：
- 三个场景全部测试
- 结论明确（支持/不支持/有限支持）
- 如果不支持，提出替代方案
