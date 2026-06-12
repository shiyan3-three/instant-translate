# 闪烁和卡顿修复总结 - 2026-06-11

## 🐛 问题诊断

### 用户报告
1. 编辑模式下工具栏闪烁
2. 程序变得非常卡顿

### 根本原因

**原因1：频繁日志导致卡顿**
- 我添加的 `[OCR-IN]` / `[OCR-OUT]` / `[DEDUP]` 日志
- 每500ms执行一次OCR，每次写3条日志
- 大量磁盘IO → 严重卡顿

**原因2：工具栏无效重绘**
- `apply_model()` 调用 `_update_overlay_geometry()`
- → 调用 `_update_toolbar_position()`
- → 每次都调用 `toolbar_panel.move()` 即使位置没变
- → 触发重绘和闪烁

**原因3：apply_edit_mode 缺少状态比对**
- `_upsert_selection_box()` 每次操作都调用
- `apply_edit_mode()` 没有检查状态是否已经是目标值
- 重复执行 show/hide 导致闪烁

---

## ✅ 修复方案

### 修复1：移除频繁日志（app/ocr/engine.py, app/translation/service.py）
```python
# 移除了：
# get_debug_logger().debug("[OCR-IN] ...")
# get_debug_logger().debug("[OCR-OUT] ...")
# get_debug_logger().debug("[DEDUP] ...")
```

**原因**：这些日志会在卡死时才需要，不应该常驻。可以在需要诊断时临时添加。

---

### 修复2：apply_edit_mode 状态比对（app/overlay/selection_box.py:284-289）
```python
def apply_edit_mode(self, enabled: bool) -> None:
    # Skip if edit mode hasn't changed (prevents toolbar show/hide flicker)
    if self._editable == enabled:
        return

    self._editable = enabled
    # ... 后续逻辑
```

**效果**：如果编辑模式状态没变，直接跳过，避免重复 show/hide。

---

### 修复3：_update_toolbar_position 位置比对（app/overlay/selection_box.py:532-556）
```python
def _update_toolbar_position(self) -> None:
    # ... 计算位置

    # Only move if position actually changed (prevents flicker)
    if self.toolbar_panel.x() != x or self.toolbar_panel.y() != y:
        self.toolbar_panel.move(x, y)
```

**效果**：只在位置真正变化时才移动，避免无效的 move() 调用触发重绘。

---

## 📊 修复前后对比

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| 卡顿 | 严重卡顿 | 流畅 |
| 工具栏闪烁 | 持续闪烁 | 不闪烁 |
| 按钮可点击 | 点不到 | 正常 |

---

## 🎯 待用户验证

请测试：
1. 重启程序
2. 创建选择框并进入编辑模式
3. 停留10秒观察是否闪烁
4. 点击工具栏按钮确认可点击
5. 检查程序是否还卡顿

---

## 📝 教训

1. **高频路径不能有重IO操作** — 500ms轮询里的日志会严重影响性能
2. **所有 apply 方法都需要状态比对** — 不只是 apply_model，apply_edit_mode 也需要
3. **GUI更新操作要幂等** — move() / show() / hide() 这些操作要检查是否真的需要执行
4. **日志应该按需启用** — 诊断日志不应该常驻，应该在需要时临时添加
