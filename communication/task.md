# 任务：翻译模板 GUI 重构

> 发给执行 AI。完成后审查 AI 验收。

---

## 目标

侧边栏从四页改为三页。"翻译模板"合并当前"翻译方向"和"提示词"两页。

---

## 当前状态

```
侧边栏                        右侧内容
┌──────────┐                ┌──────────────────┐
│ 翻译方向  │ → LanguagePage   │ 选择源/目标语言    │
│ 模型配置  │ → ModelPage      │ Base URL 等       │
│ 提示词   │ → PromptPage     │ 约束层+引用层       │
│ 设置     │ → SettingsPage   │ 快捷键            │
└──────────┘                └──────────────────┘
```

## 目标状态

```
侧边栏                        右侧内容
┌──────────┐                ┌──────────────────┐
│ 翻译模板  │ → TemplatePage  │ 语言方向          │
│ 模型配置  │                │ 约束层编辑         │
│ 设置     │                │ 知识引用层         │
└──────────┘                │ 编译预览+保存      │
                            └──────────────────┘
```

---

## 具体要求

### 改动 1：新建 `TemplatePage`

**文件**：`app/gui/main_window.py`

新建 `TemplatePage(QWidget)` 类，内部用 `QVBoxLayout` 垂直排列：

1. **语言方向区** — 顶部。源语言 QComboBox + ⇄ 交换按钮 + 目标语言 QComboBox（从 `LanguagePage` 迁移）
2. **约束层区** — 预览按钮 + 弹窗编辑（从 `PromptPage` 迁移）
3. **知识引用区** — 文档列表 + 添加/移除按钮（从 `PromptPage` 迁移）
4. **编译区** — Compiled Prompt 路径只读显示 + "生成预览" + "保存并启用"（从 `PromptPage` 迁移）
5. 整个页面套 `QScrollArea`，内容多时可滚动

**信号保留**：
- `language_changed(str, str)` — 和现在 `LanguagePage` 一样

### 改动 2：修改 `MainWindow`

**文件**：`app/gui/main_window.py`

- 侧边栏 `_nav_language`、`_nav_prompt` 两个按钮 → 合并为一个 `_nav_template`（"翻译模板"）
- `_stack` 中删除 `_language_page` 和 `_prompt_page`，替换为一个 `_template_page`
- `_on_nav_clicked` 调整索引映射
- `_on_language_changed` 连接 `_template_page.language_changed`
- `default_language_pair()` 从 `_template_page` 取

### 改动 3：更新测试

**文件**：`tests/test_gui_shell.py`

- `test_main_window_has_sidebar_and_pages` 改为检查 3 个按钮 + 3 页
- 新增：`test_template_page_has_language_and_prompt_sections` 验证 TemplatePage 包含语言选择和约束层

### 不改动

- OCR、截图、巡检、翻译管线、日志、打包
- `ModelPage` 和 `SettingsPage` 内部逻辑
- `LanguagePage` 和 `PromptPage` 可以保留在文件中但不再被 MainWindow 引用

---

## 验收标准

- ✅ `python -m unittest discover` 全部通过（至少 78 个，新增 1+）
- ✅ 侧边栏只有 3 个按钮：翻译模板、模型配置、设置
- ✅ 翻译模板页包含：语言选择 + 约束层预览按钮 + 知识引用列表 + 编译区
- ✅ 翻译模板页可滚动
- ✅ 点击约束层预览按钮弹出编辑弹窗（现有功能不变）
- ✅ 现有功能不受影响（模型配置页、设置页、测试连接）

---

## 执行规则

1. 改一小块就跑 `test_gui_shell` 和全量测试
2. 不改 OCR、翻译管线、日志、打包
3. 卡住 20 分钟以上→停止，写 `communication/blocked.md`
4. **提交前自己跑一次 `python -m unittest discover`**
