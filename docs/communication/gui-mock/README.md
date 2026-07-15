# GUI Mock · 实现交接说明

> **给谁看**：负责把视觉装进 **PySide6 真 GUI** 的 AI / 工程师。  
> **不是运行时**：本目录是浏览器视觉沙盒，**不**参与 `app/` 启动链路。  
> **实现 AI 主读（代码规格，不依赖视觉）**：**`implementer.html`** — 结构树 / 尺寸 / QSS / 断言。  
> **人类看图（可选）**：`index.html`。  
> **设计 token 文档**：`docs/goat/14-GUI设计规范.md`（与 mock `:root` 对齐）。

---

## 0. 30 秒

| 项 | 内容 |
|----|------|
| 产品 | Windows 屏幕区域 OCR + AI 翻译悬浮工具 |
| Mock 目的 | 冻结主窗 / 任务流 / 弹窗 / Overlay 的**外观与交互意图** |
| 落地目标 | 改 `app/gui/` + `app/overlay/`（QSS + 布局），**不要**把 HTML 搬进 Python |
| 本阶段状态 | 视觉沙盒基本可冻结；细抠像素可后置 |
| 仓库 | `https://github.com/shiyan3-three/instant-translate` · 分支 `main` |

**默认工作顺序**

```text
1. 读 implementer.html（代码树/数字/QSS）+ 本 README
2. 读现有 app/gui/main_window.py objectName + QSS（以代码为准）
3. 无视觉模型：不要假设能「看截图」；只按 implementer 数字实现
4. 一次只落地一块（先方案再动手；用户常要求「先说方案」）
5. 单测 + 几何断言思路见 implementer §11；人类可用 index.html 抽检
```

---

## 1. 目录文件

| 文件 | 角色 |
|------|------|
| `index.html` | 分阶段视觉沙盒（Phase 1–4） |
| `styles.css` | 全部 mock 样式与 token（对齐 14 规范） |
| `app.js` | 阶段切换、侧栏、Tab、Overlay 轻交互（无后端） |
| **`implementer.html`** | **代码规格页**（结构树/尺寸/QSS/断言；无视觉模型主读） |
| `README.md` | 本文件：细节、映射、坑、验收 |

只允许这三个交互资源 + 交接文档：`html/css/js` 核心三件套；不要再拆成多页应用。

---

## 2. Mock 四阶段（index.html）

| Phase | 名称 | 你要对齐什么 |
|:-----:|------|----------------|
| 1 | 主窗骨架 | 标题栏品牌 + 当前页名；侧栏选中左条；内容卡片；每页一个主按钮 |
| 2 | 任务流 | 复用 Phase1 shell；侧栏切到「模型配置」「优化翻译」看对照布局 |
| 3 | 弹窗 | 编译预览（易懂/高级 Tab）；AI 审查（侧栏状态点 + 短按钮文案） |
| 4 | Overlay | 选择框、主工具栏、译文/OCR 卡、角工具栏、◎ 沉浸、多组示意、拖拽 |

**Phase 2 实现细节**：`app.js` 在 phase=2 时**同时显示** phase-1 shell，便于在真实窗体内切换页面。

---

## 3. 代码映射（Mock → PySide6）

| Mock 区域 | 优先改的代码 | 备注 |
|-----------|--------------|------|
| 主窗壳 / 侧栏 / 页 | `app/gui/main_window.py` | `TitleBar`、`NavButton`、各 `*Page`、`setObjectName` + QSS |
| 设置窗（若仍独立） | `app/gui/settings_window.py` | 与主窗风格一致 |
| 托盘 | `app/gui/tray_icon.py` | 一般不动视觉 |
| 选择框 / 主工具栏 | `app/overlay/selection_box.py` | 语言对、重选、OCR、删除、组沉浸 |
| 译文悬浮窗 | `app/overlay/translation_window.py` | 白底卡、badge、角工具 |
| OCR 悬浮窗 | `app/overlay/ocr_text_window.py` | 深色卡、暂停/刷新/复制 |
| 编辑模式 | `app/overlay/edit_mode_controller.py` | 编辑态显工具栏 |
| 选区编排 | `app/overlay/selection_manager.py` 等 | 多组、信号 |

**已有 objectName 示例（落地时沿用，勿另起一套）：**

- `titleBar` / `titleBarButton` / `titleBarCloseButton`
- `navButton`
- `contentPage` / `pageTitle` / `pageDesc`
- `swapButton` / `primaryButton` / `secondaryButton`
- `compiledPromptDialog` / `compiledPromptTabs` / `compiledPromptViewer`
- `modelSelectCombo` / `hintLabel` / 进度条相关 id（若已有 `#aiProgressBar` 等）

QSS 优先集中写在主窗 stylesheet；**禁止**大面积 `setStyleSheet` 碎片化，除非局部状态色（成功/失败提示）。

---

## 4. Token（必须一致）

与 `styles.css` `:root` / `14-GUI设计规范.md` 对齐：

| Token | 值 | 用途 |
|-------|-----|------|
| 最深背景 | `#0a0f1a` | 侧栏、标题栏、状态栏 |
| 主背景 | `#0f172a` | 内容区 |
| 卡片 | `#1e293b` | 卡片、输入底 |
| 边框 | `#334155` | 描边 |
| 主文字 | `#e2e8f0` | 正文 |
| 次要 | `#94a3b8` | 描述、未选中 |
| 弱 | `#64748b` | 占位、状态栏 |
| 强调 | `#60a5fa` | 焦点、侧栏选中 |
| 主按钮 | `#2563eb` | primary |
| 成功 / 警告 / 危险 | `#4ade80` / `#fbbf24` / `#f87171` | 状态 |

组别色：G1 `#60a5fa` · G2 `#4ade80` · G3 `#fbbf24`。

字体：`"Segoe UI", "Microsoft YaHei", sans-serif`；页面标题约 22px/700；正文 13px。

---

## 5. 已拍板的交互 / 布局决策

落地时**按此实现**，不要自行「优化成并排」或改回旧布局。

### 5.1 主窗 · 翻译模板

- 语言方向：源语言 select · **交换按钮 ⇄** · 目标语言 select。
- **⇄ 必须与 select 底边平齐**（不要相对整列垂直居中导致偏上）。Mock：`.swap { align-self: flex-end; }`。
- 每页**一个**主按钮（保存并启用）；次要操作为 secondary。

### 5.2 主窗 · 模型配置

- Fast / Thinking 模型为 **不可手输的下拉**，列表来自 `GET /models`，**禁止硬编码模型目录**。
- 已有 `list_models()` / 自动拉取逻辑时，只改 UI 呈现，不改协议语义。

### 5.3 主窗 · 优化翻译（Feedback）

- **OCR 原文**、**当前译文**：**上下各一整行**（`compare-stack`），**不要**左右 `grid-2` 并排挤在一起。
- 四个文本区高度约定（mock 已定）：
  - `.compare-ta`（OCR / 当前译文）：**90px**
  - `.compare-ta-sm`（备注 / 认可译文）：**90px**
- Tab：`译文修正` | `长期记忆（可选）`。
- 底栏：保存备注 · 让 AI 优化 · **仅采用译文**（主）· 忽略。
- 长期记忆需**单独确认**，不要和「仅采用译文」绑死。

### 5.4 弹窗

- **编译预览**：深色 Tab（`#compiledPromptTabs` 一类）；易懂说明 / 高级内容；主按钮「确认并启用」。
- **AI 审查**：短按钮文案（跳过 / 采用优化 / 确认修改 / 完成审查）；侧栏 **绿/红/灰状态点**（`reviewState` 一类）。

### 5.5 Overlay

- **主工具栏**（编辑模式）：语言对（无「源/目标」标签）+ 纯文本 `→`（**不要圆圈包箭头**）+ 操作图标统一约 **28×28**：重选、OCR、删除、◎ 组沉浸。
- **角工具栏**：译文侧停靠 ↑↓←→ · 复制 · 有误 · ◎；OCR 侧暂停 · 刷新 · 复制 · ◎；›/‹ 收缩。
- **◎ 沉浸（invisible）**：
  - 按钮默认要**够明显**；开启态用 **accent 高亮（is-on）**，不要用「变淡 is-dim」当唯一反馈。
  - **框隐字显**：去掉背景/描边/阴影；**正文屏幕坐标不得上移**。
  - 实现注意：隐藏标题/badge 时用 **占位保留**（visibility/固定高度），**禁止** `display:none` + 清零 padding 导致文字上跳。
  - **组 ◎**：隐藏选择框轮廓 + 两卡 chrome + 角工具栏；字仍在。

### 5.6 产品原则（改 UI 时别违背）

- 不打断、低干扰、稳定、速度优先。
- Overlay 最多 **3 组**。
- 记忆在本地、用户确认后才长期写入（反馈链路）。

---

## 6. 明确不要做

- 不要把 mock 当产品依赖打包进 exe。
- 不要为「好看」引入新依赖（WebEngine 嵌 HTML 等），除非产品方书面批准。
- 不要用 `as any` / 乱吞异常；Python 侧保持类型与现有测试习惯。
- 不要在修 bug 时顺便大重构翻译管线。
- 不要提交 `logs/`、`poc/results/`、密钥。
- 用户未说「提交/push」时不要 git commit。

---

## 7. 建议落地顺序（给实现 AI）

1. **主窗 QSS 对齐 token**（壳 + 侧栏 + 按钮主次）  
2. **翻译模板**：语言行 + ⇄ 对齐  
3. **优化翻译页**：对照上下布局 + 四框高度  
4. **弹窗**：Tab / 审查状态点（代码已有部分，对齐 mock）  
5. **Overlay**：主工具栏 → 角工具 → ◎ 布局不跳  

每步：方案 → 用户点头（若工作流要求）→ 改代码 → 跑相关 `tests/`（如 `test_gui_shell.py`）。

---

## 8. 验收清单（最短）

- [ ] 打开 mock Phase1–4，真窗与 mock **同结构**（不是像素级抄 HTML）
- [ ] 优化页 OCR/译文上下整行，高度约 90px 级可用
- [ ] ⇄ 与语言 select 底对齐
- [ ] ◎ 开启后文字位置不跳；按钮开启态可见
- [ ] 主按钮每页唯一；危险操作用危险色
- [ ] 模型下拉仍来自 `/models`
- [ ] 相关单测通过

---

## 9. 协作与文档

| 文档 | 用途 |
|------|------|
| `docs/communication/00-项目认知快照.md` | 项目全局认知（新会话第一站） |
| `docs/goat/14-GUI设计规范.md` | 正式 token / 组件规格 |
| `docs/goat/10-项目进度.md` | 偏旧，以代码 + history 为准 |
| `docs/history/2026/07/` | 近期改动日记 |

更新本 mock 或落地完成后：在 history 记一笔；重大 GUI 决策可补 13-决策记录。

---

## 10. 维护约定

- 改视觉：先改 mock，再改 PySide6（或同步 PR 里说明差异）。
- 改决策：更新 **本 README §5** 与 **`implementer.html`**，避免下一任 AI 读过期假设。
- Mock 交互只服务演示；**真实行为以 `app/` 为准**。
