# 项目当前状态

> 最后更新：2026-07-16

---

## 📊 整体进度

**完成度**：~85%

| 模块 | 状态 | 说明 |
|------|------|------|
| OCR 引擎 | ✅ 完成 | PaddleOCR + Tesseract fallback |
| 截图和变化检测 | ✅ 完成 | BLAKE2b 哈希，500ms 轮询 |
| 翻译服务 | ✅ 完成 | Agent 化，按组独立 |
| GUI 界面 | ✅ 完成 | 4 个生产页面（TemplatePage、模型、反馈、设置）；旧 LanguagePage/PromptPage 已清理 |
| 选择框和悬浮窗 | ✅ 完成 | 支持编辑模式 |
| Prompt 编译系统 | ✅ 完成 | 三层架构 + 优化器 |
| 打包 | ✅ 完成 | Python 3.12 + PyInstaller onedir，发布前执行冻结版 OCR 烟测 |
| 性能优化 | 🔧 持续改进 | 精确 OCR 变体缓存、去重逻辑 |

> 当前状态以本文档和 `docs/goat/09-dev-environment.md` 为准；下方较早的性能、测试数量和启动命令记录是历史快照。

## ✅ 本轮正确性与发布阻断修复（2026-07-16）

- `settings.json` 使用 `utf-8-sig` 读取，同时兼容普通 UTF-8 与 UTF-8 BOM；DPAPI、旧明文迁移和原子保存路径保持不变。
- Agent Thinking 重试日志已区分“返回 Fast best-effort”和“拒绝并重新抛出”，Fast 的 `validation_reason` 仍用于 best-effort 判断。
- 翻译服务增加按组、精确文本、TTL 与容量受限的 OCR 变体缓存；语言、模型、Prompt、选择组重置时失效，并保留 pending/generation/stale 保护。
- 仅过滤整体结构明确的 UI-only 噪声；被过滤日志只保留长度和摘要，不记录 OCR 原文。
- 已确认无运行时、导入或测试引用后删除旧 `LanguagePage` 与 `PromptPage`，生产路径继续使用异步 `TemplatePage`。
- 新增 Python 3.12 锁文件 `requirements-py312.lock`，发布 CI 先安装锁定依赖，再以 `--no-deps --no-build-isolation` 安装项目。
- FeedbackStore 增加线程安全的知识 revision；反馈规则、纠错和后台归纳的持久化成功后会使 OCR 变体缓存失效，保存失败不会推进 revision。
- `TranslationResult` 区分 `api` 与 `cache` 来源，缓存恢复日志不再伪装成 API success；CJK/Kana 空调用不再被泛化过滤。
- 当前用户 Win+R 入口已绑定到工作区 `.venv312\Scripts\instant-translate.exe`，启动日志记录 Python、解释器和源码根路径。
- 当前 spec 保持完整 PaddleOCR 收集逻辑；未为消除可选 PSE 警告进行高风险裁剪。

## 📦 当前发布基线

- Python：发布环境为 3.12.10；源码公开版本范围为 `>=3.10,<3.13`，不支持 Python 3.9。
- 配置密钥：API Key 使用 Windows DPAPI 加密保存；旧明文配置只在迁移时读取并原子写回，不削弱用户隔离。
- 冻结版 OCR：本轮使用 `.venv312` 完成 clean PyInstaller 构建，`--smoke-ocr` 约 6.4 秒退出码 `0`，普通启动也保持运行。
- GUI：生产配置路径为异步 `TemplatePage`、模型、反馈和设置页面；`LanguagePage`、`PromptPage` 已确认无引用并删除。

---

## 🚀 最近完成（2026-06-12）

### 1. Agent 按组独立会话
- 每组独立 Agent 实例，不同语言对不互相污染
- 语言对切换时自动重置 Agent
- 删除组时清理内存

### 2. GUI 重构 - 翻译模板页
- 合并"翻译方向"+"提示词" → "翻译模板"
- 侧边栏 4 页 → 3 页
- 使用 QScrollArea，内容可滚动

### 3. **重大 Bug 修复：DeepSeek API 返回空翻译** ✅
- **问题**：content 为空，只有 reasoning
- **根因**：max_tokens=1024 太小，reasoning 占满预算
- **解决方案**（4 项）：
  1. max_tokens: 1024 → 4096
  2. 检测绝对化约束并改写
  3. 移除 reasoning_content fallback
  4. Messages compact 保留最近 10 轮
- **验证**：POC 测试 8/8 场景通过
- **详见**：`docs/history/2026/06/12-summary.md`

### 4. 日志系统改造
- 按日期分文件夹：`logs/<type>/YYYY/MM/DD.log`
- 同天多次运行自动编号：`DD-2.log`, `DD-3.log`

### 5. 文档结构优化
- Communication 清理（只保留当前问题）
- History 按日期归档（`YYYY/MM/DD.md`）

---

## 🐛 已知问题

### 高优先级

1. **翻译速度慢**（新发现 2026-06-12）
   - 现象：单次翻译耗时 5-30秒，最慢达到 27秒
   - 根因分析：
     - Messages 累积（当前保留 10 轮）
     - Thinking 模式本身慢
     - max_tokens=4096 增加处理时间
   - 可能方案：
     - 减少 MAX_HISTORY_PAIRS (10 → 3-5)
     - 添加"快速翻译"指令
     - 切换到 Flash 模型
     - 降低 max_tokens (需测试是否导致 content 为空)
   - 状态：待决策和测试
   - 详见：`docs/history/2026/06/12.md`

### 中优先级

2. **OCR 反复初始化**（诊断中）
   - 现象：启动后初始化 16 次，每次 2-5s
   - 位置：`app/ocr/engine.py`
   - 诊断：已添加日志，待查看 `instance_id`

3. **大段文本卡顿**
   - 根因：PaddleOCR 处理长文本慢
   - 缓解：已添加去重逻辑

4. **Agent 可能输出历史翻译汇总**
   - 现象：快速切换选择框时，最后可能输出所有历史翻译
   - 根因：DeepSeek 看到历史 messages，认为用户要汇总
   - 影响：低（用户看到的是最新的，历史的被覆盖）
   - 状态：观察中，可能不需要修复

### 低优先级

5. 编辑模式闪烁（已通过状态比对缓解）
6. API 超时后无自动重试
7. 翻译质量评分机制缺失

---

## 🎯 下一步计划

### 立即执行
1. 排查 OCR 重复初始化问题
2. 运行完整测试套件验证最近修改

### 短期（本周）
1. 实现 Agent 上下文 compact
2. 优化长文本处理性能
3. 添加更多单元测试

### 中期（下周）
1. 用户反馈修正机制（✓/✗ 按钮）
2. 术语索引提取（从知识引用自动提取）
3. 翻译质量监控

详见：`docs/05-MVP实现计划.md`

---

## 📈 技术指标

### 性能
- **首次翻译**：5-10s（消化规则，thinking 模式）
- **后续翻译**：5-30s（当前有性能问题，见已知问题）
- **OCR 识别**：50-300ms
- **变化检测**：10-30ms

### 资源
- **内存占用**：~300MB（运行时）
- **打包大小**：~720MB（包含 PaddleOCR 模型）
- **测试覆盖**：Python 3.12 全量 pytest：705 passed、1 skipped、179 subtests；验收同时包含 `pip check`、`compileall`、干净导入和 `git diff --check`

### API 配置
- **max_tokens**：4096（已优化，避免 content 为空）
- **temperature**：0.0（确定性输出）
- **thinking**：True（DeepSeek 推理模式）
- **messages compact**：保留最近 10 轮

---

## 🔧 技术债务

### 已识别
1. **翻译速度优化**（高 - 新增）
   - 需要测试不同方案的速度/质量平衡
   - 可能需要切换模型或调整参数
2. OCR 引擎实例复用（高）
3. 错误恢复机制（中）
4. GUI 响应性优化（低）

### 已解决
1. ~~Agent 上下文无限增长~~ ✅（已实现 compact）
2. ~~API 返回空翻译~~ ✅（已修复 max_tokens）

### 重构候选
1. 将 `SelectionManager` 拆分（过大，800+ 行）
2. Prompt 编译系统独立为库
3. OCR 引擎池（多语言预加载）

---

## 📚 相关文档

- **设计**：`docs/01-14*.md`
- **待办**：`docs/12-待解决问题清单.md`
- **决策**：`docs/13-决策记录.md`
- **日志**：`docs/history/`

---

## 🔍 快速诊断

### 应用无法启动
```bash
# 检查 Python 版本
python --version  # 发布环境应为 3.12.x；源码边界为 >=3.10,<3.13

# 检查依赖
pip list | grep -E "PySide6|paddleocr"

# 查看日志
cat logs/pipeline.log
```

### 翻译不工作
1. 检查 API 配置（主窗口 > 模型配置）
2. 点击"测试连接"验证 API
3. 查看 `logs/debug.log` 中的 API 请求日志

### OCR 识别错误
1. 检查选择框是否覆盖文字
2. 尝试切换语言（工具栏下拉框）
3. 查看 `logs/pipeline.log` 中的 OCR 输出

---

**如有疑问，请查阅 `docs/communication/01-交接文档.md`**
