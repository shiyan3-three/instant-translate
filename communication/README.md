# Communication 目录说明

> AI 协作的对话记录和任务管理

---

## 📁 目录结构

```
communication/
├── README.md           # 本文件，目录导航
├── jiaojie.md         # 给技术专家 AI 的项目交接文档
├── react.md           # 技术专家 AI 的分析反馈
├── huifu.md           # 项目方对技术专家反馈的回应
├── task.md            # 给执行 AI 的任务指令
└── leadersession/     # 架构师 AI (Kiro) 的上下文记忆
    ├── context.md     # 当前会话上下文和角色定位
    └── memory.md      # 跨会话的关键决策和经验
```

---

## 🔄 对话流程

```
1. jiaojie.md  →  [技术专家 AI]  →  2. react.md
                        ↓
3. huifu.md  ←  [项目方评审]
                        ↓
4. task.md  →  [执行 AI]  →  开始干活
                        ↓
              [架构师 AI (Kiro) 监督]
```

---

## 📄 文件说明

### jiaojie.md
- **用途**：给新接手的技术专家 AI 看的项目交接文档
- **内容**：项目概述、技术栈、核心矛盾、待解决的 5 个问题
- **更新时机**：有新的关键问题需要外部 AI 评审时

### react.md
- **用途**：技术专家 AI 对项目的分析和建议
- **内容**：5 个问题的详细分析、风险评估、修复方案、优先级建议
- **更新时机**：技术专家 AI 完成分析后、或读完 huifu.md 有补充时

### huifu.md
- **用途**：项目方对技术专家反馈的回应
- **内容**：采纳/不采纳的方案、原因、下一步行动
- **更新时机**：项目方评审完 react.md 后

### task.md
- **用途**：给执行 AI 的明确任务指令
- **内容**：4 个具体任务、执行规则、验收标准
- **更新时机**：根据 huifu.md 的决策生成任务时

### leadersession/
- **用途**：架构师 AI (Kiro) 的记忆和上下文，用于跨会话恢复状态
- **内容**：角色定位、当前任务、关键决策、待办事项
- **更新时机**：每次会话结束前

---

## 👥 AI 角色分工

| 角色 | AI | 职责 |
|------|-----|------|
| 产品经理 | 用户 | 提出需求、最终决策 |
| 架构师 (Team Lead) | Kiro (Claude Code) | 技术架构、任务分配、代码审查 |
| 技术顾问 | 外部 AI | 第三方视角评审、指出盲区 |
| 执行工程师 | 另一个 AI | 按任务指令实现代码 |

---

## 🔄 使用流程

### 启动新会话
1. 架构师 AI 读 `leadersession/context.md` 恢复状态
2. 读 `leadersession/memory.md` 了解历史决策
3. 继续监督执行 AI 或处理新问题

### 需要外部评审时
1. 更新 `jiaojie.md` 添加新问题
2. 让技术顾问 AI 读 `jiaojie.md` + `plans/` 生成 `react.md`
3. 项目方评审后写 `huifu.md`
4. 架构师 AI 根据 `huifu.md` 生成 `task.md`

### 执行任务时
1. 执行 AI 读 `task.md`
2. 按顺序完成任务，更新 `progress.md`
3. 卡住了写 `blocked.md`，有疑问写 `question.md`
4. 架构师 AI 响应 blocked/question，代码审查

---

## 📝 注意事项

- 所有 AI 的输出都在 `communication/` 下，不污染项目根目录
- `leadersession/` 只给架构师 AI 用，其他 AI 不读
- 对话文件只增不删，保留完整决策链路
- 任务文件 (progress/blocked/question) 用完可以归档到 `leadersession/archive/`
