# Communication 目录

> AI 协作的对话记录和任务管理

```
communication/
├── README.md       # 本文件
├── jiaojie.md     # 给技术专家 AI 的问题
├── react.md       # 技术专家 AI 的回复
├── huifu.md       # 项目方回应 + 审查结论
└── task.md        # 当前待执行任务（执行 AI 读这个）
```

## 流程

```
jiaojie.md → [技术专家 AI] → react.md
    ↓
huifu.md ← [项目方评审]
    ↓
task.md → [执行 AI] → 代码
    ↓
审查 AI 跑测试验收
```

## 角色

| 角色 | 职责 |
|------|------|
| 用户 | 需求、决策 |
| 审查 AI | 跑测试、审代码、验收 |
| 执行 AI (Kiro + OpenCode) | 读 task.md → 写代码 |
