# 最终 Agent 生命周期 POC：Pro / Flash 共享同一 Session

这轮验证完整闭环，而不是继续比较静态提示词。

## 事件序列

```text
Pro 启动并写入可见 handoff
→ Flash 翻译
→ 用户确认纠正
→ 无刷新 handoff 的 Flash 控制分支
→ Pro 阅读完整历史并刷新 handoff
→ Flash 使用“原始纠正 + 新 handoff”翻译相似句
→ 切换到 Pro 思考模式翻译下一句
→ 切回 Flash 翻译带否定的新句
```

所有主分支调用都从磁盘重新加载同一份 Session。Session 只保存可见
`role/content` 消息和事件元数据，不保存 `reasoning_content`。

控制分支与主分支拥有相同原始用户纠正，但控制分支看不到刷新后的 Pro handoff，
因此可以避免把原始历史本身的效果错误归因给 Pro。

## 数据

使用本机真实用户确认的“测试语境中，跑完应使用实施/执行表达”反馈。

后续未见句依次覆盖：

- 回归测试已跑完
- 性能测试已跑完
- 安全测试尚未跑完（额外检查否定是否在 Pro→Flash 切换后保留）

## 正式运行

```powershell
.\.venv\Scripts\python.exe poc_agent_lifecycle.py
```

正式运行调用：

- Pro：3 次（启动 handoff、反馈后刷新、思考模式翻译）
- Flash：4 次（初始翻译、无刷新控制、共享 handoff、Pro 后继续翻译）

运行者只能执行一次并原样返回完整终端输出、`Results` 和 `Shared session` 路径。
不得修改文件、执行 Git、分析、修复、重试或再次运行。

## 离线干跑

```powershell
.\.venv\Scripts\python.exe poc_agent_lifecycle.py --dry-run
```

干跑生成并持久化完整 15 条消息 Session，但不调用 API。

## 通过条件

- Pro 刷新 handoff 明确包含用户确认的“じっし/じっこう”偏好
- 带刷新 handoff 的 Flash 在相似新句中遵守偏好
- 思考模式 Pro 在同一 Session 中继续遵守偏好
- 切回 Flash 后仍遵守偏好，并保留“尚未完成”的否定
- 所有展示译文通过统一格式检查
- Session 至少经过 7 次保存后重载，最终 handoff 版本为 2
- 持久化消息中不存在 `reasoning_content`

如果无刷新控制分支和刷新 handoff 分支质量相同，则说明原始纠正历史已经足够，
Pro handoff 主要承担会话管理、冲突解决和压缩，而不是凭空提升每一句翻译。
