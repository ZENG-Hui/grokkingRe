# HEARTBEAT.md

**⛔ 你必须按顺序执行以下每一步。不允许跳过任何步骤。不允许在执行完所有步骤之前回复 HEARTBEAT_OK。**

## 步骤 1：调用 session_status
- 检查 context 使用率
- 如果 > 70%：记住需要在步骤 3 中告知 Hui

## 步骤 2：调用 subagents list
- 这一步**必须执行**，不管步骤 1 的结果如何
- 记录：有没有活跃的 sub-agent？有没有最近完成的？

## 步骤 3：调用 message 工具发送 💓 状态给 Hui
- 这一步**必须执行**（除非满足下面的唯一豁免条件）
- 消息以 💓 开头，内容包含：
  - context 百分比（如果 > 70% 则标注预警）
  - sub-agent 状态（什么在跑/完成了/空闲）
- **唯一豁免条件：** 没有活跃 sub-agent、没有新完成的 sub-agent、context < 70%、且 `memory/heartbeat-state.json` 中 `idleNotified` 为 `true`。此时可以跳过 message，直接回复 HEARTBEAT_OK。
- 如果是第一次空闲（idleNotified 不是 true），必须发送 💓 空闲通知，然后写 idleNotified=true。

## 步骤 4：进度持久化（仅在 context > 70% 或有任务完成时）
- 更新 progress 文件
- 写入未持久化的决策/指导

## 步骤 5：回复 HEARTBEAT_OK
