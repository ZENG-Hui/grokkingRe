# HEARTBEAT.md

## 1. Context 健康检查
- 调用 `session_status` 检查 context 使用率
- **> 70%**：用 `message` 通知 Hui 当前百分比，确认关键信息已持久化
- **> 85%**：用 `message` 紧急通知 Hui，建议发 `/compact`
- compact 是系统自动触发的（~175K），但 Hui 可以随时手动 `/compact`

## 2. Sub-agent 状态检查
- 调用 `subagents list` 检查有没有活跃的 sub-agent
- 如果有：用 `message` 给 Hui 发简短状态更新（什么在跑、跑了多久）
- 如果没有且最近有完成的：检查结果是否已汇报给 Hui

## 3. 进度持久化检查
- 有没有已完成但未记录的任务？→ 更新 progress 文件
- 有没有重要决策/指导只存在于 context 里？→ 写入文件
- 今天的 `memory/YYYY-MM-DD.md` 是否及时更新？
