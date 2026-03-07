# HG OpenCode 远程控制 — 操作手册

_Created 2026-03-07, Updated 2026-03-07_

## 两种模式

### 1. 透传模式（`>>` 前缀）
- Hui 在飞书发 `>> <指令>` → 我转发给 HG opencode → 结果原样返回
- **我是纯管道，不做任何加工**：不总结、不润色、不加自己的话
- 支持 `#session_name`：`>> #thesis 改一下第15行`
- 支持 `>> --sessions`（列出映射）和 `>> --rm <name>`（删除映射）
- 完整规则在 AGENTS.md "HG OpenCode 透传模式" 章节

### 2. 委派模式（我来管理）
- Hui 用自然语言描述任务 → 我理解意图 → 调 /opencode → 审查结果 → 回传
- 复杂任务我可以拆分、迭代、审查代码质量
- 我申请用哪种模式，Hui 决定

## API 调用规范

### /opencode 端点
- URL: `https://yinghuo-hg.deepseek.com/zenghui/dev-cpu/agent-server/opencode`
- 代理: `http://172.29.4.175:22222`
- Auth: `Bearer T1moNCxHPLy4VpVlMFJlDpjPVtUT-tezJCSp0RzUmGo`

### 请求格式
```json
{
  "message": "指令内容",
  "session": "ses_xxx（可选，复用已有 session）"
}
```

**⚠️ 注意：**
- 字段名是 `session`（不是 `session_id` 也不是 `session_name`）
- **不要用 `session_name` 字段** — HG 端映射有 bug，会重复创建
- 映射逻辑在本地 `memory/hg-sessions.json` 维护

### 响应字段
- `ok` — 是否成功
- `session_id` — opencode 的 session ID（`ses_xxx`）
- `text` — 文本回复
- `warnings` — ⚠️ 每次必须检查
- `events` — 执行步骤详情
- `code` — 退出码
- `stderr` — 错误输出

### Session 管理端点
- `POST /sessions/list` — 列出所有 session（含名字、创建时间、使用次数）
- `POST /sessions/info` — 查看某 session 详情
- `POST /sessions/rename` — 重命名 session
- `POST /sessions/delete` — 删除 session
- `GET /sessions/cleanup` — 清理 30 天以上旧 session

## Session 复用策略（⚠️ 必须遵守）

**按项目/模块划分，一个 session 从头用到尾：**
```
teneral-thesis-chap01   ← 第一章所有编辑
teneral-thesis-chap03   ← 第三章所有编辑
teneral-thesis-compile  ← 所有编译任务
teneral-grokkingRe      ← grokkingRe 项目所有工作
```

- 新开 session 前先查 `memory/hg-sessions.json`，有没有能复用的
- 每次创建新 session 立即记录映射
- **并发限制：5**，不要同时发太多请求
- Hui 通过 `!hg` 创建的命名 session 和我的是独立的

## 安全检查清单
每次 /opencode 响应后必须：
1. 检查 `warnings` 数组 — 非空则立即停止并通知 Hui
2. 检查 `code` — 非 0 说明有执行错误
3. 检查 `stderr` — 有内容则留意
4. 审查 `text` 中的代码修改是否合理

## 省 token 原则
- 能用 `/ls`、`/cat`、`/run` 解决的，**不用 /opencode**
- opencode 每次调用 ~13K+ input tokens
- 只在需要编码判断力时用 opencode
