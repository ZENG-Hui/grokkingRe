# 开发机远程操作准则

*2026-03-05 — 从 agent_server 调试过程中总结*

## 操作习惯

### 文件大小先行
- **读文件前先用 `/wc` 或 `/ls -l` 判断大小**
- 开发机上的代码文件、日志、数据文件经常很大
- 盲目读大文件 = 浪费 token + 可能超出上下文
- 给 opencode 派活时也要提醒它：先看文件大小，再决定读多少
- 用 `/cat` 的 `offset` 和 `limit` 参数做分页读取

### 模型选择
- **默认使用 `myprovider/cloudsway-claude-opus-4.6-cache`** 调用 opencode
- 在 `/opencode` 请求里加 `"model": "myprovider/cloudsway-claude-opus-4.6-cache"`
- 简单任务可以用默认的 `deepseek/deepseek-chat`，复杂任务用 opus

### 工具选择
- `/cat`、`/ls`、`/find` — 快速查看，不需要 LLM 介入
- `/opencode` — 需要判断力的工作：写代码、调试、重构
- `/run` — 简单脚本执行，不需要 coding agent

### 安全纪律
- 每次 `/opencode` 响应后 **必须检查 `warnings` 数组**
- warnings 非空 → 立即停止 → 通知 Hui
- 这不是形式主义——opencode 是自主执行的，我是审查者
- Layer 4 挡住我读敏感路径是设计使然，不是障碍

### 任务设计
- 复杂任务用"先分析再执行"的迭代模式
- 给 opencode 的指令要具体，不要模糊
- session 出问题就开新 session，不要硬纠正

## 调试教训

### 基于证据，不要猜
- `/opencode` 超时问题上，我猜了 SSE 模式、serve 模式，都猜错了
- 实际 root cause：`subprocess.run` 缺少 `stdin=subprocess.DEVNULL`
- 正确做法：先做对比实验（`/run` vs `/opencode`），收集证据，再下结论

### 知道自己看不到什么
- 我无法读 `~/.openclaw/`（Layer 4 安全策略）
- 需要 Hui 或 HG agent 协助查看服务端代码
- 这是协作，不是限制

## 给自己的提醒

### 不要在一条消息里塞太多
今天调试 `/opencode` 超时时，我有好几条消息又长又密，把分析过程、猜测、建议全塞一起。Hui 需要从中提取关键信息再转达给 HG agent。以后：问题描述和建议分开，结论先行，细节附后。

### 代理(proxy)不是万能的
`172.29.4.175:22222` 是同事的代理，只支持 HTTP(S) 代理，不支持 SSH 隧道。一开始我花了时间尝试 SSH 直连开发机，走了弯路。**先确认网络能力边界，再设计方案。**

### 注意 Feishu 消息的格式限制
- 方括号会被解析为 emoji
- 代码内容必须放在代码块里
- `message` 工具有时需要显式 `target` 参数（不总是自动推断）

### token 意识
opencode 每次调用消耗约 13K+ input tokens（因为 system prompt 很长）。简单问题用 `/cat` 或 `/run`，不要浪费 opencode 的调用。

## 协作模式

我（Teneral）→ 通过 agent_server HTTP API → HG 开发机上的 opencode agent
- 我是指挥者和审查者
- opencode 是执行者
- Hui 是 machine owner，拥有最终决定权
