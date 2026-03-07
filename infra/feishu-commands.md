# 飞书命令速查

> 在飞书私聊中直接发送以下命令，由 OpenClaw Gateway 拦截处理，不经过模型。

## Session 控制

| 命令 | 说明 |
|------|------|
| `/stop` | 中断当前 run。也支持发 `停止`、`stop`、`abort` 等自然语言 |
| `/new` | 开启新 session，清空对话历史。AGENTS.md / SOUL.md 等重新注入 |
| `/reset` | 同 `/new` |
| `/compact [指令]` | 手动触发 context 压缩。可附带指令，如 `/compact 保留项目进度相关内容` |

## 状态查询

| 命令 | 说明 |
|------|------|
| `/status` | 当前 session 状态（模型、token 用量、heartbeat 等） |
| `/context` | context 使用详情（token 数、压缩次数等） |
| `/subagents` | 列出当前 session 的 sub-agent 及状态 |
| `/models` | 列出所有可用模型。`/models <provider>` 按 provider 筛选 |
| `/help` | 显示所有可用命令 |

## 调试 & 管理

| 命令 | 说明 |
|------|------|
| `/config` | 查看当前配置（需 `commands.config=true`，已开启） |
| `/debug` | 显示调试信息（需 `commands.debug=true`，已开启） |
| `/export` | 导出当前 session 对话记录 |

## Shell 执行

| 命令 | 说明 |
|------|------|
| `!<命令>` | 在容器内直接执行 shell 命令，如 `!ps aux`、`!ls /tmp/openclaw/` |
| `/bash <命令>` | 同上 |

需 `commands.bash=true`（已开启）。命令在容器内执行，受 kill wrapper 保护（无法杀 openclaw 进程）。超过 2 秒自动转为后台执行。

## HG 远程 OpenCode

通过 Teneral 透传命令到 HG 开发机上的 OpenCode。以 `>>` 开头的消息进入透传模式，Teneral 原样转发 prompt、原样回传结果，不做任何加工。

| 命令 | 说明 |
|------|------|
| `>> <prompt>` | 新 session 执行 |
| `>> #name <prompt>` | 命名 session 执行（自动复用，保持上下文） |
| `>> --sessions` | 列出所有命名 session |
| `>> --rm <name>` | 删除 session 映射 |

示例：
```
>> 读一下 main.py 的前 20 行
>> #thesis 改一下摘要第 3 段
>> #thesis 编译看看有没有错
```

不加 `>>` 前缀则走委派模式（Teneral 理解意图后自行处理）。

## 中断时机

- **模型正在推理/执行工具**：发 `/stop` 立即中断，session 恢复 idle，上下文不丢失
- **exec 卡住**：发 `/stop` 中断，比重启 gateway 轻量
- **Gateway 进程死亡**：命令无法送达，需从宿主机重启（看门狗会自动告警）

## `/stop` 触发词完整列表

除了 `/stop`，以下自然语言也能触发中断（不区分大小写）：

```
stop, esc, abort, wait, exit, interrupt, halt, please stop, stop please
停止, やめて, 止めて, остановись, arrête, anhalten, stopp, pare
```
