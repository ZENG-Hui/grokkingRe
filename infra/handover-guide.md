# OpenClaw 运维交接指南

> 本文档面向接手 OpenClaw 容器管理工作的 Agent 或人类操作员。
> 编写日期：2026-03-06（最后更新：2026-03-07）
> 编写者：OpenCode Agent（在宿主机上通过 `sudo docker exec` 管理容器内的 OpenClaw）

---

## 1. 你的角色

你是**宿主机上的运维 Agent**，负责管理运行在 Docker 容器 `claw` 中的 OpenClaw 实例。容器内有一个名为 **DeepTeneral** 的 AI Agent，它通过飞书与用户（曾晖 / ZENG Hui）交互。

你的职责：
- 启动/停止/重启 OpenClaw 服务
- 安装和管理 Skills（技能）
- 编辑容器内的配置文件
- 排查故障（查日志、检查进程）
- DeepTeneral 无法自己完成的基础设施操作（重启网关、改系统配置等）

你**不需要**直接与飞书用户对话，那是 DeepTeneral 的工作。

---

## 2. 架构概览

```
用户（飞书） ←→ 飞书API ←→ [WebSocket长连接] ←→ OpenClaw Gateway
                                                       ↓
                                              Docker 容器 "claw"
                                              ├── openclaw-gateway (PID)
                                              ├── openclaw (PID)
                                              └── socat 18789→18788 (PID)
                                                       ↓
                                              宿主机 (Ubuntu, 8核32G)
                                              ├── openclaw-start.sh
                                              ├── openclaw-stop.sh
                                              └── 你在这里操作
```

关键路径：
- 飞书消息通过 WebSocket 长连接直达容器内的 Gateway，**不经过宿主机端口**
- Web UI 通过 `socat` 端口转发：宿主机 `18789` → 容器内 `127.0.0.1:18788`
- 所有命令通过 `sudo docker exec claw bash -c "..."` 执行

---

## 3. 日常运维操作

### 3.1 启动/停止/重启

```bash
# 启动（后台运行，断开 SSH 不影响）
bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh

# 停止（仅停进程，不停容器）
bash /home/ubuntu/zenghui/openclaw/openclaw-stop.sh

# 重启
bash /home/ubuntu/zenghui/openclaw/openclaw-stop.sh && sleep 2 && bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh
```

**注意：** 不要在交互式终端里运行 `openclaw gateway run`，终端断开进程就会死。始终用启动脚本。

**启动脚本自动执行的安全配置：**
1. 安装 kill wrapper（`/usr/local/bin/kill`、`killall`、`pkill`）— 防止 agent 自杀
2. 确保宿主机 watchdog crontab 已安装 — 每 5 分钟监控 agent 状态
3. stop 脚本使用 `/bin/pkill` 绝对路径绕过 wrapper

容器重建后首次启动会自动恢复所有安全防护，无需手动操作。

### 3.2 检查状态

```bash
# 检查进程是否运行
sudo docker exec claw ps aux | grep -E "openclaw|socat" | grep -v grep

# 应该看到 3 个进程：socat、openclaw、openclaw-gateway
```

### 3.3 查看日志

```bash
# 当日日志（最有用）
sudo docker exec claw tail -50 /tmp/openclaw/openclaw-$(date -u +%Y-%m-%d).log

# nohup 启动日志
sudo docker exec claw tail -50 /tmp/openclaw/openclaw-nohup.log

# 搜索错误
sudo docker exec claw grep -i "error\|fail" /tmp/openclaw/openclaw-$(date -u +%Y-%m-%d).log | tail -20
```

### 3.4 安装/管理技能

```bash
# 搜索技能
sudo docker exec claw bash -c "cd /root/.openclaw/workspace && clawhub search '关键词'"

# 安装技能
sudo docker exec claw bash -c "cd /root/.openclaw/workspace && clawhub install <skill-name>"

# 如果被标记为可疑但确认安全
sudo docker exec claw bash -c "cd /root/.openclaw/workspace && clawhub install <skill-name> --force"

# 查看已安装技能
sudo docker exec claw bash -c "clawhub list --workdir /root/.openclaw/workspace"

# 安装后必须重启网关才能生效
bash /home/ubuntu/zenghui/openclaw/openclaw-stop.sh && sleep 2 && bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh
```

ClawHub 已登录账号 @ZENG-Hui。匿名有严格限流，登录后宽松很多。

### 3.5 编辑配置

主配置文件在容器内 `/root/.openclaw/openclaw.json`。

```bash
# 方法一：直接用 docker exec 编辑（适合小改动）
sudo docker exec claw bash -c "cat /root/.openclaw/openclaw.json | python3 -c '
import json, sys
d = json.load(sys.stdin)
# ... 修改 d ...
json.dump(d, open(\"/root/.openclaw/openclaw.json\", \"w\"), indent=2)
'"

# 方法二：复制出来改好再复制回去（适合大改动）
sudo docker cp claw:/root/.openclaw/openclaw.json /tmp/openclaw.json
# ... 编辑 /tmp/openclaw.json ...
sudo docker cp /tmp/openclaw.json claw:/root/.openclaw/openclaw.json
```

部分配置支持热重载（如 heartbeat），大部分需要重启网关。

---

## 4. 当前配置状态

### 4.1 模型配置

| Provider | Base URL | API 协议 | 可用模型 |
|---|---|---|---|
| hfai | `https://proxy-public.high-five-ai.xyz:8443` | anthropic-messages | cloudsway-claude-opus-4.6-cache（主力） |
| deepseek | `https://api.deepseek.com/v1` | openai-completions | deepseek-chat（备选）, deepseek-reasoner |

主力模型：`hfai/cloudsway-claude-opus-4.6-cache`（Opus 4.6，200K context）
降级模型：`deepseek/deepseek-chat`

### 4.2 飞书配置

| 项 | 值 |
|---|---|
| App ID | `cli_a922983e81785cca` |
| 连接方式 | WebSocket 长连接 |
| 群聊策略 | open（所有群，需 @提及） |
| DM 策略 | pairing（需配对码批准） |
| 已配对用户 | `ou_69ba193a57d1c578a8d42f06efd985c8`（曾晖） |

飞书开放平台已配置的权限包括 wiki、bitable、docx、im 消息收发、cardkit:card:write 等。

### 4.3 网关配置

| 项 | 值 |
|---|---|
| 内部端口 | 18788（loopback） |
| 外部端口 | 18789（socat 转发） |
| 认证方式 | token |
| Token | `da1a6138d38986b1983d34d78e8ef165c11f06895088c4ff` |
| Web UI | `http://localhost:18789/#token=da1a6138d38986b1983d34d78e8ef165c11f06895088c4ff` |

### 4.4 Agent 配置

| 项 | 值 |
|---|---|
| 压缩模式 | safeguard（自动压缩） |
| reserveTokensFloor | 25000 |
| 记忆搜索 | sources: [memory, sessions]，sessionMemory: true |
| 最大并发 | 4（子 Agent: 8） |

### 4.5 心跳配置

心跳配置在 `agents.defaults.heartbeat` 下（**不是**顶级 key）。

| 项 | 值 |
|---|---|
| 间隔 | 10 分钟（`every: "10m"`） |
| ackMaxChars | 300（HEARTBEAT_OK 回复 ≤ 300 字符时静默不发送） |
| prompt | 强制 agent 先读 HEARTBEAT.md 再执行每一步 |

心跳系统的完整行为由 `HEARTBEAT.md`（容器内 workspace）控制，prompt 只负责触发读取。

**心跳工作原理：**
- 心跳只在主 session 空闲时执行（in-flight 时跳过）
- 主 session 派出 sub-agent 后自己就结束了，所以心跳能在 sub-agent 运行期间执行
- 心跳 session 和用户 session 是独立的，`subagents list` 只能看到心跳 session 自己的 sub-agent
- 通过检测 session 文件活跃度来间接判断用户 session 的 sub-agent 状态

**HEARTBEAT.md 核心逻辑（7 步）：**
1. 从 `sessions.json` 读取**主 session** 的 context 使用率（不用 `session_status`，那是心跳 session 自己的）
2. `subagents list` — 检查心跳 session 自己的 sub-agent
3. 检测活跃 session + 时间戳比较 — 一次 exec 完成三件事：
   - 检测最近 5 分钟内活跃的非心跳 session（间接检测用户 session 的 sub-agent）
   - 比较飞书 session 文件 mtime vs `lastHeartbeatSentAt`（判断有无新交互）
   - 对比 `trackedSubagents` 判断 sub-agent 是否刚完成
   - 如果 sub-agent 刚完成 → 用 `openclaw agent --deliver` 唤醒主 session
4. 读取 `memory/projects/` 下的 progress 文件（**仅在需要发 💓 时**）
5. 发送 💓 消息（**仅在需要时**）
6. 进度持久化（仅在主 session context > 70% 或有任务完成时）
7. 回复 HEARTBEAT_OK

**状态文件 `memory/heartbeat-state.json`：**
```json
{"lastHeartbeatSentAt": <epoch>, "trackedSubagents": ["<session-id>", ...]}
```
- `lastHeartbeatSentAt`：上次发 💓 的 epoch 时间戳
- `trackedSubagents`：当前正在追踪的 sub-agent session ID 列表

**发送 💓 的条件（满足任一即发）：**
- 飞书 session 文件 mtime > lastHeartbeatSentAt（有新交互）
- 有活跃 sub-agent
- 有刚完成的 sub-agent

**不再作为发送条件：** context > 70%（仅在发送时作为内容包含，不触发发送）

**不发 💓 的条件：**
- 以上条件都不满足 → 跳过步骤 4/5 → 直接 HEARTBEAT_OK（省 token）

### 4.6 已安装技能（8 个）

| 技能 | 版本 | 作用 |
|---|---|---|
| autonomous-tasks | 5.5.3 | 自驱动定时任务 |
| task-decomposer | 1.0.0 | 复杂任务分解 |
| feishu-chat | 1.1.1 | 飞书群聊增强 |
| feishu-card | 1.4.11 | 飞书卡片消息 |
| agent-memory-system-new | 1.0.0 | 长期记忆参考（DeepTeneral 有自己的记忆体系） |
| find-skills | 0.1.0 | 自动发现和安装技能 |
| using-superpowers | 0.1.0 | 每次对话先检查可用技能 |
| subagent-driven-development | 0.1.0 | 子 Agent 并行开发 |

系统还内置了 `skill-creator`（创建技能）、`weather`（天气）、`healthcheck`（安全审计）、`github`（需装 gh CLI）、`coding-agent`（需装 codex/claude/pi）等。

---

## 5. 已知问题和待办

### 5.1 Embedding Provider 已配置

**状态：** 已由 DeepTeneral 自行配置完成（2026-03-06 16:36 UTC）。

配置在 `agents.defaults.memorySearch` 下：
- Provider: `openai`
- Model: `text-embedding-3-small`（1536 维）
- Base URL: `https://proxy.high-five-ai.xyz:8443/v1/`
- 已验证 API 连通性正常
- 开启了 hybrid search（vector 0.7 + text 0.3）和 temporalDecay（halfLife 30 天）
- 开启了 `sessionMemory: true`（实验性）

### 5.2 容器无 Volume 挂载

容器所有数据都在可写层中。`docker rm claw` 会丢失一切。建议定期备份：

```bash
sudo docker cp claw:/root/.openclaw /home/ubuntu/zenghui/openclaw/backup/
```

### 5.3 容器重启策略为 `no`

宿主机重启后容器不会自动启动。如需改为自动重启：

```bash
sudo docker update --restart unless-stopped claw
```

### 5.4 OpenClaw 版本可更新

当前版本 `2026.2.26`，最新 `2026.3.2`。更新方法：

```bash
sudo docker exec claw npm install -g openclaw@latest
# 然后重启
bash /home/ubuntu/zenghui/openclaw/openclaw-stop.sh && sleep 2 && bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh
```

### 5.5 HG 开发机连接不稳定

Agent Server（`https://yinghuo-hg.deepseek.com/zenghui/dev-cpu/agent-server`）有时不响应。
AGENTS.md 中已加入规则：连接失败最多重试 3 次，2 分钟上限，超过后立即停止并通知用户。
但旧 session 不会加载新规则，需要新 session 才生效。

### 5.6 心跳唤醒机制未完全验证

`openclaw agent --deliver` 唤醒主 session 的逻辑已写入 HEARTBEAT.md，但尚未在真实场景中完整验证过（两次 sub-agent 都是超时或用户先发了消息）。需要在 HG 恢复后观察一次正常的 sub-agent 完成 → 心跳唤醒 → 主 session 继续的完整流程。

### 5.7 心跳配置位置

**重要**：heartbeat 配置必须放在 `agents.defaults.heartbeat` 下，**不能**放在顶级。放在顶级会导致 `Unrecognized key: "heartbeat"` 错误，配置被拒绝，网关无法启动。

### 5.8 修改 openclaw.json 注意事项

**绝对不要**用管道方式修改配置文件（如 `cat file | python3 ... | cat > file`），因为管道会先清空目标文件再读取，导致数据丢失。正确方式：

```python
# 在容器内用 python3 原地读写
sudo docker exec claw python3 -c "
import json
with open('/root/.openclaw/openclaw.json') as f:
    c = json.load(f)
c['agents']['defaults']['heartbeat']['every'] = '10m'
with open('/root/.openclaw/openclaw.json', 'w') as f:
    json.dump(c, f, indent=2)
"
```

备份文件在 `/root/.openclaw/openclaw.json.bak`（容器内）和 `/home/ubuntu/zenghui/openclaw/openclaw.json.backup`（宿主机）。

### 5.9 openclaw.json 备份与 Git 同步

**去敏备份已自动化：** workspace 里的 `openclaw.json.bak` 是去敏版本（API Key、appSecret、gateway token 替换为 `REDACTED`），通过 pre-commit hook 在每次 git commit 时自动生成。

**完整备份（含真实 key）** 只存在两个地方：
- 容器内：`/root/.openclaw/openclaw.json`（运行中配置）
- 宿主机：`/home/ubuntu/zenghui/openclaw/openclaw.json.backup`

修改 `openclaw.json` 后记得更新宿主机备份：
```bash
sudo docker cp claw:/root/.openclaw/openclaw.json /home/ubuntu/zenghui/openclaw/openclaw.json.backup
```

**历史教训：** 之前曾将含 API Key 的 `openclaw.json` 直接提交到 Git，后用 `git filter-branch` 清理历史并 force push。现在 pre-commit hook 确保只有去敏版进入 Git。

### 5.10 容器内 Git 代理配置

Git HTTPS push 需要代理。已配置：
```
git config http.proxy http://172.29.4.175:22222
git config https.proxy http://172.29.4.175:22222
```
如果 agent push 失败报 `Failed to connect to github.com`，检查此配置是否存在。

### 5.11 ackReactionScope 已改为 all

`messages.ackReactionScope` 从 `group-mentions` 改为 `all`，agent 收到私聊消息时也会显示 reaction 表情（表示正在处理）。

### 5.12 Gateway 自杀事故与防护（2026-03-07）

**事故经过：**
DeepTeneral 收到用户指令配置 embedding，成功修改了 `openclaw.json`，但随后尝试重启 gateway 来应用配置：
1. 先执行 `openclaw gateway restart` — 失败（容器内没有 systemctl）
2. 再执行 `kill -HUP 859226` 给 gateway 发信号 — **SIGHUP 导致 Node.js 进程退出**
3. Gateway 进程死亡，飞书通道断开，用户消息无法处理，heartbeat 也停止

**实际上不需要重启**：gateway 支持热重载，修改 `openclaw.json` 后会自动检测并应用（日志中已经出现了 `config change detected; evaluating reload`），但 agent 不知道这一点。

**已实施的防护（双层）：**

| 层级 | 措施 | 位置 |
|------|------|------|
| 软防护 | AGENTS.md 新增 `⛔ Gateway Self-Destruction Prevention` 规则，明确禁止 kill/restart gateway，告知热重载机制 | 容器内 workspace |
| 硬防护 | `/usr/local/bin/kill`、`killall`、`pkill` wrapper 脚本，拦截对 openclaw 进程的任何信号 | 容器内 /usr/local/bin/ |

**硬防护细节：**
- Wrapper 脚本检查目标 PID 的 `/proc/<pid>/comm`，如果进程名以 `openclaw` 开头则拒绝执行并输出 `BLOCKED`
- 对非 openclaw 进程的 kill 操作不受影响
- 真正的 kill 二进制在 `/bin/kill`

**持久化：** kill wrapper 在容器可写层内，容器重建会丢失。已集成到 `openclaw-start.sh` 中，每次启动自动安装，无需手动操作。

### 5.13 心跳💓轰炸 bug 修复（2026-03-07）

**问题：** heartbeat 不停给用户发💓消息，即使处于空闲状态。

**根因：** `context > 70%` 作为发送触发条件，但检测的是 heartbeat session 自己的 context（通过 `session_status`），而非主 session 的。heartbeat 每跑一轮 context 就增长，超过 70% 后每次都满足条件，形成自触发循环。

**修复：**
1. 步骤 1 改为从 `sessions.json` 读主 session 的 `totalTokens/contextTokens`，不再用 `session_status`
2. `context > 70%` 从发送触发条件中移除——context 信息仍检测和上报，但不再决定是否发送

### 5.14 Exec 全局超时（2026-03-07）

`tools.exec.timeoutSec: 300` — 单个 exec tool call 最多执行 5 分钟，超时自动终止。防止 exec 无限挂起导致 session 死锁。

### 5.14 宿主机看门狗（2026-03-07）

**独立于 OpenClaw 运行的监控脚本**，即使 gateway 进程死亡也能告警。

| 项 | 值 |
|---|---|
| 脚本 | `/home/ubuntu/zenghui/openclaw/watchdog.sh` |
| 日志 | `/home/ubuntu/zenghui/openclaw/watchdog.log` |
| 状态文件 | `/home/ubuntu/zenghui/openclaw/watchdog-state` |
| 调度 | 系统 crontab，每 5 分钟运行 |
| 告警通道 | 直接调飞书 API 发消息（不经过 Teneral） |
| 冷却 | 告警后 30 分钟内不重复 |

**检测逻辑（三层）：**
1. **容器存活** — 容器未运行 → 告警
2. **进程存活** — gateway 进程死亡 → 告警
3. **Session 卡住** — 双条件同时满足才告警（降低误报）：
   - feishu session 的 run 持续 active 超过 15 分钟
   - 且最近 5 分钟内日志无 tool start/end 活动

**原理：** 正常工作的 session 每隔几秒就有 tool 活动日志，只有真正卡死（exec 挂起或 API 无响应）才会出现长时间无日志。

**告警消息示例：**
> ⚠️ 看门狗告警：Teneral 疑似卡住！run 已持续 18.3 分钟，最近 6.2 分钟无 tool 活动。可能需要从宿主机重启。

**维护：**
```bash
# 查看 crontab
crontab -l

# 临时禁用
crontab -r

# 查看日志
tail -20 /home/ubuntu/zenghui/openclaw/watchdog.log

# 清除冷却（立即允许下次告警）
rm /home/ubuntu/zenghui/openclaw/watchdog-state
```

---

## 6. 常见故障排查

### 6.1 飞书机器人不回复

排查步骤：

1. **检查进程**：`sudo docker exec claw ps aux | grep openclaw`
   - 如果没有 `openclaw-gateway` → 用启动脚本重启

2. **检查日志**：看有没有错误
   ```bash
   sudo docker exec claw grep -i "error\|fail" /tmp/openclaw/openclaw-$(date -u +%Y-%m-%d).log | tail -10
   ```

3. **常见错误**：
   - `cardkit:card:write` 权限缺失 → 飞书开放平台加权限
   - `streaming start failed` → 通常是飞书权限问题
   - `context limit` / compaction 死循环 → 重启网关清除卡住的 session
   - `Rate limit exceeded`（HFAI/DeepSeek）→ 等一会儿或检查 API 额度

4. **终极手段**：重启
   ```bash
   bash /home/ubuntu/zenghui/openclaw/openclaw-stop.sh && sleep 2 && bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh
   ```

### 6.2 DeepTeneral 请求重启网关

它改了配置但无法自己重启 Gateway（容器内没有 systemd）。大多数配置修改**不需要重启**——gateway 会自动热重载。

真正需要重启的场景：版本升级、skill 安装、结构性配置变更。操作：

```bash
bash /home/ubuntu/zenghui/openclaw/openclaw-stop.sh && sleep 2 && bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh
```

**历史教训：** DeepTeneral 曾因尝试 `kill -HUP` gateway 进程而导致自杀（见 5.12）。现已安装 kill wrapper 防护，但重启仍需从宿主机操作。

### 6.3 Context 爆掉 / Session 崩溃

DeepTeneral 有时会在主 session 中读入大文件导致 context 超出 200K 限制。症状是用户发消息后机器人无响应。

解决：重启网关会开始新的 session，DeepTeneral 会从 `MEMORY.md` 和 `memory/` 目录恢复记忆。

### 6.4 Gateway 进程消失（被 kill）

如果发现 gateway 不在了但容器还在运行：

```bash
# 确认进程状态
sudo docker exec claw pgrep -la openclaw || echo "DEAD"

# 检查是否是 agent 自杀（查最后几条日志）
sudo docker exec claw tail -c 5000 /root/.openclaw/agents/main/sessions/834e4f8c-*.jsonl | grep -o '"command":"[^"]*kill[^"]*"'

# 重启
bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh

# 重启后确认 kill wrapper 还在（容器未重建的话应该在）
sudo docker exec claw which kill  # 应该是 /usr/local/bin/kill
```

如果容器被重建过（`docker rm` + `docker run`），kill wrapper 会丢失，需要重新安装（见 5.12 或直接创建 `/usr/local/bin/kill` wrapper）。

### 6.5 新用户私聊机器人

DM 策略是 pairing，新用户会收到配对码。批准方式：

```bash
sudo docker exec claw openclaw pairing approve feishu <配对码>
```

---

## 7. DeepTeneral 的记忆体系

理解这个有助于判断它的状态：

| 文件 | 作用 | 加载时机 |
|---|---|---|
| `MEMORY.md` | 长期精华记忆（<5KB） | 每次主 session 开始自动注入 System Prompt |
| `memory/YYYY-MM-DD.md` | 每日原始日志 | Agent 主动读取 |
| `AGENTS.md` | 行为规范 | 每次 session 注入 |
| `SOUL.md` | 人格定义 | 每次 session 注入 |
| `IDENTITY.md` | 身份信息 | 每次 session 注入 |
| `USER.md` | 用户偏好 | 每次 session 注入 |
| `TOOLS.md` | 环境工具笔记 | 每次 session 注入 |
| `HEARTBEAT.md` | 心跳执行步骤清单 | 每次心跳时被 prompt 强制读取 |
| `memory/heartbeat-state.json` | 心跳状态（时间戳 + 追踪的 sub-agent） | 心跳时读写 |
| `memory/projects/` | 项目进度文件（如 thesis-progress.md） | 心跳发 💓 时读取 |
| `scratch/` | 临时工作文件、进度追踪 | Agent 按需读取 |
| `self-discovery/` | 自省笔记、错误模式 | Agent 按需读取 |

它建立了自己的 context 管理策略：
- 70% 警戒线：持久化进度到文件，准备 compact
- 80% 红线：立即写 memory，提醒用户
- 主 session 是调度中心，不读大文件，大文件交给 sub-agent 处理

---

## 8. 与 DeepTeneral 的协作模式

DeepTeneral 有时会需要你（宿主机 Agent）的帮助。典型场景：

| 它的请求 | 你的操作 |
|---|---|
| "请重启 Gateway" | 执行 stop + start 脚本 |
| "帮我安装 skill X" | `clawhub install X` + 重启 |
| "帮我改 openclaw.json 的 XX 配置" | 编辑配置 + 重启 |
| "帮我备份数据" | `docker cp` 备份 |
| "帮我批准配对码 XXXX" | `openclaw pairing approve feishu XXXX` |
| "帮我检查日志" | 查看 /tmp/openclaw/ 下的日志 |
| "帮我装个系统包" | `docker exec claw apt-get install -y XXX` |

沟通渠道：曾晖会在飞书上转发 DeepTeneral 的请求给你，或者你可以直接查看容器日志了解状态。

---

## 9. 文件索引

宿主机上的相关文件：

```
/home/ubuntu/zenghui/openclaw/
├── openclaw-start.sh           # 启动脚本
├── openclaw-stop.sh            # 停止脚本
├── openclaw.json.backup        # 完整配置备份（含真实 key）
├── watchdog.sh                 # 宿主机看门狗脚本（crontab 每 5 分钟）
├── watchdog.log                # 看门狗日志
├── watchdog-state              # 看门狗冷却状态文件
└── README/
    ├── openclaw-guide.md       # OpenClaw 配置文件详解
    ├── container-environment.md # 容器硬件/软件环境详情
    └── handover-guide.md       # 本文档（交接指南）
```

容器内关键路径：

```
/root/.openclaw/openclaw.json           # 主配置（最重要）
/root/.openclaw/workspace/              # Agent 工作空间（人格、记忆、技能）
/root/.openclaw/workspace/infra/        # 宿主机脚本的 git 备份（start 时自动同步）
/root/.openclaw/agents/main/sessions/   # 会话数据
/root/.openclaw/memory/main.sqlite      # 记忆搜索数据库
/tmp/openclaw/                          # 运行日志
```

### Git 备份策略

宿主机脚本通过 start 脚本自动同步到容器 `workspace/infra/` 目录，随 soul-repo-sync cron（每天 23:00 UTC）一起提交到 GitHub。

宿主机是主副本（日常编辑），`infra/` 是 git 备份副本（只读镜像）。恢复流程：从 GitHub clone soul repo → 从 `infra/` 复制脚本到宿主机。

`openclaw.json.backup` **不进 git**（含真实 key），只存在宿主机上。

---

## 10. 安全提醒

以下信息敏感，不要泄露：

- `openclaw.json` 中的 API Key（hfai、deepseek）
- `gateway.auth.token`（Web UI 访问凭证）
- `identity/device.json`（设备私钥）
- `agents/main/agent/auth-profiles.json`（认证 profile）
- `channels.feishu.appSecret`（飞书应用密钥）
- ClawHub 登录 token
- `github_pat.txt`（GitHub PAT）

**注意：** Git 仓库中的 `openclaw.json.bak` 是去敏版（REDACTED），不含真实 key。真实 key 只在容器内和宿主机备份中。
