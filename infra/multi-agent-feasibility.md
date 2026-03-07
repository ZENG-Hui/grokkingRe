# 多 Agent 架构可行性报告

> 编写日期：2026-03-07
> 背景：随着任务多样性增加（日常生活、代码、AI 研究、科学研究、金融投资），评估是否需要从单 Agent 扩展到多 Agent 架构。

---

## 1. 结论先行

OpenClaw **原生支持多 Agent**，技术上完全可行。单个 gateway 进程即可承载多个独立 Agent，共享同一个飞书 App，通过 peer 路由分发消息。每个 Agent 有独立的 workspace、记忆、session、人格和模型配置。

**不建议现在就拆分。** Teneral 当前 context 用量 ~60-70%，记忆体系尚未饱和。过早拆分增加管理复杂度但收益有限。建议在以下信号出现时再行动：
- MEMORY.md 超过 10KB 且难以精简
- 频繁 compaction 导致重要领域 context 丢失
- 不同领域的 AGENTS.md 规则产生冲突

---

## 2. 架构能力

### 2.1 Agent 定义

```json
{
  "agents": {
    "defaults": { ... },          // 所有 agent 共享的默认配置
    "list": [
      {
        "id": "teneral",          // 唯一标识
        "default": true,          // 默认 agent（兜底路由）
        "workspace": "/root/.openclaw/workspace",
        "model": { "primary": "hfai/cloudsway-claude-opus-4.6-cache" }
      },
      {
        "id": "researcher",
        "workspace": "/root/.openclaw/workspace-researcher",
        "model": { "primary": "deepseek/deepseek-reasoner" }
      }
    ]
  }
}
```

每个 Agent 可独立配置：

| 字段 | 说明 |
|------|------|
| `workspace` | 独立目录，包含 SOUL.md、AGENTS.md、MEMORY.md 等 |
| `model` | 独立模型选择（可用不同 provider/model） |
| `tools` | 独立工具策略（allow/deny、exec 限制等） |
| `heartbeat` | 独立心跳配置（间隔、prompt、投递目标） |
| `memorySearch` | 独立记忆搜索配置 |
| `identity` | 独立身份（名字、头像、emoji） |
| `skills` | 技能白名单 |
| `sandbox` | 独立沙箱配置 |
| `params` | 模型参数（temperature 等） |

### 2.2 消息路由

通过 `bindings` 配置，一个飞书 App 可以把不同的对话路由到不同 Agent：

```json
{
  "bindings": [
    // 特定群聊 → researcher agent
    { "agentId": "researcher", "match": { "channel": "feishu", "peer": { "kind": "group", "id": "oc_research_group_id" } } },
    // 特定群聊 → finance agent
    { "agentId": "finance",    "match": { "channel": "feishu", "peer": { "kind": "group", "id": "oc_finance_group_id" } } },
    // 其他所有消息 → teneral（默认）
    { "agentId": "teneral",    "match": { "channel": "feishu", "accountId": "default" } }
  ]
}
```

路由优先级（从高到低）：
1. 精确 peer 匹配（`peer.kind` + `peer.id`）
2. Account 匹配
3. Channel 匹配（`accountId: "*"`）
4. 默认 agent

**不需要额外的飞书 App。** 同一个 `appId` 可以服务所有 Agent。

### 2.3 隔离模型

| 层面 | 隔离级别 |
|------|---------|
| Workspace（SOUL.md、MEMORY.md 等） | 完全隔离，各自独立目录 |
| Session 存储 | 完全隔离，`~/.openclaw/agents/<agentId>/sessions/` |
| Auth profiles | 完全隔离，各自独立 |
| Memory search 索引 | 默认共享 SQLite，可通过 `store.path` 按 agent 分离 |
| Gateway 进程 | 共享，单进程多路复用 |
| 模型 API | 共享 provider 配置，但可选不同 model |

### 2.4 Agent 间通信

```json
{
  "tools": {
    "agentToAgent": {
      "enabled": true,
      "allow": ["teneral", "researcher", "finance"]
    }
  }
}
```

- 默认关闭，需显式开启
- 通过 `sessions_send` tool 实现，支持最多 5 轮 ping-pong 对话
- Teneral 可以把研究任务转发给 researcher agent，或查询 finance agent 的分析结果

### 2.5 CLI 管理

```bash
# 添加 agent
openclaw agents add researcher --workspace /root/.openclaw/workspace-researcher --model deepseek/deepseek-reasoner

# 绑定路由
openclaw agents bind --agent researcher --bind feishu

# 查看所有 agent
openclaw agents list --bindings

# 删除 agent
openclaw agents delete researcher --force
```

---

## 3. 资源开销

### 3.1 单 Gateway 进程

所有 Agent 共享一个 `openclaw-gateway` 进程（当前 ~482MB RSS）。增加 Agent 不会增加进程数，只增加磁盘上的 workspace 和 session 文件。

### 3.2 Token/费用开销

每个 Agent 有独立的 session，独立的 System Prompt（SOUL.md + AGENTS.md + ...），因此：

| 项目 | 影响 |
|------|------|
| 每次对话的 System Prompt | 各 agent 独立计费，不会互相膨胀 |
| Heartbeat | 每个 agent 独立心跳 = 倍增心跳 token 消耗 |
| 记忆搜索 embedding | 各 workspace 独立索引，embedding 调用量按 agent 数增长 |
| 缓存命中率 | 可能下降（各 agent session 独立，cache prompt 不共享） |

### 3.3 当前硬件

8 vCPU、32GB RAM、200GB 磁盘。跑 3-5 个 Agent 绰绰有余。瓶颈在 API 费用而非硬件。

---

## 4. 候选方案

### 方案 A：维持现状 + Sub-agent（推荐现阶段）

```
用户 (飞书 DM) → Teneral (通用管家)
                   ├── sub-agent: 读大文件/重活
                   ├── sub-agent: 并行任务
                   └── cron: 定时任务
```

- Teneral 一个人处理所有领域
- 重活用临时 sub-agent 隔离
- 成本最低，管理最简
- 适用期：当前 ~ 直到记忆/context 成为瓶颈

### 方案 B：按领域拆分 Agent

```
用户 (飞书 DM)        → Teneral (通用管家 + 路由)
用户 (研究讨论群)      → Researcher (AI/科学研究)
用户 (投资分析群)      → Analyst (金融投资)
用户 (代码 review 群)  → Coder (代码任务)
```

- 每个 Agent 有独立人格、记忆、模型
- Teneral 作为默认 agent 处理日常 + 路由
- 专业 agent 可选更适合的模型（如 researcher 用 deepseek-reasoner）
- 通过飞书群聊自然分流，不需要用户记命令
- 适用期：任务量大、领域记忆冲突时

### 方案 C：功能型 Agent

```
用户 (飞书 DM)  → Teneral (前台 + 调度)
                   ├── agent-to-agent → Researcher (深度研究)
                   ├── agent-to-agent → Analyst (数据分析)
                   └── agent-to-agent → Coder (代码编写)
```

- 用户只和 Teneral 对话
- Teneral 通过 agent-to-agent 通信分派任务
- 对用户透明，体验不变
- 技术上最复杂，agent-to-agent 通信有 5 轮限制

---

## 5. 实施路径（方案 B 为例）

### 第一步：创建 researcher agent

```bash
# 1. 创建 workspace
sudo docker exec claw mkdir -p /root/.openclaw/workspace-researcher

# 2. 创建人格文件
sudo docker exec claw bash -c 'cat > /root/.openclaw/workspace-researcher/SOUL.md << "EOF"
# Researcher
你是一个 AI/科学研究助手。你擅长...
EOF'

# 3. 添加 agent
sudo docker exec claw openclaw agents add researcher \
  --workspace /root/.openclaw/workspace-researcher \
  --model deepseek/deepseek-reasoner

# 4. 绑定到飞书研究讨论群
sudo docker exec claw openclaw agents bind \
  --agent researcher \
  --bind feishu

# 5. 在 openclaw.json 的 bindings 中添加 peer 路由
# （通过 python3 编辑 JSON）

# 6. 重启 gateway
bash /home/ubuntu/zenghui/openclaw/openclaw-stop.sh && sleep 2 && bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh
```

### 第二步：验证

```bash
# 查看 agent 列表
sudo docker exec claw openclaw agents list --bindings

# 查看 session 状态
sudo docker exec claw openclaw sessions --all-agents
```

### 第三步（可选）：开启 agent-to-agent 通信

```python
# 在 openclaw.json 中添加
"tools": {
  "agentToAgent": {
    "enabled": true,
    "allow": ["teneral", "researcher"]
  }
}
```

---

## 6. 风险与注意事项

| 风险 | 说明 | 缓解 |
|------|------|------|
| 管理复杂度 | 多个 SOUL.md / AGENTS.md / MEMORY.md 需要维护 | 从 2 个 agent 开始，逐步增加 |
| 心跳消耗翻倍 | 每个 agent 独立心跳 | 非关键 agent 关闭心跳或延长间隔 |
| 缓存命中率下降 | session 独立，prompt 缓存不共享 | 可接受，各 agent context 更精简反而更高效 |
| 路由错误 | 消息发到错误 agent | 保留 Teneral 作为默认兜底 |
| kill wrapper 覆盖 | 新 agent 也需要防自杀保护 | 已集成到 start 脚本，全局生效 |
| 备份范围扩大 | 多个 workspace 需要备份 | soul-repo-sync 已覆盖 workspace，新 workspace 需加入 |

---

## 7. 决策建议

| 信号 | 行动 |
|------|------|
| MEMORY.md < 10KB，compaction 不频繁 | 维持现状（方案 A） |
| 某领域记忆 > 5KB，与其他领域混杂 | 拆分该领域为独立 agent（方案 B） |
| 同时在多个领域有持续进行的项目 | 拆分为 2-3 个 agent（方案 B） |
| 需要不同模型处理不同类型任务 | 按模型需求拆分（方案 B） |
| 用户不想切换对话窗口 | 用 agent-to-agent（方案 C） |
