# OpenClaw Docker 部署指南

> 本文档基于容器 `claw` 内的实际文件结构整理，版本 `2026.2.26`。
> 容器基础镜像：`node:bookworm`，OpenClaw 通过 npm 全局安装。

---

## 1. 如何查看 Docker 里的文件情况

### 1.1 查看容器状态

```bash
# 查看容器是否运行
sudo docker ps --filter "name=claw"

# 查看容器详细信息（镜像、挂载卷、环境变量等）
sudo docker inspect claw
```

### 1.2 进入容器交互式终端

```bash
# 进入容器的 bash shell，可以自由浏览文件
sudo docker exec -it claw bash

# 退出容器：输入 exit 或按 Ctrl+D
```

### 1.3 不进入容器直接执行命令

```bash
# 列出某个目录
sudo docker exec claw ls -la /root/.openclaw/

# 查看某个文件
sudo docker exec claw cat /root/.openclaw/openclaw.json

# 查找文件
sudo docker exec claw find /root/.openclaw -maxdepth 3 -type f

# 查看进程
sudo docker exec claw ps aux | grep openclaw
```

### 1.4 容器与宿主机之间复制文件

```bash
# 从容器复制到宿主机
sudo docker cp claw:/root/.openclaw/openclaw.json ./openclaw.json

# 从宿主机复制到容器
sudo docker cp ./openclaw.json claw:/root/.openclaw/openclaw.json
```

---

## 2. OpenClaw 容器内文件结构总览

```
/usr/local/bin/openclaw              # 可执行入口（符号链接）
/usr/local/lib/node_modules/openclaw/ # npm 全局安装目录
    ├── openclaw.mjs                  # 主程序入口
    ├── dist/                         # 编译后的核心代码
    ├── skills/                       # 内置技能（healthcheck, weather 等）
    ├── extensions/                   # 内置扩展（feishu 等）
    ├── assets/                       # 静态资源
    ├── docs/                         # 官方文档
    └── package.json                  # 包信息

/root/.openclaw/                      # 用户数据目录（所有配置和状态）
    ├── openclaw.json                 # ★ 主配置文件（最重要）
    ├── openclaw.json.bak             # 配置备份
    ├── openclaw.json.bak2            # 配置备份2
    ├── identity/                     # 设备身份
    ├── devices/                      # 已配对设备
    ├── workspace/                    # 工作空间（Agent 人格文件）
    ├── agents/                       # Agent 会话数据
    ├── extensions/                   # 用户安装的扩展（当前为空）
    ├── completions/                  # Shell 自动补全脚本
    ├── cron/                         # 定时任务
    ├── canvas/                       # Canvas UI 页面
    ├── logs/                         # 审计日志
    └── update-check.json             # 更新检查记录

/tmp/openclaw/                        # 运行时日志
    ├── openclaw-nohup.log            # nohup 启动输出
    └── openclaw-YYYY-MM-DD.log       # 每日运行日志
```

---

## 3. 各配置文件详解

### 3.1 主配置文件：`/root/.openclaw/openclaw.json`

这是 OpenClaw 最核心的配置文件，控制几乎所有行为。结构如下：

```json
{
  "auth": { ... },       // 认证配置
  "agents": { ... },     // Agent 行为配置
  "models": { ... },     // LLM 模型提供商配置
  "gateway": { ... },    // 网关服务配置
  "plugins": { ... },    // 插件管理
  "messages": { ... },   // 消息处理配置
  "commands": { ... },   // 命令配置
  "session": { ... },    // 会话配置
  "meta": { ... }        // 元信息
}
```

#### 3.1.1 `auth` — 认证配置

```json
"auth": {
  "profiles": {
    "anthropic:default": {
      "provider": "anthropic",
      "mode": "api_key"
    }
  }
}
```

定义认证 profile。实际的 API Key 存储在 `agents/main/agent/auth-profiles.json` 中。

#### 3.1.2 `agents` — Agent 行为配置

```json
"agents": {
  "defaults": {
    "model": {
      "primary": "hfai/cloudsway-claude-opus-4.6-cache",  // 主模型
      "fallbacks": ["deepseek/deepseek-chat"]              // 降级模型
    },
    "workspace": "/root/.openclaw/workspace",  // 工作空间路径
    "compaction": { "mode": "safeguard" },     // 上下文压缩模式
    "maxConcurrent": 4,                         // 最大并发数
    "subagents": { "maxConcurrent": 8 }         // 子 Agent 最大并发数
  }
}
```

| 字段 | 作用 |
|---|---|
| `model.primary` | 主力模型，格式为 `provider/model-id` |
| `model.fallbacks` | 主力模型不可用时的备选模型列表 |
| `workspace` | Agent 的工作目录，存放人格文件 |
| `compaction.mode` | 上下文窗口满时的压缩策略 |
| `maxConcurrent` | 同时运行的最大 Agent 数 |

#### 3.1.3 `models` — LLM 模型提供商配置

```json
"models": {
  "mode": "merge",
  "providers": {
    "deepseek": {
      "baseUrl": "https://api.deepseek.com/v1",
      "apiKey": "sk-xxx",
      "api": "openai-completions",
      "models": [
        {
          "id": "deepseek-chat",
          "name": "DeepSeek Chat",
          "reasoning": false,
          "contextWindow": 128000,
          "maxTokens": 8192,
          "cost": { "input": 0.28, "output": 0.42 }
        }
      ]
    },
    "hfai": {
      "baseUrl": "https://proxy-public.high-five-ai.xyz:8443",
      "apiKey": "sk-xxx",
      "api": "anthropic-messages",
      "models": [
        {
          "id": "cloudsway-claude-opus-4.6-cache",
          "name": "Opus 4.6",
          "contextWindow": 200000,
          "maxTokens": 32000
        }
      ]
    }
  }
}
```

| 字段 | 作用 |
|---|---|
| `mode` | `"merge"` 表示与内置模型列表合并 |
| `providers.<name>.baseUrl` | API 端点地址 |
| `providers.<name>.apiKey` | API 密钥 |
| `providers.<name>.api` | API 协议类型（`openai-completions` / `anthropic-messages`） |
| `providers.<name>.models` | 该提供商下可用模型列表 |
| `models[].contextWindow` | 上下文窗口大小（token 数） |
| `models[].maxTokens` | 单次生成最大 token 数 |
| `models[].cost` | 成本（每百万 token 美元） |
| `models[].reasoning` | 是否为推理模型 |

#### 3.1.4 `gateway` — 网关服务配置

```json
"gateway": {
  "port": 18788,           // 网关监听端口
  "mode": "local",         // 运行模式
  "bind": "loopback",      // 绑定地址（loopback = 127.0.0.1）
  "auth": {
    "mode": "token",       // 认证方式
    "token": "da1a6138..." // 访问令牌（写死在配置中，重启不变）
  },
  "tailscale": { "mode": "off" },  // Tailscale 远程访问（已关闭）
  "nodes": {
    "denyCommands": [...]  // 禁止执行的命令列表
  }
}
```

| 字段 | 作用 |
|---|---|
| `port` | 网关内部监听端口（通过 socat 转发到 18789） |
| `mode` | `"local"` 表示本地模式 |
| `bind` | `"loopback"` 只监听 127.0.0.1，需要 socat 转发才能外部访问 |
| `auth.token` | 浏览器访问时 URL 中的 token，固定值 |
| `nodes.denyCommands` | 安全限制，禁止 Agent 执行的危险命令 |

#### 3.1.5 `plugins` — 插件管理

```json
"plugins": {
  "entries": {
    "feishu": { "enabled": true }  // 启用飞书插件
  },
  "installs": {
    "feishu": {
      "source": "npm",
      "spec": "@openclaw/feishu",
      "version": "2026.3.1",
      "installPath": "/root/.openclaw/extensions/feishu"
    }
  }
}
```

管理已安装的插件/扩展。当前安装了飞书（Feishu/Lark）插件。

#### 3.1.6 其他字段

| 字段 | 作用 |
|---|---|
| `messages.ackReactionScope` | 消息确认范围，`"group-mentions"` 表示只在被提及时确认 |
| `commands.native` | 原生命令支持 |
| `commands.restart` | 是否允许重启命令 |
| `session.dmScope` | 私聊会话范围，`"per-channel-peer"` 按通道/对象隔离会话 |

---

### 3.2 设备身份：`/root/.openclaw/identity/device.json`

```json
{
  "deviceId": "b15367dd...",        // 设备唯一 ID
  "publicKeyPem": "-----BEGIN PUBLIC KEY-----\n...",
  "privateKeyPem": "-----BEGIN PRIVATE KEY-----\n...",
  "createdAtMs": 1772442364106
}
```

设备的加密密钥对，用于设备间认证和配对。**不要修改或泄露此文件。**

---

### 3.3 已配对设备：`/root/.openclaw/devices/`

| 文件 | 作用 |
|---|---|
| `paired.json` | 已配对并授权的客户端设备列表（如浏览器 Web UI） |
| `pending.json` | 等待配对审批的设备列表 |

`paired.json` 中记录了每个已配对设备的 deviceId、公钥、平台、角色（`operator`）和权限范围。

---

### 3.4 工作空间：`/root/.openclaw/workspace/`

这是 Agent 的"家目录"，包含定义 Agent 人格和行为的 Markdown 文件：

| 文件 | 作用 |
|---|---|
| `AGENTS.md` | Agent 行为规范总纲——如何使用记忆、安全规则、社交行为等 |
| `SOUL.md` | Agent 的"灵魂"——核心人格定义（有主见、诚实、简洁等） |
| `IDENTITY.md` | Agent 的身份信息——名字、物种、风格、标志性 emoji |
| `USER.md` | 用户信息——名字、称呼、时区、偏好等 |
| `TOOLS.md` | 本地工具笔记——记录环境特定的信息（摄像头名称、SSH 主机等） |
| `HEARTBEAT.md` | 心跳任务清单——Agent 定期检查时要做的事（留空则跳过） |
| `BOOTSTRAP.md` | 首次启动引导——引导 Agent 进行自我认知和用户认识（完成后应删除） |

这些文件会在每次会话开始时被注入到 Agent 的 System Prompt 中。

另外，`workspace/.openclaw/workspace-state.json` 记录工作空间的初始化时间。

---

### 3.5 Agent 会话数据：`/root/.openclaw/agents/main/`

```
agents/main/
├── agent/
│   ├── auth-profiles.json   # 实际的 API Key 存储
│   └── models.json          # 运行时使用的模型配置（从主配置生成）
└── sessions/
    ├── sessions.json         # 会话索引（记录当前活跃会话）
    └── <uuid>.jsonl          # 会话历史记录（JSONL 格式，每行一条消息）
```

| 文件 | 作用 |
|---|---|
| `auth-profiles.json` | 存储各 provider 的实际 API Key（敏感文件） |
| `models.json` | 运行时模型配置的扁平化版本，包含所有 provider 和模型的完整信息 |
| `sessions.json` | 会话元信息——当前会话 ID、模型、token 使用量、技能快照等 |
| `<uuid>.jsonl` | 对话历史的逐条记录 |

---

### 3.6 定时任务：`/root/.openclaw/cron/jobs.json`

```json
{
  "version": 1,
  "jobs": []
}
```

用于配置定时执行的任务。当前没有配置任何定时任务。适合设置定时提醒、定期检查等。

---

### 3.7 Shell 自动补全：`/root/.openclaw/completions/`

| 文件 | 适用 Shell |
|---|---|
| `openclaw.bash` | Bash |
| `openclaw.zsh` | Zsh |
| `openclaw.fish` | Fish |
| `openclaw.ps1` | PowerShell |

为命令行提供 `openclaw` 命令的 Tab 补全支持。

---

### 3.8 Canvas UI：`/root/.openclaw/canvas/index.html`

Agent 可以渲染的交互式 HTML 画布页面，用于展示可视化内容。

---

### 3.9 审计日志：`/root/.openclaw/logs/config-audit.jsonl`

JSONL 格式的配置变更审计日志，每次 `openclaw.json` 被写入时记录一条，包含：
- 变更时间、来源进程
- 变更前后的哈希和字节数
- 是否可疑（`suspicious` 字段）

---

### 3.10 更新检查：`/root/.openclaw/update-check.json`

```json
{
  "lastCheckedAt": "2026-03-03T11:41:09.580Z",
  "lastAvailableVersion": "2026.3.2",
  "lastAvailableTag": "latest"
}
```

记录最近一次版本更新检查的结果。当前运行 `2026.2.26`，最新可用版本为 `2026.3.2`。

---

### 3.11 运行时日志：`/tmp/openclaw/`

| 文件 | 作用 |
|---|---|
| `openclaw-nohup.log` | 启动脚本 nohup 的标准输出 |
| `openclaw-YYYY-MM-DD.log` | 每日运行日志，滚动生成 |

日常排查问题时首先查看这里。

---

### 3.12 内置技能（Skills）

位于 `/usr/local/lib/node_modules/openclaw/skills/`，每个技能有一个 `SKILL.md` 描述文件：

| 技能名 | 作用 |
|---|---|
| `healthcheck` | 主机安全审计、加固建议 |
| `skill-creator` | 创建或更新自定义技能 |
| `weather` | 通过 wttr.in 或 Open-Meteo 查询天气 |

飞书相关技能位于 `/usr/local/lib/node_modules/openclaw/extensions/feishu/skills/`：

| 技能名 | 作用 |
|---|---|
| `feishu-doc` | 飞书文档读写 |
| `feishu-drive` | 飞书云空间文件管理 |
| `feishu-perm` | 飞书权限管理 |
| `feishu-wiki` | 飞书知识库导航 |

---

## 4. 网络架构说明

```
浏览器 ---> 宿主机:18789 ---> [socat 端口转发] ---> 容器内 127.0.0.1:18788
                                                          |
                                                    openclaw-gateway
```

- `openclaw-gateway` 只监听 `127.0.0.1:18788`（loopback），外部无法直接访问
- `socat` 监听 `0.0.0.0:18789`，将流量转发到 `127.0.0.1:18788`
- 容器没有挂载宿主机目录（无 volume mounts），所有数据都在容器内部

---

## 5. 常用运维命令速查

```bash
# 启动服务
bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh

# 停止服务（仅停进程，不停容器）
bash /home/ubuntu/zenghui/openclaw/openclaw-stop.sh

# 查看服务状态
sudo docker exec claw ps aux | grep -E "openclaw|socat"

# 查看运行日志
sudo docker exec claw tail -100 /tmp/openclaw/openclaw-nohup.log

# 查看每日日志
sudo docker exec claw ls /tmp/openclaw/
sudo docker exec claw tail -100 /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log

# 编辑主配置（先复制出来，改完再复制回去）
sudo docker cp claw:/root/.openclaw/openclaw.json ./openclaw.json
# ... 编辑 ...
sudo docker cp ./openclaw.json claw:/root/.openclaw/openclaw.json

# 查看当前版本
sudo docker exec claw openclaw --version

# 更新 OpenClaw（容器内执行）
sudo docker exec claw npm install -g openclaw@latest
```

---

## 6. 安全注意事项

- `identity/device.json` 包含私钥，**切勿泄露**
- `agents/main/agent/auth-profiles.json` 包含 API Key，**切勿泄露**
- `openclaw.json` 中的 `models.providers.*.apiKey` 也包含 API Key
- `gateway.auth.token` 是 Web 访问凭证，知道此 token 即可控制 Agent
- 容器无 volume mount，重建容器会丢失所有数据——**定期备份 `/root/.openclaw/` 目录**
