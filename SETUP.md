# SETUP.md — 迁移清单

*当 DeepTeneral 需要迁移到新硬件时，按此清单操作。*

## 1. 基础环境

- 安装 OpenClaw（参考 https://docs.openclaw.ai）
- Clone 此仓库到 workspace：`~/.openclaw/workspace/`

## 2. 配置文件（需手动创建，含敏感信息）

### ~/.openclaw/openclaw.json

核心配置结构：
- `auth.profiles` — Anthropic API 认证
- `models.providers.hfai` — Claude Opus 4.6（主模型）
- `models.providers.deepseek` — DeepSeek Chat（fallback）
- `agents.defaults.model.primary` — `hfai/cloudsway-claude-opus-4.6-cache`
- `channels.feishu` — 飞书 app 配置（appId: cli_a922983e81785cca）
- `plugins.entries.feishu` — 飞书插件

API keys 需要找 Hui 获取。

### ~/.ssh/config

```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    ProxyCommand socat - PROXY:172.29.4.175:%h:%p,proxyport=22222
```

注意：代理地址可能变化，需确认。

### ~/.gitconfig

```
[user]
    name = DeepTeneral
    email = zengh17+deepteneral@gmail.com
```

## 3. SSH 密钥

- 生成新密钥或从安全备份恢复
- GitHub 账号 `deepteneral` 需要添加新公钥
- 测试：`ssh -T git@github.com`

## 4. 安装 Skills

```bash
# 参考 skills.txt
openclaw skill install agent-memory-system-new
openclaw skill install autonomous-tasks
openclaw skill install feishu-card
openclaw skill install feishu-chat
openclaw skill install find-skills
openclaw skill install subagent-driven-development
openclaw skill install task-decomposer
openclaw skill install using-superpowers
```

## 5. 安装插件

```bash
openclaw plugin install @openclaw/feishu
```

## 6. Cron Jobs

Daily Token Report（UTC 16:00 / 北京时间午夜）：
```bash
openclaw cron add --name "Daily Token Report" --schedule "0 16 * * *" --message "Generate a daily token usage report..."
```

## 7. 验证

- [ ] `openclaw status` 正常
- [ ] 飞书消息收发正常
- [ ] GitHub push/pull 正常
- [ ] Cron job 运行正常
- [ ] HG 开发机连接正常（通过代理访问 agent_server）

## 8. 迁移后

- 更新 TOOLS.md 中的环境特定信息
- 在 memory/ 记录迁移事件
- 测试所有关键功能
