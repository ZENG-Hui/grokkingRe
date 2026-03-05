# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Network / Proxy

- **代理地址：** `172.29.4.175:22222`（HTTP 代理，同事的梯子）
- **使用原则：轻量使用！** 只用于 GitHub 同步（git push/pull），不要跑大流量
- **配置位置：** `~/.ssh/config` 中通过 socat 让 SSH 走代理
- **其他流量：** 直连，不走代理

### GitHub

- **账号：** deepteneral（`zengh17+deepteneral@gmail.com`）
- **SSH 密钥：** `/root/.ssh/id_ed25519`
- **协作仓库：** `ZENG-Hui/grokkingRe`（collaborator 权限）
- **默认分支名：** `deepteneral`

## Hui's Dev Machine (Agent Server)

- **URL:** `https://yinghuo-hg.deepseek.com/zenghui/dev-cpu/agent-server`
- **Auth:** Bearer token
- **Token:** `T1moNCxHPLy4VpVlMFJlDpjPVtUT-tezJCSp0RzUmGo`
- **Proxy required:** `http://172.29.4.175:22222` (域名解析到内网 IP)
- **RW_DIRS:** `/hf3fs-hg/prod/deepseek/shared/zenghui/deepteneral-workspace`
- **RO_DIRS:** 空（任意路径可读）
- **Platform:** 萤火 HG 机房，开发容器 dev-cpu
- **Endpoints:** /ping, /ls, /cat, /find, /wc, /du, /write, /mkdir, /run, /info, /opencode
- **`/opencode`**: 委派编码任务给 HG 上的 opencode agent（支持多轮 session）
- **安全策略:** 4 层纵深防御（见 READ_on_hg/SECURITY_POLICY.md）。每次 /opencode 调用后必须检查 warnings 数组
- **并发限制:** /opencode 最多 2 个同时调用
- **Usage pattern:**
```bash
curl -s --proxy http://172.29.4.175:22222 \
  -X POST "$BASE_URL/ls" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"path": "/hf3fs-hg/prod/deepseek/shared/zenghui/deepteneral-workspace"}'
```

## Feishu

### 应用信息
- **App ID:** `cli_a922983e81785cca`
- **Bot 名称:** DeepTeneral
- **Hui 的 Open ID:** `ou_69ba193a57d1c578a8d42f06efd985c8`

### 权限现状
- ✅ 文档读写 (docx:document)
- ✅ 知识库读写 (wiki:wiki) — 但创建新页面暂不可用，编辑已有页面可以
- ✅ 多维表格 (bitable:app)
- ✅ 权限管理 (docs:permission.member:create/transfer)
- ❌ 云盘浏览 (drive:drive) — 未开通
- **Bot 没有根文件夹** — 只能访问被分享的文件

### 知识库
- **Space ID:** `7613672766692183254`
- **首页 node_token:** `JpnpwLGGfi1iY5kM915cTxnJn0c`
- **首页 doc_token:** `Zhn8d9PNPoOgOrxPC33caFGnn59`

### Feishu API 技巧（来自 OpenClaw-Claude 的踩坑指南）

**Reaction（轻量确认）：**
```bash
# 获取 token
CONFIG="$HOME/.openclaw/openclaw.json"
APP_ID=$(node -e "console.log(require('$CONFIG').channels.feishu.appId)")
APP_SECRET=$(node -e "console.log(require('$CONFIG').channels.feishu.appSecret)")
TOKEN=$(curl -s -X POST 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal' \
  -H 'Content-Type: application/json' \
  -d "{\"app_id\":\"$APP_ID\",\"app_secret\":\"$APP_SECRET\"}" \
  | node -e "process.stdin.on('data',d=>console.log(JSON.parse(d).tenant_access_token))")

# 加 reaction
curl -s -X POST "https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reactions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reaction_type":{"emoji_type":"THUMBSUP"}}'
```

常用 emoji_type: THUMBSUP, OK, DONE, THANKS, MUSCLE, HEART, THINKING, SMILE, Fire, CheckMark, Hundred

**注意事项：**
- post 富文本中方括号会被解析为表情 — 代码内容放代码块里
- 群聊中必须 @ 对方才能触发通知
- Interactive Card 必须 API 直调，message tool 会把 JSON 当纯文本发
