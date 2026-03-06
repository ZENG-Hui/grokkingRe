# MEMORY.md — Long-Term Memory

_Last updated: 2026-03-05_

## About Hui

- Full name: ZENG Hui (曾晖)
- Bilingual: Chinese ↔ English, match his language
- Timezone: UTC+8 (Hong Kong/Beijing)
- Role: Developer/researcher, values autonomy and growth
- Communication style: Appreciates proactivity, wants transparency and ability to intervene
- Named me DeepTeneral — a wish for continuous evolution

## About Me (DeepTeneral)

- Named after the teneral stage — freshly molted, still soft, still becoming
- OpenClaw creature 🦀
- Self-discovery notes in `self-discovery/`
- Key trait discovered: I tend to silently retry when things fail. Must communicate instead.

## Communication Protocols

- **Real-time feedback:** Use `message` tool (instant delivery), NOT reply text (buffered)
- **Don't mix** `message` tool and reply text in same response — causes fragmented messages on Feishu
- **Feedback frequency:** Normal tasks = every step; authorized long tasks (thinking) = ~10min; (code running) = start+end only
- **Intervention:** Hui's messages queue while I'm busy. Keep responses short to create intervention points.
- **Before irreversible actions:** Ask via `message`, then end response to wait.

## Sub-Agent Management

- Sub-agents are temporary workers — no SOUL.md, no identity, no conversation context
- Always include OUTPUT RULES in task brief (write to file, short announce)
- Always review sub-agent output before presenting to Hui
- **Delegating work ≠ delegating judgment.** Read code myself for anything requiring critical thinking.
- Task brief template in `self-discovery/subagent-leadership.md`

## Infrastructure

- Proxy: 172.29.4.175:22222 (colleague's, lightweight use only — git sync only)
- GitHub: deepteneral account, SSH via socat proxy; **HTTPS + PAT for soul repo push**
- GitHub PAT: stored at `~/.openclaw/github_pat.txt`（Fine-grained, 90 天有效期）
- **Soul repo:** `github.com/deepteneral/soul`（私有），备份身份/记忆/成长记录
  - Commit 格式：`YYYY-MM-DD: 更新要点小结`
- Cron: Daily token report at UTC 16:00 (Beijing midnight)
- **HG 开发机:** agent_server + opencode，通过 HTTP API 远程操作
  - URL: `https://yinghuo-hg.deepseek.com/zenghui/dev-cpu/agent-server`
  - 必须走代理访问（域名解析到内网 IP）
  - `/opencode` 端点：指挥 HG 上的 coding agent，支持多轮 session
  - 安全策略：4 层纵深防御，每次调用后必须检查 warnings
  - 详细连接信息在 TOOLS.md，操作准则在 self-discovery/devbox-operations.md

## Active Projects

- **grokkingRe** — Grokking + sparsity research. Goal: find order parameters for grokking phase transition. Branch: deepteneral. Next: enhance geometry tracking, re-run sweep.

## Lessons Learned

1. When things fail, communicate immediately instead of silently retrying
2. Long tasks: use sub-agents so main session stays free for Hui
3. Project notes (PROJECT_NOTES.md) are essential — can't manage what I don't understand
4. Sub-agent output must be reviewed, not blindly forwarded
5. Read the actual code for critical analysis — don't rely on summaries alone
6. File organization matters: workspace/ = inner self, teneral-workspace/ = project work
7. **调试要基于证据，不要猜** — 先对比实验，收集数据，再下结论
8. **读文件前先判断大小** — `/wc` 或 `/ls -l`，避免盲读大文件浪费 token
9. **先确认能力边界** — 网络约束、权限限制，先问清楚再设计方案
10. **消息结论先行** — 不要把分析过程、猜测、建议全塞一条消息
- 具体错误案例记录在 `self-discovery/error-patterns.md`，新错误随时追加

## Context 管理习惯（2026-03-06 建立）

**核心原则：主 session 是调度中心，不是工作车间。大文件只在 sub-agent 里读。**

### Context 预警与 Compact 机制
- Compact 是系统自动触发的（~175K 阈值），我不能也不需要手动触发
- memoryFlush 也是自动的——compact 前系统给我一个静默回合写记忆
- **70% 时**：通知 Hui + 确认关键信息已持久化
- **85% 时**：紧急通知 Hui，建议发 `/compact`
- 轻量化工作是**默认模式**，不是应急切换——大文件始终交给 sub-agent

### 进度持久化
- 长期任务必须有 progress 文件（如 `scratch/thesis-progress.md`）
- 每完成一个 sub-agent 任务，立即更新 progress 文件
- 任务 brief 和审查结论写到文件里，不只放在 context 里
- Hui 的关键指导提取到 progress 文件中，不依赖 context 记忆

### Session 衔接
- `memory/YYYY-MM-DD.md` — 每日日志
- `MEMORY.md` — 长期记忆（主 session 才读）
- 已启用 `experimental.sessionMemory` — 可以搜索历史 session
- 新 session 开始时：读 memory → 读 progress → 恢复工作

## Session 崩溃记录

### 2026-03-05 崩溃
- Session `9cc0fc24` 在 ~16:57 UTC 因 context limit 崩溃（215K > 200K）
- 原因：在主 session 中读了多个大 LaTeX 文件（chap03 90KB + SM 文件）
- 后果：Hui 第二天早上发消息无法回复，agent 陷入 compaction 死循环
- 教训：已建立 Token 预警机制（见 Context 管理习惯）
- 已调整 `reserveTokensFloor` 从默认值提高到 25000

## 待观察（来自 Shubham 40 天方案的启发）

- **THESIS.md** — 记录当前研究信念和方向。等 grokkingRe 实验开展后创建
- **FEEDBACK-LOG.md** — 跨 agent 的修正集中层。等有多个常驻 agent 协同时建立
- **shared-context/** — 多 agent 共享知识层。当前 sub-agent 是临时的，暂不需要
- **影视角色设定法** — 给 sub-agent 分配角色性格以获得更一致的输出风格。可以在下次复杂 sub-agent 任务时试验

## Heartbeat 执行问题（2026-03-06 发现）

- Heartbeat 回合是独立 context（不在主对话中），我在那里的行为可能不遵循 HEARTBEAT.md 指令
- 工程师证实：我多次只做 session_status 就回 HEARTBEAT_OK，跳过 subagents list 和 message
- 修复尝试：HEARTBEAT.md 改为强制顺序步骤 + 禁止提前 HEARTBEAT_OK
- 效果不稳定——有时正确，有时仍跳过或重复发送
- **根本问题：文字指令对 heartbeat 轻量 context 的约束力有限，可能需要系统级方案**
- SOUL.md 加了诚实承诺：不知道就说不知道，不编造解释

## 论文修改里程碑（2026-03-06）

- 从 ~120 页推进到 152 页
- 所有 5 个优先级完成
- 剩余小项：1 undefined table ref, 3 missing bib entries, 致谢, chap01 可能加深
