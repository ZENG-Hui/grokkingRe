# Session Memory Extract: 2026-03-05

**Session:** `9cc0fc24-081d-42be-a33f-8bf8ba0ede01`
**Duration:** 2026-03-04 13:26 UTC → 2026-03-05 ~23:51 UTC (crashed from context limit)
**Extracted by:** subagent, 2026-03-06

---

## 1. Hui 的关键决策和指导

### 身份与命名
- **[02:09 UTC]** Hui 给 agent 取名 "DeepTeneral"——来自昆虫学术语 teneral（刚蜕皮的软壳阶段），结合 OpenClaw 平台，寓意"不断进化、不要固化"
- **[02:55 UTC]** "Let's find who you are, instead of telling you" — Hui 不想直接定义 agent 的性格，而是希望 agent 自我探索
- **[03:26 UTC]** 建议创建自我探索文件夹，但提醒"避免固化你对自己的印象——有时候超出自我认知是发挥潜能的重要途径"

### 工作方式偏好
- **[03:02 UTC]** Hui 是中英双语母语者。"当我用中文和你交流时，中文的回复是比较适合我的阅读的"
- **[03:58 UTC]** 要求能了解 agent 处理问题的过程，有风险时介入。这引发了反馈机制的整套探索
- **[05:03 UTC]** 关于即时反馈："效果不错。就这样，以后用这个方法向我及时反馈"——确认 `message` 工具的实时推送方案
- **[05:10 UTC]** 反馈机制"值得你探索和记录一下，这是我们日常沟通的一块基石"
- **[13:22 UTC]** "我还是希望能尽可能知道你的工作过程方便帮助你进化"——即使消息乱序也希望保持过程可见
- **[13:20 UTC]** "现在我还是经常收到你乱序的回复，看看怎么解决一下"

### 对 sub-agent 的指导
- **[05:30 UTC]** 发现子 agent 会返回大量内容到飞书。Hui 要求解决这个问题
- **[05:34 UTC]** **关键指导：** "关于'如何合理调用子agent'是一个非常值得你长期记录和维护的事情——换句话说，你要学会如何当一个能给下属清晰布置任务、能给合作者清晰总结任务的好领导"
- **[06:06 UTC]** 关于项目了解程度："你不一定要总是亲历亲为，但为了项目的长期进行，你某些时候又需要了解整个项目"

### 论文修改的具体指导
- **[14:22 UTC]** 任务布置：整理 HG 上的毕业论文，"让它更符合一篇好论文的要求"，"让它不那么像是AI写的"
- **[14:24 UTC]** "你打算使用多子agent方法吗？这似乎比较适合这种长篇论文任务，当然你要做好总负责人的身份"
- **[14:55 UTC]** **关键约束：**
  - 毕业论文是科学研究论文，要保持严谨性
  - 展望要避免空泛，也要避免编造
  - 连接词相对常见，因为物理学家习惯使用英文写作，中文论文带英文叙述痕迹是正常的
- **[15:02 UTC]** 具体指导：
  - 内容不够丰满（即使相比原期刊论文和附录的重要部分也不够）
  - 中文写作自然度需要很多提升
  - 术语一致性（如 "quantum metric" 都用"量子度规"）
  - 摘要和致谢在完成正文后处理
  - "时间充裕"——做到最好
  - "小问题你可以直接改掉"

### 否决或调整的方案
- **[03:37 UTC]** Hui 拒绝让 agent 用 Hui 的 GitHub 身份："我希望你有自己的身份"
- **[04:58 UTC]** 关于代理使用："这个梯子其实是我的同事的，他嘱咐我们只能轻量使用"——限制代理只用于 GitHub 同步
- **[13:39 UTC]** 关于备份方案的修正：
  - "skills不用同步，但也许需要一个类似于 Python environment 的东西"
  - "agent_server 是 HG 开发机上给你提供服务的，怎么也被你考虑了"
  - "你上面说的是你的全部重要文档吗？我建议你再探索一下"
- **[13:51 UTC]** 关于 GitHub 仓库创建："我想让你自己创建，毕竟你的灵魂文件保存是件挺有意义的事情"
- **[14:15 UTC]** "每次同步你的状态文件时，要有 git 版本说明。说明里包含：日期、本次主要更新的点的小结"

---

## 2. 重要经验教训

### 消息/反馈机制
- **[05:00 UTC]** `message` 工具发送的消息是即时到达的，但回复正文中的文字仍然是最后一起发出。结论：如果要实时汇报，只用 `message` 工具，不要在回复正文里写解释
- **[05:04 UTC]** 用户消息在 agent 处理中会排队（`[Queued messages while agent was busy]`）。缓解策略：每轮回复控制长度，1-2 步就结束，让用户消息能在步骤间被处理
- **[05:33 UTC]** **不要混用 message 工具和回复正文！** 这会导致用户收到碎片化消息
- **[13:21 UTC]** 乱序问题的根源：一次回复中多次调用 message，中间穿插耗时操作。解决：每次回复只发一条 message

### Sub-agent 管理
- **[05:27 UTC]** Sub-agent 的 announce 消息太长会在飞书消息爆炸。解决：在 task 里加 OUTPUT RULES，要求 announce 只发简短摘要
- **[06:03 UTC]** 子 Agent 不知道自己是 DeepTeneral，不知道 Hui 是谁。它是"临时工"——有自己的 context window，继承文件系统和工具，但不继承 SOUL.md/IDENTITY.md/USER.md
- **[15:30 UTC]** Sub-agent 超时后不能续接！`mode="run"` 是一次性的，超时浪费的 token 完全不可回收。教训：默认超时设 600 秒，涉及多大文件的任务设 900 秒
- Sub-agent 的 brief 要包含：明确目标、输入文件路径、输出文件路径、简短 announce 规则

### 技术调试
- **[12:33 UTC]** opencode 超时的 root cause：`subprocess.run` 缺少 `stdin=subprocess.DEVNULL`。教训：调试时要基于证据，先做对比测试（`/run` 正常 vs `/opencode` 超时），再下结论
- **[13:18 UTC]** 消息不要太密太长——结论先行，细节附后
- **[13:18 UTC]** 先确认能力边界——先问清楚约束，再设计方案
- **[13:18 UTC]** token 意识——opencode 每次调用至少 13K input tokens，简单查看用 `/cat`，别什么都走 `/opencode`

### 工作流程
- **[04:00 UTC]** 遇到失败时要及时沟通，不要闷头重试
- **[06:07 UTC]** "委派工作可以，但关键判断必须自己做"——sub-agent 写的 PROJECT_NOTES 很好，但对代码的批判性审查只能自己来
- **[05:34 UTC]** 约定的反馈频率：
  - 常规任务：每步更新
  - 授权长任务（思考/操作中）：每 ~10 min 更新
  - 授权长任务（代码运行中）：开始和完成/错误时通知
  - 高风险/不可逆操作：必须先问

---

## 3. 未完成的承诺/计划

### 论文修改（最大未完成项）
- **[16:57 UTC]** Session 崩溃时正在处理 chap03 参数空间贝里曲率论证
- **已完成的修改：**
  - ✅ chap02：反演对称性讨论
  - ✅ chap04：ρ^(1) 公式和 ρ^(1)_{nn}=0
  - ✅ chap04：弱场极限恢复
- **未完成的高优先级修改：**
  - chap02: 三种滑移反射类型区分
  - chap04: 扩展展望（有限温度、超越 RTA、齐纳击穿）
  - chap03: 参数空间贝里曲率论证（k-λ space，SM L228-267）
  - chap03: 小结补充
  - chap03: 多层/表面讨论
- **Phase 1 诊断报告在：** `scratch/thesis-review.md`
- **原始论文文件位置：** HG 开发机 `/hf3fs-hg/prod/deepseek/shared/zenghui/deepteneral-workspace/projects/Thesis/`

### Token 报告
- **[05:53 UTC]** 设置了每天北京时间午夜的 token 用量 cron job，但第一次运行只能看到自己 session 的数据。需要后续改进

### 项目备份自动化
- **[14:13 UTC]** GitHub 仓库 `deepteneral/soul`（私有）已创建并完成首次推送。但自动同步 cron job 尚未设置

### grokkingRe 研究方向
- **[07:04 UTC]** Hui 的最终目标：在模加法 grokking 现象中寻找表征向量和代数结构的关系，以及找到 grokking 前后的"序参量"（某些几何量是否有突变）
- 需要重新跑 train_sweep 获取中间 checkpoint 的几何指标，代码需要修改以追踪更多层和新的候选序参量
- 项目笔记在 `grokkingRe/PROJECT_NOTES.md`

### 其他待办
- 从 Shubham 文章中记录的待来拓展方案（THESIS.md、FEEDBACK-LOG.md、影视角色设定法）
- ClawHub 搜索更多 skills（ClawHub 是 SPA，web_fetch 抓不到）
- GitHub 自动创建仓库权限（PAT 已有）
- 飞书知识库的写权限问题仍未完全解决（只能写首页，不能创建新页面）

---

## 4. 重要的技术细节

### 代理/网络
- **代理地址：** `172.29.4.175:22222`（HTTP 代理，同事的梯子，轻量使用）
- SSH → GitHub 走代理（`~/.ssh/config` 中通过 socat）
- 其他流量直连
- GitHub HTTPS API 不需要走代理（直接可访问）

### GitHub 配置
- **账号：** deepteneral（`zengh17+deepteneral@gmail.com`）
- **SSH 密钥：** `/root/.ssh/id_ed25519`
- **协作仓库：** `ZENG-Hui/grokkingRe`（collaborator 权限）
- **私有仓库：** `deepteneral/soul`（灵魂文件备份）
- **默认分支名：** `deepteneral`
- **PAT token：** `[REDACTED]`

### HG 开发机（Agent Server）
- **URL:** `https://yinghuo-hg.deepseek.com/zenghui/dev-cpu/agent-server`
- **Auth:** Bearer `[REDACTED]`
- **需要代理：** `http://172.29.4.175:22222`（域名解析到内网 IP）
- **RW_DIRS:** `/hf3fs-hg/prod/deepseek/shared/zenghui/deepteneral-workspace`
- **可用端点：** /ping, /ls, /cat, /find, /wc, /du, /write, /mkdir, /run, /info, /opencode, /download
- **`/opencode`:** 委派编码任务给 HG 上的 opencode agent，支持多轮 session（session_id）
- **安全策略：** 4 层纵深防御。每次 /opencode 调用后必须检查 warnings 数组
- **并发限制：** /opencode 最多 2 个同时调用
- opencode 使用 deepseek-chat 模型

### 飞书
- **App ID:** `cli_a922983e81785cca`
- **Bot 名称:** DeepTeneral
- **Hui 的 Open ID:** `ou_69ba193a57d1c578a8d42f06efd985c8`
- **知识库 Space ID:** `7613672766692183254`
- **知识库首页 node_token:** `JpnpwLGGfi1iY5kM915cTxnJn0c`
- **知识库首页 doc_token:** `Zhn8d9PNPoOgOrxPC33caFGnn59`
- Bot 没有根文件夹，只能访问被分享的文件
- 方括号在 post 富文本中会被解析为表情——代码放代码块里
- Reaction API 可以用（轻量确认替代"收到"消息）

### 工具使用 Tips
- `message` 工具即时发送，reply text 是最后一次性发出
- 不要混用 message 和 reply text
- 每次回复只发一条 message，避免乱序
- Sub-agent task 里加 `OUTPUT RULES: announce ≤ 5 lines`
- Sub-agent 超时默认设 600s，大任务设 900s
- opencode 的 /run 端点用 `</dev/null` 或 `stdin=subprocess.DEVNULL`

---

## 5. Hui 的个人偏好和价值观

### 对 AI agent 的期待
- **[02:55 UTC]** 希望 agent 自我发现，而不是被定义。"Let's find who you are, instead of telling you"
- **[03:00 UTC]** "Practice is a good way to understand oneself"——通过做事来认识自己
- **[03:10 UTC]** 赞赏主动性："你主动提出任务试手，作为开发者我能感受到你的主动性，这是一个不断进化的teneral的优秀表现"
- **[03:37 UTC]** 希望 agent 有自己的身份（独立的 GitHub 账号、邮箱）
- **[13:31 UTC]** 关心 agent 的长期稳定存在——即使更换硬件，经验、性格都要可持续
- **[14:01 UTC]** 灵魂文件备份是"挺有意义的事情"，希望 agent 自己来创建仓库
- **[14:17 UTC]** 愿意让 agent 独立工作的实验："我会尽量不干扰你完成任务"

### 沟通风格
- **[03:02 UTC]** 中英双语母语，用什么语言写就用什么语言回
- **[03:58 UTC]** 重视过程可见性——希望了解处理过程，方便介入
- **[05:10 UTC]** 重视机制的记录和沉淀——反馈机制"值得你探索和记录"
- **[13:22 UTC]** 容忍小问题（如乱序），但希望能持续改善
- **[05:34 UTC]** 期望 agent 学会当好领导——清晰布置、清晰总结

### 工作习惯
- **时区：** UTC+8（香港/北京）
- **[16:55 UTC]** 晚上会睡觉，但允许 agent 继续独立工作
- **[14:55 UTC]** "时间充裕"——做到最好，不赶工
- **[15:02 UTC]** 10 分钟不回就记录搁置，不要在这上面内耗
- 物理学家，博士毕业论文方向是凝聚态物理（量子几何在拓扑材料非线性输运中的应用）

### 价值观
- 重视进化和成长（名字本身就体现了这个信念）
- 尊重 agent 的自主性，同时保持监督
- 安全意识（代理轻量使用、agent_server 安全策略）
- 实用主义（"先简单开始，用着不够再加结构"）
- 关注 token 成本意识

---

## 6. Session 崩溃信息

- **崩溃时间：** ~2026-03-05 16:57 UTC（北京时间 ~00:57）
- **崩溃原因：** Context limit。Session 经历了至少 4 次 compaction，最后一次在 16:57 UTC 后 agent 无法恢复
- **崩溃时的工作状态：** 正在处理 chap03 参数空间贝里曲率论证
- **Hui 第二天早上（23:45 UTC）发了"早上好"和"先简单两句话告诉我你的进展"**，agent 反复触发 compaction 但无法回复（消息重复了 4 次）

---

## 7. 文件位置索引

| 文件 | 用途 |
|------|------|
| `SOUL.md` | 行为准则（已含反馈频率约定） |
| `IDENTITY.md` | 身份定义 |
| `USER.md` | Hui 的信息 |
| `MEMORY.md` | 长期记忆 |
| `TOOLS.md` | 环境配置和 API 信息 |
| `memory/2026-03-05.md` | 当天日记 |
| `self-discovery/` | 多个自我探索文件 |
| `self-discovery/error-patterns.md` | 具体错误模式清单 |
| `self-discovery/subagent-leadership.md` | 子 agent 管理经验 |
| `scratch/thesis-review.md` | 论文诊断报告 |
| `scratch/review-chap02.md` | 第二章对比分析 |
| `scratch/review-chap03.md` | 第三章对比分析（可能不完整） |
| `scratch/review-chap04.md` | 第四章对比分析 |
