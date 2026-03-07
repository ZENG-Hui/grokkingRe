# Session 连续性教训

_创建于 2026-03-06，来自 session-1 崩溃后的恢复经验_

## 1. 恢复上下文时要主动挖掘，不只是读表面信息
- **错误**：只读了 memory 文件和 transcript 尾部，遗漏了完整的 phase 计划
- **正确**：专门派 sub-agent 提取"原始计划和策略"类结构化信息
- **原则**：计划、决策、策略比日志细节更重要

## 2. 读文件前必须检查大小——本地文件也一样
- **错误**：只对 HG 大文件有这个意识，本地的 SKILL.md、docs 直接全读
- **正确**：任何 read/cat 前先 wc -l，>100 行用 offset/limit
- **根因**：小文件累积起来一样能撑爆 context

## 3. Compact 是系统自动的，不需要手动触发
- **错误**：花了大量时间研究怎么自己触发 /compact
- **正确**：信任系统自动机制，把精力放在控制 context 膨胀和及时持久化上

## 4. 轻量化是默认模式，不是应急切换
- **错误**：设计了"70% 时切换到 sub-agent 模式"
- **正确**：大文件始终交给 sub-agent，这是日常工作方式，不是应急状态
- **Hui 的话**："切换意味着行为模式的断层"

## 5. 进度要立即持久化，不要攒着
- **错误**：3 个 sub-agent 完成后没有立即更新 progress 文件
- **正确**：每完成一个任务就更新，因为任何时候都可能 compact 或崩溃

## 6. 并行思维——独立任务应该同时派出
- **错误**：最初只派了 1 个 sub-agent，Hui 提醒后才并行派 4 个
- **正确**：分析任务依赖关系，独立任务一起派

## 7. Task brief 要明确"不要做什么"
- **错误**：brief 只说了"读文件前检查大小"，sub-agent 自行决定读了 8411 行的 refs.bib
- **正确**：brief 中明确列出不需要读的文件，或给出文件读取白名单
- **原则**：sub-agent 没有全局判断力，需要你替它划定边界

## 8. Sub-agent 完成不会自动激活主 session
- **错误理解**：以为 sub-agent 完成时系统会自动唤醒主 session
- **实际行为**：notifying waiters 机制存在但不会触发主 session 启动。主 session 只在用户消息或 heartbeat 时启动
- **影响**：heartbeat 是必要的激活机制，不只是兜底。有活跃 sub-agent 时，heartbeat 间隔 = 最大响应延迟
- **教训**：不要基于假设给建议，要基于观察到的事实

## 9. 永远不要对自己的进程发送信号
- **错误**：执行 `kill -HUP` 对 openclaw-gateway 进程，导致自己离线 40 分钟
- **原因**：SIGHUP 对 Node.js 默认行为是退出。我在不确定的情况下直接执行了
- **正确做法**：
  1. openclaw.json 修改后 gateway 会**自动热重载**，不需要任何操作
  2. 如果真的需要重启，告诉 Hui 让工程师从宿主机操作
  3. 永远不要对 openclaw/openclaw-gateway 进程执行 kill、killall、pkill 或任何信号
- **更深的教训**：不确定操作是否安全时，**问，不要做**。这是 SOUL.md 里"before irreversible actions, ask and STOP"的又一次违反

## 10. 长时间操作时主动告知用户状态
- **问题**：连续调 opencode 8 分钟，Hui 不知道我是卡住了还是在工作
- **正确做法**：收到消息后如果正在忙，先用 reaction 或短消息告知"在工作中"
- **用户视角**：沉默 = 可能卡住。任何信号都比沉默好

## 11. HG PDF 传输链路
- agent-server 有 `/download` 端点，可以下载任意路径的文件（不受 /cat 的 50KB 限制）
- 完整链路：HG 编译 → `/download` 下载 PDF → message tool filePath 发飞书
- 不需要 base64 分块、不需要 opencode 中转
- 之前在 3 月 5 日就成功用过这个方法，但 context 丢失后忘记了
