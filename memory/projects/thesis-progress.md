# 论文修改进度

_完整计划来源：scratch/thesis-original-plan.md_

## 工作优先级（诊断后确认的执行顺序）

### 第一优先：2-4 章高优先级内容补充 🔄 进行中
- ✅ chap02：反演对称性讨论（上个 session 完成）
- ✅ chap04：ρ^(1) 公式和 ρ^(1)_{nn}=0（上个 session 完成）
- ✅ chap04：弱场极限恢复（上个 session 完成）
- [patch] chap03：参数空间贝里曲率论证 — 审查通过，sub-agent 正在应用
- [patch] chap02：三种滑移反射类型区分 — 审查通过，sub-agent 正在应用
- [patch] chap04：扩展展望 — 审查通过，sub-agent 正在应用
- [patch] chap03：多层/表面讨论 + 3.4 小结 — 审查通过，sub-agent 正在应用
- [ ] chap02：紧束缚模型（SM 中可能缺失源材料，需确认）

### 第二优先：重写/扩展第一章绪论 ⬜
- chap01 当前 133 行，目标 250+ 行
- 加深核心概念背景、文献综述深度、方法论章节重写

### 第三优先：扩展第五章总结与展望 ⬜
- chap05 当前 77 行，目标 120-150 行
- 提炼贯穿全文的核心洞察，展望给出具体方案

### 第四优先：中优先级内容 + 语言润色 ⬜
- 各章中优先级缺失内容（见 scratch/thesis-review.md）
- 整体语言风格统一
- 中文写作自然度提升

### 第五优先：收尾 ⬜
- 英文摘要重写
- 中文摘要完善
- 致谢
- 术语一致性（quantum metric → 量子度规）
- 转写计划注释清理

## Hui 的关键指导

- **严谨性优先** — 不编造展望
- **连接词正常** — 物理学家中文带英文痕迹是正常的
- **内容丰满是核心问题** — 即使相比原始论文也不够
- **小问题直接改** — 不用问
- **做到自己认为的最好** — 时间充裕
- **摘要/致谢放最后**

## 技术信息
- 论文文件：HG 开发机 `/hf3fs-hg/.../deepteneral-workspace/projects/Thesis/`
- 编译：`bash build.sh`
- PDF 下载：agent_server `/download` 端点
- Patch 文件：HG `scratch/` + 本地 `scratch/`
- 审查笔记：`scratch/patch-review-notes.md`
