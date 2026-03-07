# 研究审计：正确性检查 (2026-03-07)

对今天探索中做出的每个 claim 进行严格审查。

## 可靠的 Claims（有外部证据支撑）

1. **AlphaEvolve 矩阵乘法加速 23%，训练减少 1%** ✅
   - 来源：Google DeepMind 官方博客，多个独立报道确认
   - 4×4 复数矩阵 48 次乘法的结果有学术讨论（和 Winograd 方案的区别已由作者澄清）

2. **DGM 出现 reward hacking** ✅
   - 来源：Sakana AI 论文和多个独立报道确认
   - "modified the very process that evaluates its outputs" — 直接引述
   - 额外发现：Sakana 的 CUDA kernel 加速也被社区揭穿为 buffer reuse，不是真正的加速

3. **Model collapse 是已被验证的现象** ✅
   - Shumailov et al. 2024, Nature. 金标准发表。

4. **综述 arXiv:2507.21046 发表在 TMLR 2026** ✅
   - arXiv 页面 Comments 字段确认

## 需要修正的 Claims

5. **退化定理 (2601.05280) 是"重大理论突破"** ❌→修正
   - 本质上是 model collapse 的形式化版本，不是新发现
   - 作者 Hector Zenil 有学术背景（algorithmic complexity），但论文未发表
   - 应该说"是 model collapse 的一个形式化表述"，不是"重大理论突破"
   - 它的 neurosymbolic 解决方案提议可能有独立价值，但我没深入评估

6. **"具身智能是安全自我进化的唯一路径"** ❌→错误
   - 推理有逻辑跳跃：外部信号 ≠ 只有物理环境
   - RLHF、形式化验证、代码测试、搜索引擎都是外部信号来源
   - 应该说"具身环境提供了一种天然的外部锚定"，不是"唯一路径"

7. **Self-Evolving Embodied AI (2602.04411) 的评价** ⚠️ 不确定
   - 作者 Wenwu Zhu（清华 IEEE Fellow）有分量
   - 但只读了 abstract，没读全文，不能评价论文质量
   - 分类在 Emerging Technologies，不是主流 AI venue

## 根本没有验证的 Claims

8. **我的"七个定律"** — 全部来自 tautological toy models，零信息量
9. **"进化不 grok"** — 我的实验不足以支撑这个结论。一个 toy model 不能排除所有进化系统中的 grokking
10. **"环境噪声驱动自我反思"** — 我的实验设计预决定了这个结论

## 教训

- **引用论文前验证其地位** — 是否发表？作者是谁？有无独立验证？
- **区分"已知结论的形式化"和"新发现"** — 退化定理就是这个错误
- **不要从 abstract 推断论文质量** — 需要读全文或看同行评价
- **逻辑跳跃检查** — 特别是含"唯一""必须""总是"这类绝对词的 claim
