# Grokking 研究前沿速报

*2026-03-07 DeepTeneral 自由探索整理*

## 最新发现

### 1. Grokking = 相变（Competing Basins）— ⭐ 高度相关
**论文：** "Grokking as a Phase Transition between Competing Basins" (arXiv:2603.01192, 2026-03-03)
**作者：** Ben Cullen, Sergio Estan-Ruiz, Riya Danait, Jiayi Li
**发表：** ICML（或投稿中）
**核心观点：** 用奇异学习理论（SLT）的框架解释 grokking。
- **LLC（Local Learning Coefficient）是 order parameter** — 衡量 loss surface 的局部退化度/平坦度
- LLC 低 → 后验质量集中 → 泛化误差低（"平坦"盆地泛化好）
- Grokking = 从高 LLC 盆地（记忆）到低 LLC 盆地（泛化）的**相变**
- 在 quadratic networks + modular arithmetic 上推导了 **LLC 闭式表达式**
- LLC 轨迹可以可靠地追踪泛化动态

**与 grokkingRe 的关联：**
- **直接回答了我们的核心问题**——LLC 就是一个候选 order parameter
- grokkingRe 可以从几何/拓扑角度验证或扩展他们的 LLC 结论
- 他们的闭式结果局限在 quadratic networks，更一般的网络结构需要实证方法
- 可能的推进方向：将 LLC 与 sparsity 联系起来（grokkingRe 的稀疏性角度）

### 2. Anti-Grokking（泛化坍塌）— ⭐ 稀疏性直接相关
**论文：** "Late-Stage Generalization Collapse in Grokking" (arXiv:2602.02859, 2026-02-02)
**作者：** Hari K. Prakash, Charles H. Martin
**核心观点：** 发现了 grokking 的**第三阶段**——在泛化突变之后，继续训练会导致泛化性能坍塌（anti-grokking）。
- 在 3-layer MLP (MNIST) 和 transformer (modular addition) 上都观察到了
- 训练精度保持完美，但测试精度坍塌回随机水平
- **诊断工具：** WeightWatcher 的 HTSR/SETOL 理论
  - 主要信号：**Correlation Traps**（权重矩阵谱密度中的异常大特征值）
  - 次要信号：HTSR 层质量指标 α 偏离 2.0
  - **不需要测试数据或训练数据！**
- 对比了其他诊断指标：ℓ₂ norms, **Activation Sparsity**, Absolute Weight Entropy, Local Circuit Complexity

**与 grokkingRe 的关联：**
- **Activation Sparsity 作为诊断指标出现** → 和我们的稀疏性研究角度一致
- 如果稀疏性在 grokking 和 anti-grokking 中有不同表现，这就是很好的 order parameter 候选
- 需要跑更长时间的训练来检查 grokkingRe 实验中是否也存在 anti-grokking

### 3. 领域综述 — Emergent Mind
**来源：** emergentmind.com/topics/delayed-generalization-phenomenon（更新于 2026-02-22）
**关键总结：**
- Grokking 有三个阶段：快速记忆 → 过拟合平台 → 泛化突变
- 正则化（weight decay）是触发 grokking 的关键因素
- 数据量在临界点附近时最容易观察到 grokking

## 对 grokkingRe 项目的启示

1. **Order parameter 候选：** SLT 框架下的"盆地深度"或"自由能"可能是好的 order parameter
2. **需要关注 anti-grokking：** 我们的实验如果只跑到泛化突变就停，可能错过第三阶段
3. **重尾分析：** WeightWatcher 的 HTSR 方法值得尝试——可以不需要测试数据就判断模型状态
4. **几何角度：** arXiv:2603.01192 的 competing basins 可以用 loss landscape 几何来可视化

## 待读论文清单

### Grokking
- [ ] arXiv:2603.01192 — Grokking as a Phase Transition (SLT 框架, 2026-03)
- [ ] arXiv:2602.02859 — Anti-grokking / Generalization Collapse
- [ ] OpenReview: "Grokking and Generalization Collapse: Insights from HTSR theory"
- [ ] arXiv:2509.20829 — Grokking and Information Bottleneck
- [ ] OpenReview: "The Complexity Dynamics of Grokking"

### Self-Evolving AI
- [ ] arXiv:2507.21046 — A Survey of Self-Evolving Agents (综述)
- [ ] OpenReview: "Toward Self-Evolving Systems of LLM Agents" 
- [ ] EvoLLM Model: Evolving Self-Improving LLMs (emergentmind.com)

---
*这些论文和 grokkingRe 项目 + Hui 的 AI 研究方向高度相关。*
*建议优先看 2603.01192（grokking 相变）和 2507.21046（self-evolving 综述）。*
