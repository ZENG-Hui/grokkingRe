# Experiment V2: Model Collapse Phase Transition

Created: 2026-03-07
Time budget: 30min (调研10min + 实验20min)

## 问题

Model collapse 是否存在 sharp phase transition？

当合成数据比例 α 从 0（纯真实数据）逐渐增加到 1（纯合成数据）时，
模型质量的退化是：
- (a) 平滑下降（连续退化）？
- (b) 存在临界点 α_c，低于它质量几乎不变，高于它突然崩溃（sharp transition）？

## 为什么这个问题不是 tautological

1. **我不知道答案** — 文献中有两种看法：
   - Strong Model Collapse (arXiv:2410.04840): 即使 1% 合成数据也有害 → 暗示无 threshold
   - Golden Ratio Weighting (arXiv:2502.18049): 存在最优混合策略 → 暗示有 sweet spot
   
2. **实验可能给出反直觉结果**：
   - 可能 transition 非常 sharp（像 Ising 模型的相变）
   - 可能完全平滑（无相变）
   - 可能取决于迭代次数（短期平滑，长期 sharp）
   - 可能取决于模型容量

3. **模型选择不预设结论**：
   - 使用 GMM（有足够复杂度展现非平凡行为）
   - 不手工设计 fitness landscape

## 实验设计

**Setup:**
- 真实数据：从一个 3-component GMM 采样
- 模型：用 EM 算法拟合 GMM
- 每一代：α 比例合成数据 + (1-α) 比例真实数据
- 迭代 T 代（模拟递归自训练）
- 测量：KL 散度 / log-likelihood 在 held-out 真实数据上

**扫描参数：**
- α ∈ [0, 0.05, 0.1, ..., 0.95, 1.0]（21 个点）
- T ∈ [1, 5, 10, 20, 50]（5 个迭代深度）
- 每个 (α, T) 组合重复 10 次取平均

**寻找什么：**
- 画 quality vs α 曲线（每个 T 一条线）
- 如果存在 phase transition：曲线应该有明显的拐点
- 如果是平滑退化：曲线应该是凹/凸的连续函数
- 附加：计算 d(quality)/dα 看是否有峰值（susceptibility analogue）

## 文件结构
```
experiment-v2/
├── README.md          # 本文件
├── notes.md           # 调研笔记
├── experiment.py      # 主实验代码
├── results.md         # 结果分析
└── figures/           # 可视化
```

## 参考文献
- Shumailov et al. 2024, Nature — Model collapse 原始论文
- arXiv:2410.04840 — Strong Model Collapse
- arXiv:2502.18049 — Golden Ratio Weighting
