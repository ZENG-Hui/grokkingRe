# Experiment 03: Sparsity Sweep - 稀疏化扫描实验

## 🎯 实验目标

在Dense baseline基础上，系统化地测试不同稀疏度对模型性能的影响，绘制**性能-稀疏度权衡曲线**。

---

## 📖 稀疏化训练过程详解

### 什么是稀疏化训练？

稀疏化训练通过**L0正则化**强制模型只保留最重要的权重连接，其他权重被置为0。

### 训练过程的三个阶段

```
Stage 1: Dense训练 (L0=1.0)
├─ 步数: 0% - warmup%
├─ 所有权重都活跃
└─ 模型学习任务基础特征

Stage 2: L0退火 (1.0 → final_L0)
├─ 步数: warmup% - anneal_end_ratio%
├─ 逐步减少非零权重
├─ 权重竞争保留位置
└─ 学习率动态调整 (如果use_L0_lr_scaling=True)

Stage 3: 稀疏训练 (固定final_L0)
├─ 步数: anneal_end_ratio% - 100%
├─ L0固定在目标值
├─ 微调保留的稀疏连接
└─ 达到最终性能
```

**可视化**：
```
L0  ▲
1.0 │████████\           Stage 3: 稀疏微调
    │         \          ├─────────────────►
0.5 │          ████\     
    │     Stage 2 \      Stage 2: 退火
0.1 │              ████████████████████
    │              ▲
    │   Stage 1    │ anneal_end_ratio (50%)
0.0 └──────────────┴────────────────────► steps
        warmup(1%)
```

---

## 🔧 关键参数详解

### 1. **final_L0** - 目标稀疏度

**作用**: 控制最终保留多少比例的非零权重

**取值范围**: 0.0 - 1.0
- `1.0` = 100%非零（dense baseline）
- `0.5` = 50%非零（轻度稀疏）
- `0.1` = 10%非零（重度稀疏，论文推荐）
- `0.01` = 1%非零（极度稀疏，论文目标）

**效果**:
```python
final_L0 = 0.1
# → 模型参数: 422K total
# → 非零参数: ~42K (10%)
# → 稀疏率: 90%
```

**实验建议**: 从大到小逐步测试
```
L0 = 1.0 → 0.7 → 0.5 → 0.3 → 0.1 → 0.05 → 0.01
```

---

### 2. **anneal_end_ratio** - 退火结束比例

**作用**: 控制L0退火过程在训练的前X%完成

**取值范围**: 0.0 - 1.0
- `0.0` = 不退火（立即使用final_L0）
- `0.5` = 前50%步数退火（**推荐**）
- `0.8` = 前80%步数退火（慢速退火）

**公式**:
```python
anneal_end_step = total_steps * anneal_end_ratio

if step < anneal_end_step:
    current_L0 = initial_L0 - (initial_L0 - final_L0) * (step / anneal_end_step)
else:
    current_L0 = final_L0
```

**示例**（total_steps=5000, anneal_end_ratio=0.5）:
```
Step 0:    L0 = 1.0
Step 1250: L0 = 0.55  (退火中)
Step 2500: L0 = 0.1   (退火完成)
Step 5000: L0 = 0.1   (保持)
```

**实验建议**: 固定为0.5（论文设置）

---

### 3. **use_L0_lr_scaling** - L0学习率缩放

**作用**: 根据当前L0动态调整学习率

**公式**:
```python
if use_L0_lr_scaling:
    effective_lr = base_lr * (1.0 / sqrt(current_L0))
```

**效果**:
```
L0=1.0  → lr_scale = 1.0x   (dense时正常学习率)
L0=0.25 → lr_scale = 2.0x   (稀疏时增大学习率)
L0=0.1  → lr_scale = 3.16x  (更稀疏，学习率更大)
L0=0.01 → lr_scale = 10x    (极度稀疏，学习率显著增大)
```

**为什么要缩放**？
- 稀疏模型信息容量降低
- 需要更大的学习率来快速调整剩余权重
- 论文发现这能加速收敛

**实验建议**: 
- L0 ≥ 0.3: 可以关闭（False）
- L0 < 0.3: 建议开启（True）

---

### 4. **use_activation_sparsity** - 激活稀疏性

**作用**: 使用AbsTopK强制激活向量稀疏

**机制**:
```python
# 在每个attention和FFN后应用
def AbsTopK(x, k_ratio=0.25):
    k = int(x.size(-1) * k_ratio)  # 保留25%
    topk_vals, topk_idx = torch.topk(x.abs(), k)
    mask = zeros_like(x)
    mask.scatter_(-1, topk_idx, 1.0)
    return x * mask  # 只保留top-k，其余置0
```

**与权重稀疏的区别**:
- **权重稀疏**: 永久修改模型参数（训练中置0）
- **激活稀疏**: 前向传播时动态置0（不影响权重）

**效果**:
- 进一步降低信息流
- 强迫模型学习更清晰的电路结构
- 可能降低性能（信息瓶颈）

**实验建议**:
- 先测试**仅权重稀疏**（False）
- 再测试**权重+激活稀疏**（True）
- 对比两者性能差异

---

### 5. **activation_sparsity_ratio** - 激活稀疏比例

**作用**: 控制保留多少比例的激活

**取值**: 
- `0.25` = 保留top 25%激活（**论文设置**）
- `0.5` = 保留top 50%激活
- `1.0` = 不稀疏（等同关闭）

---

### 6. **min_connections** - 最小连接数

**作用**: 每个神经元至少保留X个非零输入连接

**取值**:
- `1` = 至少1个连接（宽松约束）
- `4` = 至少4个连接（**论文设置**）

**为什么需要**？
- 防止某些神经元完全断开
- 保持网络连通性
- 特别重要当L0很小时（0.01）

---

## 📝 实验设置示例

### 实验3A: 轻度稀疏 (L0=0.5)

**目的**: 温和的稀疏化，测试性能是否下降

**配置**:
```python
SPARSE_SPECIFIC = {
    "final_L0": 0.5,              # 50%非零
    "initial_L0": 1.0,            # 从dense开始
    "anneal_end_ratio": 0.5,      # 前50%步数退火
    "use_L0_lr_scaling": False,   # 轻度稀疏不需要
    
    "use_activation_sparsity": False,  # 先测试仅权重稀疏
    "activation_sparsity_ratio": 0.25,
    "min_connections": 1,
}
```

**预期**: 性能下降<5%

---

### 实验3B: 中度稀疏 (L0=0.3)

**配置**:
```python
SPARSE_SPECIFIC = {
    "final_L0": 0.3,              # 30%非零
    "anneal_end_ratio": 0.5,
    "use_L0_lr_scaling": True,    # 开启学习率缩放
    
    "use_activation_sparsity": False,
    "min_connections": 1,
}
```

**预期**: 性能下降5-15%

---

### 实验3C: 重度稀疏 (L0=0.1)

**配置**:
```python
SPARSE_SPECIFIC = {
    "final_L0": 0.1,              # 10%非零（论文推荐）
    "anneal_end_ratio": 0.5,
    "use_L0_lr_scaling": True,
    
    "use_activation_sparsity": True,   # 开启激活稀疏
    "activation_sparsity_ratio": 0.25,
    "min_connections": 4,              # 增加最小连接
}
```

**预期**: 性能下降15-30%

**建议训练步数**: 5000-10000步

---

### 实验3D: 极度稀疏 (L0=0.01)

**配置**:
```python
SPARSE_SPECIFIC = {
    "final_L0": 0.01,             # 1%非零（论文目标）
    "anneal_end_ratio": 0.5,
    "use_L0_lr_scaling": True,
    
    "use_activation_sparsity": True,
    "activation_sparsity_ratio": 0.25,
    "min_connections": 4,
}
```

**预期**: 性能下降30-50%或更多

**建议训练步数**: 10000-30000步

---

## 🚀 实验执行流程

### Step 1: 准备

1. **确认Dense baseline已完成**
   - 作为100%非零的对照组
   - 记录最终accuracy/loss

2. **决定稀疏度序列**
   - 推荐: `1.0 → 0.5 → 0.3 → 0.1`
   - 扩展: `→ 0.05 → 0.01`

### Step 2: 修改配置

在`run_dense_vs_sparse.py`中修改**SPARSE_SPECIFIC**:

```python
# 只需修改这4-5个参数
"final_L0": 0.1,                    # 👈 主要修改
"anneal_end_ratio": 0.5,            # 👈 从0.0改为0.5
"use_L0_lr_scaling": True,          # 👈 从False改为True
"use_activation_sparsity": True,    # 👈 根据需要开启
"min_connections": 4,               # 👈 L0<0.1时改为4
```

### Step 3: 运行实验

```bash
conda run -n AI python run_dense_vs_sparse.py --sequential
```

### Step 4: 记录结果

创建实验记录表格：

| L0 | Act Sparse | Final Acc (%) | Acc Drop (%) | Params (K) | Notes |
|----|-----------|--------------|-------------|-----------|-------|
| 1.0 | No | 95.2 | - | 422 | Baseline |
| 0.5 | No | 94.8 | 0.4 | 211 | 轻度稀疏 |
| 0.3 | No | 93.5 | 1.7 | 127 | 中度稀疏 |
| 0.1 | Yes | 89.2 | 6.0 | 42 | 重度稀疏 |
| 0.01 | Yes | ? | ? | 4.2 | 极度稀疏 |

### Step 5: Wandb对比

访问: https://wandb.ai/zengh17/sparse_vs_dense

1. 选择多个runs（不同L0）
2. 点击"Compare"
3. 观察:
   - `training/accuracy`曲线
   - `sparsity/target_L0`退火过程
   - `training/learning_rate`（如果use_L0_lr_scaling=True）

---

## 📊 分析维度

### 1. 性能-稀疏度曲线

**绘制**:
```
Accuracy ▲
100%   ●─────●──────●
        \             \
90%      ●──────────●───●
          \              \
80%        \              ●
            \
0%          └───────────────────► L0
          1.0  0.5  0.3  0.1  0.01
```

**观察**:
- 性能何时开始明显下降？
- "拐点"在哪里？（性价比最优点）

### 2. 训练动态

**观察Wandb曲线**:
- 退火阶段是否平滑？
- L0固定后是否继续改善？
- 是否出现震荡？

### 3. 稀疏模式

**运行可视化**:
```bash
python visualize_sparsity.py --model checkpoints/sparse_L0_0.1.pt
```

**观察**:
- 哪些层更稀疏？
- attention还是FFN更重要？
- 是否出现结构化模式？

---

## ⚠️ 常见问题

### Q1: 稀疏训练loss不下降

**可能原因**:
- L0太激进（试试更大的值）
- 训练步数不够（增加到10K+）
- anneal太快（增加anneal_end_ratio）

**解决**:
```python
"final_L0": 0.3,          # 从0.1改为0.3
"num_steps": 10000,       # 增加步数
"anneal_end_ratio": 0.7,  # 更慢的退火
```

### Q2: Dense和Sparse性能差距很大

**这是正常的！** 稀疏化会降低模型容量。

**关注**:
- L0=0.5时差距应该<5%
- L0=0.1时差距可能15-30%
- 如果L0=0.5就差距>10%，检查实现

### Q3: 训练时间太长

**优化**:
```python
"num_steps": 2000,        # 快速测试用2K
"eval_every": 10,         # 减少验证频率
"device": "cuda",         # 如果有GPU
```

---

## 🎯 成功标准

### 验证稀疏化有效

- ✅ Wandb中能看到`sparsity/actual_sparsity`逐渐下降到final_L0
- ✅ `sparsity/nonzero_params`显著减少
- ✅ 性能下降在可接受范围内

### 后续研究

完成实验3后，可以:
1. 分析稀疏模式（哪些权重被保留）
2. 测试不同任务（x-y, x*y）
3. 尝试不同的稀疏化策略
4. 研究电路结构

---

**实验指南完成！** 🎉  
祝实验顺利！
