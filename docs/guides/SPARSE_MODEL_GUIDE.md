# Sparse模型完整指南：从构造到推理

## 📋 目录

1. [模型构造](#模型构造)
2. [训练过程](#训练过程)
3. [推理过程](#推理过程)
4. [参数详解](#参数详解)
5. [完整示例](#完整示例)

---

## 🏗️ 模型构造

### 架构组成

```python
# Sparse模型 = Dense模型 + 稀疏化组件
SparseTransformer(
    # === 基础架构（与Dense相同）===
    num_layers=2,
    dim_model=128,
    num_heads=4,
    
    # === Sparse特有组件 ===
    norm_type="rmsnorm",              # RMSNorm归一化
    activation_sparsity_ratio=0.80,   # AbsTopK层
)
```

### 关键差异

| 组件 | Dense模型 | Sparse模型 | 作用 |
|------|----------|-----------|------|
| **Normalization** | LayerNorm | RMSNorm | 保留零值语义 |
| **Activation处理** | 无 | AbsTopK层 | 动态激活稀疏 |
| **训练后处理** | 无 | Top-K强制置零 | 权重稀疏 |

### AbsTopK层详解

```python
class AbsTopK(nn.Module):
    """激活稀疏层：只保留绝对值最大的k%激活"""
    
    def __init__(self, sparsity_ratio=0.80):
        self.ratio = sparsity_ratio  # 保留比例
    
    def forward(self, x):
        # 1. 计算要保留的数量
        k = int(x.size(-1) * self.ratio)  # 80% → 保留80%
        
        # 2. 找到第k大的阈值
        threshold = torch.topk(x.abs(), k).values[:, -1:]
        
        # 3. 创建mask：|x| >= threshold 的保留
        mask = (x.abs() >= threshold).float()
        
        # 4. 应用mask
        return x * mask  # 其余置0
```

**位置**：每个Attention和FFN层之后
```python
# DecoderBlock forward
x = self.attention(x)
x = AbsTopK(x, ratio=0.80)  # 👈 激活稀疏
x = self.ffn(x)
x = AbsTopK(x, ratio=0.80)  # 👈 激活稀疏
```

**参数控制**：`activation_sparsity_ratio`

---

## 🎓 训练过程

### 完整流程图

```
初始化
  ↓
┌──────────────────────────────────────┐
│ 阶段1: Dense训练 (Warmup)            │
│ - L0 = 1.0 (100%权重)                │
│ - 学习基础特征                        │
└──────────────────────────────────────┘
  ↓
┌──────────────────────────────────────┐
│ 阶段2: L0退火 (Annealing)            │
│ - L0: 1.0 → final_L0 (逐步稀疏)      │
│ - 权重"竞争"保留位置                  │
│ - LR可能动态调整 (if use_L0_lr_scaling)│
└──────────────────────────────────────┘
  ↓
┌──────────────────────────────────────┐
│ 阶段3: 稀疏微调 (Fine-tuning)        │
│ - L0 = final_L0 (固定稀疏度)         │
│ - 优化稀疏连接                        │
└──────────────────────────────────────┘
  ↓
保存模型
```

### 每个训练步的详细流程

```python
# === 一个训练步 (step) ===

# 1. 前向传播
output = model(input)
# ├─ 经过Attention
# ├─ 经过AbsTopK(ratio) ← 激活稀疏
# ├─ 经过FFN
# └─ 经过AbsTopK(ratio) ← 激活稀疏

# 2. 计算损失
loss = criterion(output, label)

# 3. 反向传播
loss.backward()

# 4. 梯度裁剪（可选）
if grad_clip_rms > 0:
    clip_grad_rms(model.parameters(), grad_clip_rms)

# 5. 优化器更新（L2在这里起作用）
optimizer.step()  # AdamW: weight *= (1 - lr*weight_decay)

# 6. Top-K权重稀疏化
if regularization_type == "l2-topk":
    # 计算当前L0
    current_L0 = calculate_current_L0(step, config)
    
    # 强制稀疏化
    enforce_weight_sparsity(model, current_L0, min_connections)
    # → 只保留Top-K权重，其余置0

# 7. 学习率调度
if use_L0_lr_scaling:
    scheduler.step()  # 可能根据L0调整LR
```

### 参数在训练中的作用

#### 1. `final_L0` 和 `initial_L0`

**作用**：定义权重稀疏的起点和终点

**示例** (final_L0=0.80, initial_L0=1.0, num_steps=2000):
```
Step 0:    L0 = 1.0   → 保留100%权重
Step 500:  L0 = 0.95  → 保留95%权重
Step 1000: L0 = 0.90  → 保留90%权重
Step 1600: L0 = 0.80  → 保留80%权重（达到final_L0）
Step 2000: L0 = 0.80  → 保持80%
```

#### 2. `anneal_end_ratio`

**作用**：控制退火在训练的前X%完成

**公式**：
```python
anneal_end_step = total_steps * anneal_end_ratio

if step < anneal_end_step:
    # 线性退火
    progress = step / anneal_end_step
    current_L0 = initial_L0 + (final_L0 - initial_L0) * progress
else:
    # 保持final_L0
    current_L0 = final_L0
```

**示例** (anneal_end_ratio=0.80, total_steps=2000):
```
退火阶段: Step 0 - 1600  (80% * 2000 = 1600)
微调阶段: Step 1600 - 2000

Step 0:    L0 = 1.0
Step 800:  L0 = 0.9  (退火50%)
Step 1600: L0 = 0.8  (退火完成)
Step 2000: L0 = 0.8  (微调)
```

#### 3. `use_L0_lr_scaling`

**作用**：根据当前L0动态调整学习率

**公式**：
```python
if use_L0_lr_scaling:
    effective_lr = base_lr * (1 / √current_L0)
```

**示例** (base_lr=1e-3):
```
L0=1.0  → lr = 1e-3 * 1/√1.0  = 1e-3   (1.0x)
L0=0.80 → lr = 1e-3 * 1/√0.80 = 1.12e-3 (1.12x)
L0=0.25 → lr = 1e-3 * 1/√0.25 = 2e-3   (2.0x)
L0=0.1  → lr = 1e-3 * 1/√0.1  = 3.16e-3 (3.16x)
L0=0.01 → lr = 1e-3 * 1/√0.01 = 1e-2   (10x)
```

**为什么需要**：
- 稀疏模型容量降低（参数少了）
- 需要更大LR让有限参数快速学习
- 补偿信息容量损失

#### 4. `activation_sparsity_ratio`

**作用**：控制AbsTopK层保留多少激活

**效果**：
```python
ratio = 0.80  # 保留80%

input  = [0.9, 0.7, 0.5, 0.3, 0.1]
                ↓ AbsTopK
output = [0.9, 0.7, 0.5, 0.3, 0  ]  # 最小的20%被置0
```

**训练和推理都生效**（与Dropout不同！）

#### 5. `weight_decay`

**作用**：L2正则化强度

**机制**：
```python
# 在optimizer.step()内部
weight = weight - lr * gradient - lr * weight_decay * weight
#                 ↑梯度更新        ↑L2正则化
```

**配合Top-K**：
- L2让权重保持小值（防止过拟合）
- Top-K强制稀疏（置零）
- 两者互补

#### 6. `min_connections`

**作用**：每个神经元至少保留X个非零权重

**防止死神经元**：
```python
# 假设一层有100个输入，10个输出
# final_L0 = 0.01 → 只保留1%权重 = 10个

# 不使用min_connections:
neuron_1: [0.5, 0, 0, 0, ...] # 只有1个连接
neuron_2: [0, 0, 0, 0, ...]   # 0个连接（死了！）

# 使用min_connections=4:
neuron_1: [0.5, 0.3, 0.2, 0.1, ...]  # 至少4个
neuron_2: [0.4, 0.3, 0.2, 0.1, ...]  # 至少4个
```

---

## 🚀 推理过程

### 推理流程

```python
# 推理时的前向传播
model.eval()
with torch.no_grad():
    output = model(input)
```

### 与训练的区别

| 方面 | 训练 | 推理 | 是否一致？ |
|------|------|------|----------|
| **权重** | Top-K稀疏 | Top-K稀疏（已固化） | ✅ 一致 |
| **AbsTopK** | 激活 | 激活 | ✅ 一致 |
| **梯度** | 计算 | 不计算 | - |
| **优化器** | 更新 | 不更新 | - |

**关键点**：
- ✅ 权重已经稀疏（训练时置0的仍为0）
- ✅ AbsTopK层仍然工作（保留ratio%激活）
- ✅ 推理使用的是训练时发现的"电路"

### 推理示例

```python
# 训练后的权重（已稀疏）
weight_matrix = [
    [0.8, 0, 0.6, 0, 0],      # 80%是0
    [0, 0.5, 0, 0.7, 0],
    ...
]

# 推理时前向传播
x = [1.0, 2.0, 3.0, 4.0, 5.0]
                ↓
y = weight_matrix @ x  # 稀疏矩阵运算（快！）
                ↓
y = [0.8, 0.7, ...]
                ↓
y = AbsTopK(y, ratio=0.80)  # 激活稀疏（与训练一致）
                ↓
output = [0.8, 0.7, 0, ...]
```

---

## 📊 参数详解总表

### 架构参数

| 参数 | 类型 | 默认值 | 作用 | 影响阶段 |
|------|------|--------|------|---------|
| `num_layers` | int | 2 | Transformer层数 | 构造 |
| `dim_model` | int | 128 | 模型维度 | 构造 |
| `num_heads` | int | 4 | 注意力头数 | 构造 |
| `norm_type` | str | "rmsnorm" | 归一化类型 | 构造 |

### 稀疏化参数

| 参数 | 类型 | 范围 | 作用 | 影响阶段 |
|------|------|------|------|---------|
| `regularization_type` | str | "l2"/"l2-topk" | 正则化方法 | 训练 |
| `final_L0` | float | 0.0-1.0 | 目标权重稀疏度 | 训练 |
| `initial_L0` | float | 0.0-1.0 | 起始权重稀疏度 | 训练 |
| `anneal_end_ratio` | float | 0.0-1.0 | 退火结束位置 | 训练 |
| `activation_sparsity_ratio` | float | 0.0-1.0 | 激活保留比例 | 训练+推理 |
| `min_connections` | int | 1+ | 最小连接数 | 训练 |

### 优化参数

| 参数 | 类型 | 典型值 | 作用 | 影响阶段 |
|------|------|-------|------|---------|
| `weight_decay` | float | 0.1-1.0 | L2正则化强度 | 训练 |
| `use_L0_lr_scaling` | bool | True/False | LR动态缩放 | 训练 |
| `learning_rate` | float | 1e-3 | 基础学习率 | 训练 |
| `warmup_ratio` | float | 0.01 | Warmup比例 | 训练 |
| `grad_clip_rms` | float | 1.0 | 梯度裁剪 | 训练 |

---

## 💡 完整示例

### 配置示例

```python
# 实验：中度稀疏模型
SPARSE_CONFIG = {
    # === 架构 ===
    "num_layers": 2,
    "dim_model": 128,
    "num_heads": 4,
    "norm_type": "rmsnorm",
    
    # === 训练 ===
    "num_steps": 10000,
    "batch_size": 512,
    "learning_rate": 1e-3,
    
    # === 稀疏化 ===
    "regularization_type": "l2-topk",
    "weight_decay": 1.0,
    
    # 权重稀疏：100% → 10%
    "final_L0": 0.1,              # 目标10%权重
    "initial_L0": 1.0,            # 从100%开始
    "anneal_end_ratio": 0.5,      # 前5000步退火
    "use_L0_lr_scaling": True,    # 启用LR缩放
    "min_connections": 4,         # 最小4个连接
    
    # 激活稀疏：固定25%
    "use_activation_sparsity": True,
    "activation_sparsity_ratio": 0.25,  # 保留top 25%
}
```

### 训练时间线

```
Step 0 (0%):
├─ L0 = 1.0 (100%权重)
├─ LR = 1e-3 * 1.0 = 1e-3
└─ AbsTopK保留25%激活

Step 2500 (25%):
├─ L0 = 0.55 (55%权重，退火中)
├─ LR = 1e-3 * 1.35 ≈ 1.35e-3
└─ AbsTopK保留25%激活

Step 5000 (50%):
├─ L0 = 0.1 (10%权重，退火完成)
├─ LR = 1e-3 * 3.16 ≈ 3.16e-3
└─ AbsTopK保留25%激活

Step 7500 (75%):
├─ L0 = 0.1 (10%权重，微调)
├─ LR = 1e-3 * 3.16 ≈ 3.16e-3
└─ AbsTopK保留25%激活

Step 10000 (100%):
├─ L0 = 0.1 (10%权重，完成)
├─ LR → 0 (cosine decay)
└─ AbsTopK保留25%激活
```

### 最终模型

```python
# 权重稀疏度
params_total = 422,000
params_nonzero = 42,200  (10%)
params_zero = 379,800    (90%)

# 推理时
- 使用稀疏权重（10%非零）
- AbsTopK仍激活（25%激活）
- 发现的"电路"：10%权重 + 25%激活路径
```

---

## 🎯 常见场景配置

### 场景1：对齐测试
```python
# 目标：验证L0=1.0等价于Dense
"final_L0": 1.0,
"anneal_end_ratio": 0.0,
"use_L0_lr_scaling": False,
"activation_sparsity_ratio": 1.0,
```

### 场景2：轻度稀疏
```python
# 目标：保留50%权重
"final_L0": 0.5,
"anneal_end_ratio": 0.5,
"use_L0_lr_scaling": False,
"activation_sparsity_ratio": 0.25,
```

### 场景3：论文配置
```python
# 目标：极度稀疏（1%权重）
"final_L0": 0.01,
"anneal_end_ratio": 0.5,
"use_L0_lr_scaling": True,
"activation_sparsity_ratio": 0.25,
"min_connections": 4,
```

---

## ✅ 总结

### 关键理解

1. **两种稀疏性独立**：
   - 权重稀疏（Top-K）：永久修改参数
   - 激活稀疏（AbsTopK）：每次前向传播

2. **训练=推理**：
   - AbsTopK在训练和推理都工作
   - 发现的电路在推理时重现

3. **参数协同**：
   - `final_L0` + `anneal_end_ratio` 控制退火
   - `use_L0_lr_scaling` 补偿容量损失
   - `weight_decay` + Top-K 协同稀疏化

### 调参建议

**新手**：
```python
final_L0 = 0.5             # 温和稀疏
anneal_end_ratio = 0.5     # 标准退火
use_L0_lr_scaling = False  # 不需要
activation_sparsity_ratio = 0.25  # 论文推荐
```

**进阶**：
```python
final_L0 = 0.1             # 重度稀疏
anneal_end_ratio = 0.5     # 标准退火
use_L0_lr_scaling = True   # 启用LR缩放
activation_sparsity_ratio = 0.25  # 论文推荐
min_connections = 4        # 防止死神经元
```

**极限**：
```python
final_L0 = 0.01            # 极度稀疏
anneal_end_ratio = 0.5     # 标准退火
use_L0_lr_scaling = True   # 必需
activation_sparsity_ratio = 0.25  # 论文推荐
min_connections = 4        # 必需
num_steps = 30000          # 需要更多步数
```

---

**文档完成！** 🎉

查看配置文件：[`run_dense_vs_sparse.py`](file:///d:/ZENG_Hui_files/Code/2025/AntiGravityPlay/run_dense_vs_sparse.py)
