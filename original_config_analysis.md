# 原始Dense实验配置分析

## 🔍 原始实验使用的正则化

查看 `reference_implementations/dense/training.py` (lines 41-46):

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=(0.9, 0.98),
    weight_decay=config.weight_decay  # 👈 这里！
)
```

查看 `reference_implementations/dense/cli.py` (line 16):

```python
parser.add_argument("--weight_decay", type=float, default=1)
```

### 答案：**L2正则化（通过AdamW的weight_decay）**

```python
正则化类型：L2 (Weight Decay)
weight_decay = 1.0
```

---

## 📊 原始实验完整配置

### Optimizer
```python
optimizer = AdamW
learning_rate = 1e-3
betas = (0.9, 0.98)
weight_decay = 1.0        # L2正则化
eps = 1e-8                # 默认值（未指定）
```

### LR Scheduler
```python
scheduler = LinearLR
start_factor = 0.1        # 从0.1倍LR开始
total_iters = 9           # 9步后达到满LR
# → 简单的warmup，没有cosine decay
```

### Model
```python
Transformer(
    num_layers = 2
    dim_model = 128
    num_heads = 4
)
# → LayerNorm（默认）
# → 无稀疏性机制
```

### Training
```python
num_steps = 1000          # 默认1K步
batch_size = 512
device = "cuda"           # 默认GPU
```

---

## 🎯 与当前实验2的对比

| 参数 | 原始Dense | 当前实验2 | 差异 |
|------|-----------|----------|------|
| **正则化** | L2 (WD=1.0) | L0 (final_L0=1.0) | ⚠️ **不同！** |
| **weight_decay** | 1.0 | 0.1 | ⚠️ **不同！** |
| **adam_beta2** | 0.98 | 0.98 | ✅ 相同 |
| **adam_eps** | 1e-8（默认） | 1e-5 | ⚠️ 不同 |
| **norm** | LayerNorm | LayerNorm | ✅ 相同 |
| **LR schedule** | LinearLR warmup | Warmup+Cosine | ⚠️ 不同 |
| **num_steps** | 1000 | 10000 | ⚠️ 不同 |

---

## ✅ 如何对齐原始实验

### 方案A：完全对齐（推荐用于验证）

修改 `run_dense_vs_sparse.py`:

```python
SHARED_CONFIG = {
    # === Normalization ===
    "norm_type": "layernorm",  # ✅ 已对齐
    
    # === 训练配置 ===
    "batch_size": 512,
    "num_steps": 1000,         # 👈 改：从10000 → 1000
    "device": "cpu",           # 或 "cuda"
    
    # === 优化器配置 ===
    "learning_rate": 1e-3,
    "adam_beta1": 0.9,         # ✅ 已对齐
    "adam_beta2": 0.98,        # ✅ 已对齐
    "adam_eps": 1e-8,          # 👈 改：从1e-5 → 1e-8（默认值）
    
    # === LR调度 ===
    # 👈 改：模拟LinearLR的简单warmup
    "warmup_ratio": 0.009,     # 9步warmup (9/1000 ≈ 0.009)
    "use_cosine_decay": False, # 👈 改：关闭cosine
    "min_lr_ratio": 1.0,       # 保持满LR
    
    # === 其他 ===
    "grad_clip_rms": 999.0,    # 实际不裁剪（原始没有）
    "eval_every": 1,           # 每个epoch验证
}

DENSE_SPECIFIC = {
    # === 正则化 ===
    "regularization_type": "l2",  # 👈 改：从l0 → l2
    "weight_decay": 1.0,          # 👈 改：从0.1 → 1.0
    
    # L0参数（不使用）
    "final_L0": 1.0,
    "use_L0_lr_scaling": False,
}

SPARSE_SPECIFIC = {
    # 对于对齐测试，sparse也用L2
    "regularization_type": "l2",  # 👈 改：测试L2效果
    "weight_decay": 1.0,          # 👈 改：与dense一致
    
    "final_L0": 1.0,              # 不使用L0
    "use_activation_sparsity": False,
}
```

### 方案B：仅对齐核心参数（推荐用于实验）

如果想保留改进（warmup+cosine），只对齐关键参数：

```python
SHARED_CONFIG = {
    "norm_type": "layernorm",
    "adam_eps": 1e-8,          # 👈 对齐原始
    "num_steps": 1000,         # 👈 对齐原始（或保持10000）
    
    # 保留改进
    "use_cosine_decay": True,  # 保留（有助稳定）
}

DENSE_SPECIFIC = {
    "regularization_type": "l2",  # 👈 改为L2
    "weight_decay": 1.0,          # 👈 对齐原始
}
```

---

## 🧪 建议的实验序列

### 实验2B：L2 Baseline（对齐原始）

**目的**: 与原始dense实验对齐

**配置**:
```python
# DENSE_SPECIFIC
"regularization_type": "l2"
"weight_decay": 1.0

# SPARSE_SPECIFIC（也用L2作为对照）
"regularization_type": "l2"  
"weight_decay": 1.0
```

**预期**: Dense应该与原始实验结果接近

---

### 实验3：L2 vs L0对比

**目的**: 对比L2和L0两种正则化

**设置A** - L2 Sparse:
```python
SPARSE_SPECIFIC = {
    "regularization_type": "l2",
    "weight_decay": 1.0,
    # 无L0稀疏化
}
```

**设置B** - L0 Sparse:
```python
SPARSE_SPECIFIC = {
    "regularization_type": "l0",
    "weight_decay": 0.1,       # L0+L2结合
    "final_L0": 0.1,           # 10%稀疏
    "use_activation_sparsity": True,
}
```

**对比**: 
- L2能否也产生某种"隐式稀疏"？
- L0的显式稀疏性是否必要？

---

## 📋 修改清单

要完全对齐原始实验，需修改：

### run_dense_vs_sparse.py

**SHARED_CONFIG**:
- [ ] Line ~47: `"num_steps": 1000,`
- [ ] Line ~53: `"adam_eps": 1e-8,`
- [ ] Line ~57: `"use_cosine_decay": False,`

**DENSE_SPECIFIC**:
- [ ] Line ~70: `"regularization_type": "l2",`
- [ ] Line ~71: `"weight_decay": 1.0,`

**SPARSE_SPECIFIC** (如果也想用L2):
- [ ] Line ~81: `"regularization_type": "l2",`
- [ ] Line ~82: `"weight_decay": 1.0,`

---

## 🎯 推荐做法

**我的建议**：

1. **先运行实验2B（L2 baseline）**
   - 验证Dense能复现原始结果
   - 确认框架正确性

2. **然后运行实验3（L0稀疏化）**
   - 在验证过的baseline基础上
   - 测试L0的稀疏化效果

3. **可选：L2 vs L0对比**
   - 看L2能否隐式产生稀疏性
   - 理解L0的独特价值

---

**总结**: 
- ✅ 原始实验用的是 **L2正则化（weight_decay=1.0）**
- ✅ 当前用L0是为了显式控制稀疏度
- ✅ 建议先用L2对齐，再测试L0

需要我帮您修改配置文件吗？
