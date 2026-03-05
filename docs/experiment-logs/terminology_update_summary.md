# Terminology Update & Alignment Test

## ✅ 重命名完成

### 更改内容

**之前**（误导性）:
```python
"regularization_type": "l0"  # ❌ 不准确
```

**现在**（准确）:
```python
"regularization_type": "l2-topk"  # ✅ L2 + Top-K强制稀疏化
```

---

## 📝 修改的文件

### 1. `run_dense_vs_sparse.py`
- Line 98: `"l0"` → `"l2-topk"`
- 添加了实验说明注释
- 配置了对齐测试（L0=1.0）

### 2. `training.py`
- Line 80: 支持`"l2-topk"`（同时保留`"l0"`向后兼容）

### 3. `config_sparse.py`  
- Line 30: 默认值改为`"l2-topk"`

---

## 🧪 当前测试配置（实验2D）

### Dense（基线）
```python
"regularization_type": "l2"      # 纯L2
"weight_decay": 1.0
"final_L0": 1.0                  # 不使用TopK
```

### Sparse（测试对齐）
```python
"regularization_type": "l2-topk"  # L2 + TopK
"weight_decay": 1.0               # L2部分相同
"final_L0": 1.0                   # TopK保留100%
"anneal_end_ratio": 0.0           # 不退火
"use_L0_lr_scaling": False        # 不缩放
"use_activation_sparsity": False  # 不使用激活稀疏
```

### 预期结果

**应该完全对齐**，因为：
1. Dense: 仅L2
2. Sparse: L2 + TopK(L0=1.0)
   - TopK在L0=1.0时`continue`跳过
   - 等价于仅L2

**验证**：
- Dense和Sparse的loss/accuracy曲线应该几乎重叠
- 如果有差异>2%，说明有bug

---

## ✅ 可用选项

现在`regularization_type`支持：

| 值 | 含义 | 使用场景 |
|----|------|---------|
| `"l2"` | 纯L2正则化 | Dense baseline |
| `"l2-topk"` | L2 + Top-K稀疏化 | Sparse训练 |
| `"l0"` | 同`"l2-topk"`（向后兼容） | 旧脚本 |

---

## 🚀 运行对齐测试

```bash
# 运行当前配置（实验2D）
conda run -n AI python run_dense_vs_sparse.py --sequential

# 预期：
# - Dense: 纯L2
# - Sparse: L2+TopK(L0=1.0) ≈ 纯L2
# - 两者应该完全对齐
```

### 检查点

访问Wandb查看：
- `training/loss` - 应该重叠
- `training/accuracy` - 应该重叠
- `sparsity/actual_sparsity` - Sparse应该是1.0（100%非零）

---

## 📋 下一步实验

验证对齐后，启用真正的稀疏化：

```python
# 修改 SPARSE_SPECIFIC:
"final_L0": 0.1,              # 改为10%稀疏
"anneal_end_ratio": 0.5,      # 启用退火
"use_L0_lr_scaling": True,    # 启用LR缩放
```

---

**术语已更正！对齐测试已配置！** ✅
