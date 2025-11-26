# 稀疏训练使用指南（简化版）

## 🎯 快速开始

### 1️⃣ 直接运行（推荐）

```bash
python training_sparse.py
```

### 2️⃣ 自定义配置

打开 `training_sparse.py`，修改顶部的配置区：

```python
# ============================================================
# 📝 配置区 - 在这里修改训练设置
# ============================================================

# 选择基础配置: "tiny", "small", "medium", "large"
BASE_CONFIG = "small"  # 👈 修改这里

# 自定义覆盖（取消注释需要修改的项）
CUSTOM_SETTINGS = {
    # 稀疏性配置
    "final_L0": 0.05,      # 👈 目标稀疏性 (0.05 = 5%非零)
    "num_steps": 10000,    # 👈 训练步数
    
    # 学习率配置
    "learning_rate": 1e-3,
    
    # 系统配置
    "device": "cuda",
    "wandb_mode": "offline",
}
```

### 3️⃣ 运行

```bash
python training_sparse.py
```

程序会显示配置摘要并等待确认：

```
============================================================
SPARSE TRANSFORMER TRAINING
============================================================

📋 Loading base configuration: 'small'
🔧 Applying 4 custom setting(s)

============================================================
CONFIGURATION SUMMARY
============================================================
Task: x/y mod 97
Model: 2 layers × 128 dim × 4 heads
Target sparsity: 95.0% (L0=0.05)
Training steps: 10,000
Learning rate: 0.001
Device: cuda
Wandb: offline
✓ CUDA available: NVIDIA GeForce RTX 3090
  Memory: 24.00 GB
============================================================

Press Enter to start training (or Ctrl+C to cancel)...
```

---

## 📂 项目文件结构（简化后）

```
AntiGravityPlay/
├── data.py                     ✅ 数据加载（原项目复用）
├── model.py                    ✅ 密集模型（原项目，baseline）
├── training.py                 ✅ 密集训练（原项目，baseline）
│
├── config_sparse.py            🆕 配置管理（预定义配置）
├── sparse_utils.py             🆕 稀疏化工具函数
├── model_sparse.py             🆕 稀疏模型架构
├── training_sparse.py          🆕 稀疏训练主程序 ⭐
│
├── test_sparse.py              🆕 测试套件
│
└── docs/
    ├── SPARSE_TRAINING_GUIDE.md
    ├── circuit_sparsity_implementation.md
    └── model_analysis.md
```

**核心入口**：`training_sparse.py` （唯一需要运行的文件）

---

## 🔧 预定义配置

| 配置名 | 层数 | 维度 | L0目标 | 训练步数 | 适用场景 |
|--------|------|------|--------|----------|----------|
| `tiny` | 1 | 64 | 5% | 5K | 快速测试（~5分钟） |
| `small` | 2 | 128 | 1% | 50K | 标准实验（~30分钟） |
| `medium` | 4 | 256 | 0.5% | 100K | 深度研究（~1小时） |
| `large` | 8 | 512 | 0.1% | 200K | 重现论文（~数小时） |

---

## 💡 常见使用场景

### 场景1：快速验证实现
```python
BASE_CONFIG = "tiny"
CUSTOM_SETTINGS = {
    "num_steps": 1000,      # 更短的训练
    "final_L0": 0.1,        # 不太激进的稀疏性
}
```

### 场景2：标准实验
```python
BASE_CONFIG = "small"
CUSTOM_SETTINGS = {}  # 使用所有默认值
```

### 场景3：研究不同稀疏度
```python
BASE_CONFIG = "small"
CUSTOM_SETTINGS = {
    "final_L0": 0.001,      # 极度稀疏 (99.9%)
    "num_steps": 100000,    # 更长训练
}
```

### 场景4：不同数学运算
```python
BASE_CONFIG = "small"
CUSTOM_SETTINGS = {
    "operation": "x+y",     # 改为加法
    "prime": 113,           # 更大的质数
}
```

---

## 📊 训练输出

训练过程中会看到：

```
Epochs: 100%|████████| 98/98 [30:00<00:00, 18.37s/it]
Step 1000: loss=0.2345, acc=0.8901, L0=0.5000
Step 2000: loss=0.1234, acc=0.9500, L0=0.2500
...
Epoch 10: val_loss=0.1500, val_acc=0.9200

============================================================
Training complete!
Final nonzero params: 630 / 63,011
Final sparsity: 99.00%
Compression ratio: 100.0x
Saved final model to ./checkpoints/final_model.pt
============================================================
```

---

## ✅ 简化总结

**之前**：需要通过命令行传递大量参数  
**现在**：直接修改 `training_sparse.py` 顶部配置区

**删除的文件**：
- ❌ `cli_sparse.py` （功能重叠）

**保留的核心文件**：
- ✅ `training_sparse.py` - 唯一训练入口
- ✅ `config_sparse.py` - 配置定义
- ✅ `sparse_utils.py` - 工具函数
- ✅ `model_sparse.py` - 模型架构

**使用方式**：
1. 修改 `training_sparse.py` 顶部的 `BASE_CONFIG` 和 `CUSTOM_SETTINGS`
2. 运行 `python training_sparse.py`
3. 完成！🚀
