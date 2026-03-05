# Sparse Training Project Guide

## 📚 全面项目指南

本文档整合了项目的完整指南，包括对比实验、Wandb可视化和项目结构说明。

---

## 第一部分：项目概览

### 项目目标

实现论文 "Automatically Identifying Local and Global Circuits with Linear Computation Graphs" 中的稀疏训练方法，用于模运算任务。

### 核心组件

| 类型 | 文件 | 说明 |
|------|------|------|
| **Dense训练** | `training.py`, `model.py` | 原始密集模型基线 |
| **Sparse训练** | `training_sparse.py`, `model_sparse.py` | 稀疏训练实现 |
| **配置** | `config_sparse.py` | 稀疏训练配置 |
| **工具** | `sparse_utils.py` | 稀疏性工具函数 |
| **数据** | `data.py` | 模运算数据生成 |
| **可视化** | `visualize_sparsity.py` | 权重稀疏性可视化 |
| **对比** | `run_dense_vs_sparse.py` | 自动化对比脚本 |

---

## 第二部分：Dense vs Sparse对比实验

### 快速开始

```bash
# 方法1：使用对比脚本（推荐）
conda run -n AI python run_dense_vs_sparse.py --sequential

# 方法2：单独运行dense
conda run -n AI python cli.py --num_steps 10000

# 方法3：单独运行sparse  
conda run -n AI python training_sparse.py
```

### 配置对比实验

编辑 `run_dense_vs_sparse.py`：

```python
SHARED_CONFIG = {
    # 模型架构（必须相同）
    "num_layers": 2,
    "dim_model": 128,
    "num_heads": 4,
    
    # 训练配置
    "num_steps": 10000,     # 👈 修改训练步数
    "batch_size": 512,
    "learning_rate": 1e-3,
    "eval_every": 1,        # 👈 验证频率
}

SPARSE_SPECIFIC = {
    "final_L0": 0.1,        # 👈 目标稀疏度（0.1=10%非零）
    "activation_sparsity_ratio": 0.25,
}

WANDB_CONFIG = {
    "mode": "online",       # 👈 online/offline
}
```

### 在Wandb中查看结果

1. 运行完成后访问：https://wandb.ai/zengh17/sparse_vs_dense
2. 选择两个runs（`dense-baseline-XXX`和`sparse-XXX`）
3. 点击**"Compare"**按钮
4. 查看对比图表

### 关键对比指标

| 指标 | Dense | Sparse | 说明 |
|------|-------|--------|------|
| `training/accuracy` | 训练准确率 | 训练准确率 | 主要性能指标 |
| `validation/accuracy` | 验证准确率 | 验证准确率 | 泛化能力 |
| `training/loss` | 训练损失 | 训练损失 | 收敛速度 |
| - | - | `sparsity/target_L0` | L0退火曲线 |
| - | - | `sparsity/actual_sparsity` | 实际稀疏率 |
| - | - | `training/learning_rate` | 动态学习率 |

---

## 第三部分：Wandb可视化详解

### 初次设置

1. **登录Wandb**（如果使用online模式）:
   ```bash
   wandb login
   # 输入API key: 2695076dd0f87f2dba081c03fdc9cfa84acef643
   ```

2. **配置模式**（修改脚本顶部）:
   ```python
   wandb_mode = "online"   # 实时上传
   wandb_mode = "offline"  # 本地保存，稍后sync
   wandb_mode = "disabled" # 完全禁用
   ```

### 主要图表说明

#### 1. Training Metrics
- **training/loss**: 应逐渐下降至接近0
- **training/accuracy**: 应逐渐上升至>90%
- **training/learning_rate**: 
  - Dense: 固定1e-3
  - Sparse: 动态变化（随L0退火增大）

#### 2. Validation Metrics
- **validation/loss**: 泛化损失
- **validation/accuracy**: 最终性能指标

#### 3. Sparsity Metrics（仅Sparse）
- **sparsity/target_L0**: 目标L0退火曲线
  ```
  1.0 ████\
  0.5     ████\
  0.1         ████████
      0     50%    100% steps
  ```
- **sparsity/actual_sparsity**: 实际稀疏率
- **sparsity/nonzero_params**: 非零参数数量

### 离线模式使用

```bash
# 1. 离线训练
wandb_mode="offline" python training_sparse.py

# 2. 稍后同步到云端
wandb sync wandb/run-XXXXXX-XXXXX
```

---

## 第四部分：项目结构对比

### 原始项目 vs 稀疏训练项目

| 功能 | 原始项目 | 稀疏训练项目 |
|------|---------|------------|
| **模型定义** | `model.py` (LayerNorm) | `model_sparse.py` (RMSNorm + AbsTopK) |
| **训练脚本** | `training.py` | `training_sparse.py` |
| **CLI** | `cli.py` | 配置在脚本顶部 |
| **配置** | argparse | `config_sparse.py` (dataclass) |
| **Wandb** | 基础logging | 增强logging (10+ metrics) |
| **稀疏性** | ❌ 无 | ✅ Top-K + L0退火 |

### 文件关系

```
原始Dense训练:
cli.py → training.py → model.py
         ↓
      data.py

稀疏Sparse训练:
training_sparse.py → model_sparse.py
    ↓                     ↓
config_sparse.py    sparse_utils.py
    ↓
  data.py

对比实验:
run_dense_vs_sparse.py
    ↓
  training.py + training_sparse.py
```

---

## 第五部分：常见任务

### 修改稀疏度

```python
# config_sparse.py 或 training_sparse.py 顶部
CUSTOM_SETTINGS = {
    "final_L0": 0.05,  # 5%非零（更稀疏）
    "final_L0": 0.2,   # 20%非零（更宽松）
}
```

### 修改模型大小

```python
SHARED_CONFIG = {
    "num_layers": 4,     # 增加层数
    "dim_model": 256,    # 增加维度
    "num_heads": 8,      # 增加头数
}
```

### 可视化权重稀疏性

```bash
# 训练后生成可视化
conda run -n AI python visualize_sparsity.py \
    --model checkpoints/final_model.pt \
    --output visualizations
```

生成的图表：
1. `weight_sparsity_heatmaps.png` - 权重矩阵热图
2. `layer_sparsity_stats.png` - 每层统计
3. `weight_distribution.png` - 权重分布
4. `attention_connection_pattern.png` - 注意力模式

详见 `visualizations/VISUALIZATION_GUIDE.md`

---

## 第六部分：性能优化建议

### 如果Sparse收敛太慢

1. **增加训练步数**
   ```python
   "num_steps": 20000  # 从2000增加到20000
   ```

2. **放松稀疏度**
   ```python
   "final_L0": 0.3  # 从0.1改为0.3
   ```

3. **延长退火时间**
   ```python
   "anneal_end_ratio": 0.7  # 从0.5改为0.7（70%步数后才达到final_L0）
   ```

4. **关闭激活稀疏（验证权重稀疏效果）**
   ```python
   "use_activation_sparsity": False
   ```

### 如果想对齐Dense和Sparse

参见 `DENSE_VS_SPARSE_ANALYSIS.md` 第二部分的对齐策略。

---

## 第七部分：故障排查

### 常见问题

#### Q: `ModuleNotFoundError: No module named 'torch'`
```bash
# 使用conda run而不是直接python
conda run -n AI python training_sparse.py
```

#### Q: Wandb要求登录
```python
# 改为离线模式
"wandb_mode": "offline"
```

#### Q: 验证频率太高，训练太慢
```python
# 修改验证频率
"eval_every": 10  # 每10个epoch验证一次
```

#### Q: 稀疏训练loss降不下来
- 检查：是否训练步数太少（建议10K+）
- 检查：L0是否太激进（试试0.3而不是0.1）
- 参考：`DENSE_VS_SPARSE_ANALYSIS.md`

---

## 相关文档

- **详细分析**: `DENSE_VS_SPARSE_ANALYSIS.md` - Dense和Sparse的详细技术对比
- **论文实现**: `circuit_sparsity_implementation.md` - 论文要点总结
- **快速入门**: `QUICKSTART.md` - 快速上手
- **可视化指南**: `visualizations/VISUALIZATION_GUIDE.md` - 如何理解可视化

---

**最后更新**: 2025-11-25  
**项目版本**: Stage 1 - Core Sparse Training ✅
