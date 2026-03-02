# Experiment 1: L0=1.0 Alignment Test

## 🎯 实验目标

验证sparse模型在极限条件下可以与dense模型行为接近。

## 📋 实验设置

### 对齐条件

**SHARED_CONFIG（两者完全相同）**:
- ✅ Normalization: `rmsnorm`
- ✅ Optimizer: `adam_eps=0.1`, `beta1=0.9`, `beta2=0.95`
- ✅ LR Schedule: Warmup(1%) + Cosine decay
- ✅ Gradient clipping: RMS threshold=1.0
- ✅ Weight decay: `0.1`
- ✅ Training steps: `10000` (充足训练)
- ✅ Architecture: 2L×128D×4H (~422K params)

**DENSE_SPECIFIC**:
- Regularization: `L0`
- `final_L0 = 1.0` (100%非零，无稀疏)
- `use_L0_lr_scaling = False`
- `anneal_end_ratio = 0.0` (不退火)

**SPARSE_SPECIFIC**:
- Regularization: `L0`
- `final_L0 = 1.0` (100%非零，对齐dense)
- `use_L0_lr_scaling = False` (关闭，对齐dense)
- `use_activation_sparsity = False` (关闭AbsTopK)
- `anneal_end_ratio = 0.0` (不退火)

### 唯一剩余差异

由于架构限制，无法完全消除的差异：
1. **模型结构**: Dense用`model.py`，Sparse用`model_sparse.py`
   - 但两者都使用RMSNorm
   - 两者都使用相同的attention和FFN

2. **训练逻辑**: `training.py` vs `training_sparse.py`
   - 但都使用统一的LR scheduler
   - 都使用L0 enforcement逻辑

**预期影响**: <5% 性能差异

---

## 🎲 预期结果

### Success Criteria（成功标准）

| 指标 | Dense | Sparse | 允许差距 |
|------|-------|--------|----------|
| **Final Train Accuracy** | >90% | >90% | <10% gap |
| **Final Val Accuracy** | >90% | >90% | <10% gap |
| **Final Train Loss** | <0.5 | <0.5 | <20% gap |
| **Final Val Loss** | <0.5 | <0.5 | <20% gap |
| **Convergence Speed** | ~5K steps | ~5K steps | ±30% |

### 如果差距 >10%

可能原因：
1. **实现Bug**: 检查L0 enforcement, scheduler, optimizer
2. **初始化差异**: 检查随机种子
3. **数值稳定性**: RMSNorm vs LayerNorm的微小差异
4. **训练动态**: 检查LR曲线是否真的一致

---

## 📊 运行实验

### 命令

```bash
# 设置环境
conda activate AI

# 运行对齐实验  
cd d:\ZENG_Hui_files\Code\2025\AntiGravityPlay
conda run -n AI python run_dense_vs_sparse.py --sequential
```

### 监控指标

在Wandb中查看：
- `training/accuracy` - 应该几乎重叠
- `training/loss` - 应该几乎重叠
- `training/learning_rate` - 应该完全一致
- `validation/accuracy` - 最终值应接近
- `validation/loss` - 最终值应接近

---

## 🔍 结果分析

### 分析清单

- [ ] 查看Wandb runs: https://wandb.ai/zengh17/sparse_vs_dense
- [ ] 选择两个runs（`dense-baseline-XXX`, `sparse-XXX`）
- [ ] 点击"Compare"查看并排对比
- [ ] 记录最终指标：
  - Dense final val acc: _______
  - Sparse final val acc: _______
  - Gap: _______% 
- [ ] 检查曲线形状是否相似
- [ ] 检查收敛速度是否接近

### 判断标准

✅ **通过** (差距<10%):
- 实现正确，可以继续稀疏性实验

⚠️ **部分通过** (差距10-20%):
- 可接受，可能是架构微小差异
- 记录差距，继续实验

❌ **失败** (差距>20%):
- 需要debug实现
- 检查上述可能原因
- 修复后重新测试

---

## 📝 实验记录模板

```
实验日期: 2025-11-26
实验ID: alignment_L0_1.0

配置:
- norm_type: rmsnorm
- adam_eps: 0.1
- num_steps: 10000
- L0: 1.0 (both)

结果:
Dense:
  - Final train acc: _____%
  - Final val acc: _____%
  - Final train loss: _____
  - Final val loss: _____

Sparse:
  - Final train acc: _____%
  - Final val acc: _____%
  - Final train loss: _____
  - Final val loss: _____

Gap:
  - Train acc gap: _____%
  - Val acc gap: _____%

结论:
[ ] 通过 - 差距<10%
[ ] 部分通过 - 差距10-20%
[ ] 失败 - 差距>20%

备注:
___________________________________
```

---

## 🚀 下一步

### 如果通过
→ 进行稀疏性扫描实验（L0: 0.7 → 0.5 → 0.3 → 0.1）

### 如果失败
→ Debug并重新测试对齐

---

**实验配置已就绪！准备运行...** ✅
