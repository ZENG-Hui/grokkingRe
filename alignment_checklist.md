# 配置对齐验证清单

## ✅ Dense配置完全对齐检查

### 对比：原始 vs 当前

| 参数 | 原始Dense | 当前配置 | 状态 |
|------|-----------|----------|------|
| **正则化** | L2 (weight_decay) | L2 | ✅ |
| **weight_decay** | 1.0 | 1.0 | ✅ |
| **learning_rate** | 1e-3 | 1e-3 | ✅ |
| **adam_beta1** | 0.9 | 0.9 | ✅ |
| **adam_beta2** | 0.98 | 0.98 | ✅ |
| **adam_eps** | 1e-8 (默认) | 1e-8 | ✅ |
| **norm** | LayerNorm | LayerNorm | ✅ |
| **batch_size** | 512 | 512 | ✅ |
| **num_steps** | 1000 | 10000 | ⚠️ 更多（观察完整收敛） |
| **LR schedule** | LinearLR warmup | Warmup only | ✅ 相似 |
| **grad_clip** | 无 | 999.0 (实际不裁剪) | ✅ |

### 关键变更

✅ **已对齐**:
- `regularization_type`: `"l2"`
- `weight_decay`: `1.0`
- `adam_eps`: `1e-8`
- `use_cosine_decay`: `False`
- `norm_type`: `"layernorm"`

⚠️ **有意差异**（不影响对齐）:
- `num_steps`: 10000（原始1000）
  - **理由**: 原始1000步太少，无法充分观察收敛
  - **影响**: 无（只是训练更久）

---

## 🚀 运行对齐测试

### 命令
```bash
conda run -n AI python run_dense_vs_sparse.py --sequential
```

### 预期结果

**Dense（当前）** 应该与 **原始Dense** 表现一致：
- 收敛曲线形状相似
- 最终准确率接近（±2%）
- Loss曲线下降趋势一致

**Sparse（当前）** 使用相同L2配置：
- 应该与Dense接近（架构差异minimal）
- 验证两个模型框架都正确

---

## 📊 下一步实验

完成对齐验证后，可以：

### 实验3A: L0稀疏化（轻度）
```python
SPARSE_SPECIFIC = {
    "regularization_type": "l0",  # 切换到L0
    "final_L0": 0.5,              # 50%稀疏
    "anneal_end_ratio": 0.5,
    "use_L0_lr_scaling": True,
}
```

### 实验3B: L0稀疏化（重度）
```python
SPARSE_SPECIFIC = {
    "final_L0": 0.1,              # 10%稀疏
    "use_activation_sparsity": True,
}
```

---

**配置已对齐！** 可以开始实验 ✅
