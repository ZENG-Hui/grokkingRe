# 电路稀疏性论文：训练时强制模型稀疏化技术详解

## 一、核心思想

论文通过在训练过程中**强制权重稀疏性（Weight Sparsity）**来训练更可解释的Transformer模型。大部分权重被约束为零，使得L0范数（非零参数的数量）保持很小。最稀疏的模型约有**1/1000的权重非零**。

## 二、架构设计

### 2.1 基础架构
- **模型类型**：GPT-2风格的decoder-only Transformer
- **典型配置**：
  - `nlayer = 8`（层数）
  - `dmodel = 2048`（隐藏维度）
  - `nctx = 256`（上下文长度）
  - `dhead = 16`（注意力头维度，较小以提高单语义性）

### 2.2 关键架构修改

#### (1) RMSNorm替代LayerNorm
```python
# 使用RMSNorm而不是LayerNorm
# 原因：确保零值在残差流中具有特权意义
# 优势：可以将所有归一化权重折叠到MLP/注意力权重中而不改变权重L0
```

#### (2) AbsTopK激活函数
在模型的多个位置应用**AbsTopK**激活函数，强制激活稀疏性：
- 保留按幅值排序的前k大的值
- 将其他值置零
- 通常设置 `k = 1/4 * dimension`（即**1/4非零激活**）

应用位置：
- Query、Key、Value计算后
- 注意力操作后
- MLP的fc层后
- MLP的proj层后
- 残差流读取位置

```python
# AbsTopK伪代码
def AbsTopK(x, k):
    # x: 输入张量
    # k: 保留的top-k数量
    threshold = torch.kthvalue(torch.abs(x), x.shape[-1] - k + 1).values
    mask = torch.abs(x) >= threshold
    return x * mask
```

#### (3) 其他改进
- **注意力汇点（Attention Sinks）**：每个头的可学习注意力分母偏置
- **Bigram表**：单独的密集 `dvocab × dvocab` 矩阵，避免稀疏参数记忆bigram频率
- **无位置编码**：大多数实验不使用位置编码（对损失影响中性）
- **独立的嵌入/反嵌入矩阵**：embedding和unembedding不共享权重

## 三、权重稀疏性优化方法（核心技术）

### 3.1 Top-K权重选择策略

**关键机制**：在每个训练步骤后，将每个权重矩阵中除最大幅值的条目外的所有其他条目置零。

```python
# 伪代码：强制L0权重稀疏性
def enforce_L0_sparsity(weight_matrix, target_sparsity_ratio):
    """
    在每个训练步骤后应用
    
    Args:
        weight_matrix: 权重矩阵
        target_sparsity_ratio: 目标非零比例（例如 0.001 表示1/1000非零）
    """
    # 计算每个矩阵应保留的非零元素数量
    num_nonzero = int(weight_matrix.numel() * target_sparsity_ratio)
    
    # 获取绝对值最大的num_nonzero个元素的阈值
    flat_weights = weight_matrix.abs().flatten()
    threshold = torch.kthvalue(flat_weights, flat_weights.numel() - num_nonzero + 1).values
    
    # 创建mask：只保留绝对值大于等于阈值的权重
    mask = weight_matrix.abs() >= threshold
    
    # 应用mask
    weight_matrix.data = weight_matrix.data * mask
    
    return weight_matrix
```

**重要特点**：
- 所有权重矩阵具有**相同的非零比例**
- 梯度和Adam动量保持**密集**（不修改）
- 只在应用AdamW更新后零化权重

### 3.2 L0退火（L0 Annealing）

训练过程中逐渐增加稀疏性：

```python
# L0退火策略
def get_target_L0(training_progress, initial_L0=1.0, final_L0=0.001, anneal_end=0.5):
    """
    linear annealing from dense to sparse
    
    Args:
        training_progress: 训练进度 [0, 1]
        initial_L0: 初始L0比例（1.0 = 完全密集）
        final_L0: 最终L0比例（例如 0.001）
        anneal_end: 退火结束的训练进度比例（默认0.5，即前50%）
    """
    if training_progress >= anneal_end:
        return final_L0
    
    # 线性插值
    progress_ratio = training_progress / anneal_end
    current_L0 = initial_L0 + (final_L0 - initial_L0) * progress_ratio
    
    return current_L0
```

**关键参数**：
- 在训练的前**50%**线性退火L0（对于最大、最稀疏的模型增加到80%）
- 从完全密集（L0=1.0）逐渐退火到目标L0

### 3.3 防止死神经元

```python
# 最小连接数约束
MIN_CONNECTIONS = 4

def enforce_L0_with_min_connections(weight_matrix, target_L0, min_connections=4):
    """
    避免神经元或注意力通道的连接数少于min_connections
    """
    # 对于每个神经元/通道，确保至少保留min_connections个非零值
    # 这会略微增加实际L0，但减少死神经元
    
    # 实现细节省略，但核心思想是：
    # 1. 按幅值排序选择top-k权重
    # 2. 对于连接数<min_connections的神经元，强制保留min_connections个连接
    pass
```

### 3.4 学习率调度

```python
# 学习率调度（Sharkfin Schedule）
def get_learning_rate(step, total_steps, base_lr, current_L0):
    """
    结合warmup-decay和L0相关的缩放
    """
    warmup_ratio = 0.01  # 前1%为warmup
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Warmup阶段
    if step < warmup_steps:
        warmup_factor = step / warmup_steps
    else:
        # Decay阶段（余弦退火或类似）
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        warmup_factor = 0.5 * (1 + math.cos(math.pi * progress))
    
    # L0相关缩放：较小的L0需要较大的学习率
    L0_factor = 1.0 / math.sqrt(current_L0)
    
    return base_lr * warmup_factor * L0_factor
```

**关键发现**：
- **1% warmup**：对于高学习率的稳定性至关重要
- **L0因子**：学习率按 `1/√L0` 缩放，因为更稀疏的模型需要更大的学习率

### 3.5 梯度裁剪

```python
# RMS梯度裁剪
def clip_grad_rms(parameters, max_rms=1.0):
    """
    裁剪梯度的均方根到max_rms
    对于训练稳定性至关重要
    """
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm_sq += (p.grad ** 2).sum()
    
    rms = math.sqrt(total_norm_sq / sum(p.numel() for p in parameters))
    
    if rms > max_rms:
        scale = max_rms / rms
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)
```

### 3.6 优化器配置

```python
# AdamW配置
optimizer = AdamW(
    model.parameters(),
    lr=lr,  # 需要sweep
    betas=(0.9, 0.95),  # β1=0.9, β2=0.95
    weight_decay=0.1,   # λ=0.1
    eps=0.1             # ϵ=0.1（注意：比常规的1e-8大很多）
)
```

## 四、完整训练循环伪代码

```python
def train_sparse_transformer(model, dataloader, config):
    optimizer = AdamW(model.parameters(), **config.optimizer_args)
    total_steps = len(dataloader) * config.num_epochs
    
    for step, batch in enumerate(dataloader):
        # 1. 计算当前的L0目标
        progress = step / total_steps
        current_L0 = get_target_L0(progress, config.final_L0)
        
        # 2. 计算当前学习率
        lr = get_learning_rate(step, total_steps, config.base_lr, current_L0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 3. 前向传播
        logits = model(batch['input_ids'])
        loss = cross_entropy(logits, batch['labels'])
        
        # 4. 反向传播
        loss.backward()
        
        # 5. 梯度裁剪
        clip_grad_rms(model.parameters(), max_rms=1.0)
        
        # 6. 优化器步骤
        optimizer.step()
        optimizer.zero_grad()
        
        # 7. 强制权重稀疏性（关键步骤！）
        for name, param in model.named_parameters():
            if should_enforce_sparsity(name):  # 所有权重和偏置
                enforce_L0_with_min_connections(
                    param, 
                    target_L0=current_L0,
                    min_connections=4
                )
```

## 五、剪枝方法（发现任务特定电路）

训练完成后，使用梯度的结构化剪枝找到最小电路：

### 5.1 节点定义
- **节点**：单个神经元、注意力通道、残差通道读取或残差通道写入
- **边**：权重矩阵中的非零条目

### 5.2 剪枝算法

```python
def prune_circuit(model, task_data, target_loss=0.15, mean_activations):
    """
    学习一组掩码τ_i来门控节点
    
    x_i -> x_i ⊙ σ(τ_i)
    
    其中σ是Heaviside阶跃函数
    """
    # 初始化可学习掩码参数
    masks = {node_i: nn.Parameter(torch.ones(node_dim)) 
             for node_i in model.nodes}
    
    mask_optimizer = Adam(masks.values())
    
    for step in range(pruning_steps):
        logits = forward_with_masks(model, task_data, masks, mean_activations)
        
        # 联合目标：任务损失 + 电路大小
        task_loss = cross_entropy(logits, task_data.labels)
        circuit_size = sum(sigmoid(mask).sum() for mask in masks.values())
        
        total_loss = task_loss + lambda_size * circuit_size
        
        # 使用sigmoid导数替代梯度通过Heaviside函数反向传播
        # （类似Straight-Through Estimator）
        total_loss.backward()
        mask_optimizer.step()
    
    # 最终：应用阶跃函数获得二值掩码
    final_masks = {k: (sigmoid(v) > 0.5).float() for k, v in masks.items()}
    return final_masks

def forward_with_masks(model, data, masks, mean_activations):
    """
    带掩码的前向传播
    删除的节点被平均消融（冻结为预训练分布上的平均激活）
    """
    # 实现细节：
    # x_i = x_i ⊙ heaviside(τ_i) + mean_activations[i] ⊙ (1 - heaviside(τ_i))
    pass
```

### 5.3 平均消融
- 删除节点 = 将其激活冻结为预训练分布上的平均值
- 不同于零消融或随机消融
- 更好地保持模型行为

## 六、关键技术要点总结

| 技术点 | 具体实现 | 目的 |
|--------|----------|------|
| **Top-K权重选择** | 每步后只保留每个矩阵的top-k权重 | 强制L0稀疏性 |
| **L0退火** | 训练前50%从密集线性退火到目标L0 | 稳定优化 |
| **AbsTopK激活** | 在所有节点位置应用，保留1/4激活 | 强制激活稀疏性 |
| **学习率缩放** | lr ∝ 1/√L0 | 补偿稀疏性的影响 |
| **梯度裁剪** | RMS裁剪到1.0 | 训练稳定性 |
| **最小连接数** | 每个神经元至少4个连接 | 减少死神经元 |
| **RMSNorm** | 替代LayerNorm | 使零值有特权意义 |
| **结构化剪枝** | 梯度学习二值掩码 | 发现任务电路 |

## 七、效率考虑

⚠️ **重要局限**：稀疏训练需要**100-1000倍**更多的计算资源

原因：
1. 权重密集存储（保持梯度和Adam动量密集）
2. 未使用稀疏内核（优化复杂）
3. 需要更多训练步骤达到相同能力

改进方向：
- 更好的重初始化技术（减少死神经元）
- 稀疏内核实现
- 权重稀疏的MoE模型
