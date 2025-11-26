"""
权重稀疏性可视化工具

加载训练好的稀疏模型，生成以下可视化：
1. 权重矩阵热图（显示非零权重位置）
2. 每层稀疏性统计
3. 权重值分布
4. 连接模式分析

使用方法：
    python visualize_sparsity.py --model checkpoints/final_model.pt
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

from model_sparse import SparseTransformer
from config_sparse import SparseTrainingConfig


def load_model(checkpoint_path: str):
    """加载训练好的模型"""
    print(f"Loading model from {checkpoint_path}...")
    
    # PyTorch 2.6+ requires weights_only=False for custom classes
    checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
    
    # 获取配置
    config = checkpoint.get('config')
    if config is None:
        raise ValueError("No config found in checkpoint")
    
    # 创建模型
    from model_sparse import create_sparse_model
    model = create_sparse_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded successfully")
    return model, config


def visualize_weight_matrices(model, save_dir: Path):
    """可视化所有权重矩阵的稀疏模式"""
    print("\n" + "="*60)
    print("1. 权重矩阵稀疏模式可视化")
    print("="*60)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有权重矩阵
    weight_matrices = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            weight_matrices[name] = param.detach().cpu().numpy()
    
    # 为每个矩阵创建热图
    n_matrices = len(weight_matrices)
    n_cols = 3
    n_rows = (n_matrices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, weight) in enumerate(weight_matrices.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # 创建二值化版本（0=黑色/零权重，1=白色/非零权重）
        binary_weight = (weight != 0).astype(float)
        
        # 绘制热图
        im = ax.imshow(binary_weight, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 计算稀疏性
        sparsity = (weight != 0).sum() / weight.size
        
        # 设置标题
        short_name = name.replace('model.', '').replace('.weight', '')
        ax.set_title(f'{short_name}\n{weight.shape}\nSparsity: {sparsity:.2%}', 
                     fontsize=10)
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Output Dimension')
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for idx in range(n_matrices, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = save_dir / 'weight_sparsity_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def visualize_layer_sparsity(model, save_dir: Path):
    """可视化每层的稀疏性统计"""
    print("\n" + "="*60)
    print("2. 每层稀疏性统计")
    print("="*60)
    
    save_dir = Path(save_dir)
    
    # 收集每层的稀疏性统计
    layer_stats = {}
    for name, param in model.named_parameters():
        # 获取层名（去掉.weight或.bias后缀）
        layer_name = '.'.join(name.split('.')[:-1])
        
        if layer_name not in layer_stats:
            layer_stats[layer_name] = {
                'total': 0,
                'nonzero': 0,
                'params': []
            }
        
        total = param.numel()
        nonzero = (param != 0).sum().item()
        
        layer_stats[layer_name]['total'] += total
        layer_stats[layer_name]['nonzero'] += nonzero
        layer_stats[layer_name]['params'].append(name.split('.')[-1])
    
    # 计算每层的稀疏率
    layers = []
    sparsity_ratios = []
    total_params = []
    
    for layer_name, stats in sorted(layer_stats.items()):
        layers.append(layer_name)
        sparsity_ratios.append(stats['nonzero'] / stats['total'] if stats['total'] > 0 else 0)
        total_params.append(stats['total'])
    
    # 创建柱状图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图1: 稀疏率
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    bars = ax1.barh(range(len(layers)), sparsity_ratios, color=colors)
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels([l.replace('model.', '') for l in layers], fontsize=9)
    ax1.set_xlabel('Sparsity Ratio (Non-zero Parameters)')
    ax1.set_title('Sparsity per Layer', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    
    # 在柱子上标注数值
    for i, (bar, ratio) in enumerate(zip(bars, sparsity_ratios)):
        ax1.text(ratio + 0.02, i, f'{ratio:.2%}', 
                va='center', fontsize=8)
    
    # 图2: 参数数量
    bars2 = ax2.barh(range(len(layers)), total_params, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels([l.replace('model.', '') for l in layers], fontsize=9)
    ax2.set_xlabel('Total Parameters')
    ax2.set_title('Parameters per Layer', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 在柱子上标注数值
    for i, (bar, total) in enumerate(zip(bars2, total_params)):
        ax2.text(total + max(total_params)*0.01, i, f'{total:,}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    output_path = save_dir / 'layer_sparsity_stats.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()
    
    # 打印文字统计
    print("\nDetailed Statistics:")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Total Params':>12} {'Non-zero':>12} {'Sparsity':>10}")
    print("-" * 80)
    for layer_name, stats in sorted(layer_stats.items()):
        ratio = stats['nonzero'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{layer_name:<40} {stats['total']:>12,} {stats['nonzero']:>12,} {ratio:>9.2%}")
    print("-" * 80)


def visualize_weight_distribution(model, save_dir: Path):
    """可视化权重值分布"""
    print("\n" + "="*60)
    print("3. 权重值分布")
    print("="*60)
    
    save_dir = Path(save_dir)
    
    # 收集所有非零权重
    all_nonzero_weights = []
    for name, param in model.named_parameters():
        weights = param.detach().cpu().numpy().flatten()
        nonzero_weights = weights[weights != 0]
        all_nonzero_weights.extend(nonzero_weights)
    
    all_nonzero_weights = np.array(all_nonzero_weights)
    
    # 创建分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 直方图
    axes[0, 0].hist(all_nonzero_weights, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Non-zero Weight Distribution (Linear Scale)', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. 对数尺度直方图
    axes[0, 1].hist(all_nonzero_weights, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Frequency (Log Scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Non-zero Weight Distribution (Log Scale)', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. 绝对值分布
    abs_weights = np.abs(all_nonzero_weights)
    axes[1, 0].hist(abs_weights, bins=100, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('|Weight Value|')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Absolute Weight Value Distribution', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 统计信息
    axes[1, 1].axis('off')
    stats_text = f"""
    Statistics:
    
    Total Weights: {len(all_nonzero_weights):,}
    
    Mean: {all_nonzero_weights.mean():.6f}
    Median: {np.median(all_nonzero_weights):.6f}
    Std Dev: {all_nonzero_weights.std():.6f}
    
    Min: {all_nonzero_weights.min():.6f}
    Max: {all_nonzero_weights.max():.6f}
    
    Abs Mean: {abs_weights.mean():.6f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    output_path = save_dir / 'weight_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def visualize_connection_pattern(model, save_dir: Path):
    """可视化连接模式（针对attention层）"""
    print("\n" + "="*60)
    print("4. 注意力层连接模式")
    print("="*60)
    
    save_dir = Path(save_dir)
    
    # 找到所有attention层的QKV权重（只要2D的）
    attn_weights = {}
    for name, param in model.named_parameters():
        if 'self_attn' in name and 'weight' in name and param.dim() == 2:
            attn_weights[name] = param.detach().cpu().numpy()
    
    if not attn_weights:
        print("⚠ No 2D attention weights found, skipping...")
        return
    
    # 为每个attention权重创建可视化
    n_weights = len(attn_weights)
    fig, axes = plt.subplots(1, min(n_weights, 3), figsize=(15, 5))
    if n_weights == 1:
        axes = [axes]
    elif n_weights < 3:
        axes = axes[:n_weights]
    
    for idx, (name, weight) in enumerate(list(attn_weights.items())[:3]):
        ax = axes[idx]
        
        # 显示权重的连接模式
        binary_weight = (weight != 0).astype(float)
        
        im = ax.imshow(binary_weight, cmap='Blues', aspect='auto')
        ax.set_title(name.replace('model.', '').replace('.weight', ''), fontsize=10)
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Output Dimension')
        
        # 计算并显示统计信息
        sparsity = binary_weight.sum() / binary_weight.size
        ax.text(0.02, 0.98, f'Sparsity: {sparsity:.2%}', 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_path = save_dir / 'attention_connection_pattern.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


def create_summary_report(model, config, save_dir: Path):
    """创建总结报告"""
    print("\n" + "="*60)
    print("5. 生成总结报告")
    print("="*60)
    
    save_dir = Path(save_dir)
    
    # 统计信息
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    sparsity_ratio = nonzero_params / total_params
    compression_ratio = total_params / nonzero_params
    
    # 创建报告
    report = f"""
# 稀疏模型可视化报告

## 模型配置
- 任务: {config.operation} mod {config.prime}
- 层数: {config.num_layers}
- 模型维度: {config.dim_model}
- 注意力头数: {config.num_heads}
- 目标L0: {config.final_L0} ({(1-config.final_L0)*100:.1f}% 稀疏度目标)

## 稀疏性统计

- **总参数数**: {total_params:,}
- **非零参数**: {nonzero_params:,}
- **实际稀疏率**: {sparsity_ratio:.4f} ({sparsity_ratio*100:.2f}%)
- **稀疏度**: {(1-sparsity_ratio)*100:.2f}%
- **压缩比**: {compression_ratio:.1f}x

## 生成的可视化

1. `weight_sparsity_heatmaps.png` - 所有权重矩阵的稀疏模式
2. `layer_sparsity_stats.png` - 每层稀疏性统计
3. `weight_distribution.png` - 权重值分布分析
4. `attention_connection_pattern.png` - 注意力层连接模式

## 观察

- 稀疏训练成功将模型参数压缩了 {compression_ratio:.1f} 倍
- 实际稀疏率 {sparsity_ratio:.2%} 接近目标 {config.final_L0:.2%}
"""
    
    report_path = save_dir / 'README.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Report saved to {report_path}")
    
    # 也在终端打印
    print("\n" + "="*60)
    print("Model Sparsity Summary")
    print("="*60)
    print(f"Total Params:     {total_params:>12,}")
    print(f"Non-zero Params:  {nonzero_params:>12,}")
    print(f"Sparsity Ratio:   {sparsity_ratio:>12.2%}")
    print(f"Sparsity:         {(1-sparsity_ratio):>12.2%}")
    print(f"Compression:      {compression_ratio:>12.1f}x")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='可视化稀疏模型的权重')
    parser.add_argument('--model', type=str, default='checkpoints/final_model.pt',
                        help='模型checkpoint路径')
    parser.add_argument('--output', type=str, default='visualizations',
                        help='输出目录')
    args = parser.parse_args()
    
    print("="*60)
    print("稀疏性可视化工具")
    print("="*60)
    
    # 加载模型
    model, config = load_model(args.model)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成所有可视化
    visualize_weight_matrices(model, output_dir)
    visualize_layer_sparsity(model, output_dir)
    visualize_weight_distribution(model, output_dir)
    visualize_connection_pattern(model, output_dir)
    create_summary_report(model, config, output_dir)
    
    print("\n" + "="*60)
    print(f"✅ 所有可视化已保存到: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
