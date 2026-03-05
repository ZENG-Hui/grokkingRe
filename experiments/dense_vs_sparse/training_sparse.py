"""
Sparse training loop following the circuit sparsity paper.

This module implements the complete training procedure with:
- Weight sparsity enforcement via Top-K selection
- L0 annealing schedule
- Sparse learning rate adjustment
- RMS gradient clipping

============================================================
QUICK START: Modify the configuration below and run:
    python training_sparse.py
============================================================
"""

# ============================================================
# 📝 配置区 - 在这里修改训练设置
# ============================================================

# 选择基础配置: "tiny", "small", "medium", "large"
BASE_CONFIG = "small"

# 自定义覆盖（可选 - 留空则使用BASE_CONFIG的默认值）
CUSTOM_SETTINGS = {
    # 数据配置
    # "operation": "x+y",        # 运算类型: "x+y", "x-y", "x/y"
    # "prime": 97,               # 质数模
    # "training_fraction": 0.5,  # 训练数据比例
    
    # 稀疏性配置
    # "final_L0": 0.01,          # 目标稀疏性 (0.01 = 1%非零)
    # "num_steps": 50000,        # 训练步数
    
    # 学习率配置
    # "learning_rate": 1e-3,     # 基础学习率
    # "weight_decay": 0.1,       # 权重衰减
    
    # 系统配置
    # "device": "cuda",          # "cuda" 或 "cpu"
    # "wandb_mode": "offline",   # "online", "offline", 或 "disabled"
    # "seed": 42,                # 随机种子
}

# ============================================================
# 以下代码无需修改
# ============================================================

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from math import ceil
import os
import torch
from torch import nn
from tqdm import tqdm
import wandb
from typing import Optional

from core.data import get_data
from core.model_sparse import create_sparse_model
from core.config_sparse import SparseTrainingConfig, get_config
from core.sparse_utils import (
    enforce_weight_sparsity,
    get_target_L0,
    get_sparse_lr,
    clip_grad_rms,
    count_nonzero_params,
    get_param_stats_by_layer
)


def main(config: Optional[SparseTrainingConfig] = None):
    """
    Main training function for sparse models.
    
    Args:
        config: SparseTrainingConfig instance (creates default if None)
    """
    # Get configuration
    if config is None:
        config = get_config("small")
    
    # Set random seed
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            config=config.to_dict(),
            mode=config.wandb_mode
        )
        wandb_config = wandb.config
    else:
        wandb_config = config
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define metrics for wandb
    if config.use_wandb:
        wandb.define_metric("step")
        wandb.define_metric("epoch")
        wandb.define_metric("training/*", step_metric='step')
        wandb.define_metric("validation/*", step_metric='epoch')
        wandb.define_metric("sparsity/*", step_metric='step')
    
    # Load data
    print(f"Loading data: {config.operation} mod {config.prime}")
    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
    )
    
    # Create sparse model
    print("Creating sparse model...")
    model = create_sparse_model(config).to(device)
    
    # Print initial model stats
    init_stats = model.count_parameters()
    print(f"Model parameters: {init_stats['total']:,}")
    print(f"Initial nonzero: {init_stats['nonzero']:,} ({init_stats['sparsity_ratio']:.2%})")
    
    # Create optimizer (AdamW with paper's hyperparameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,  # Will be overridden by scheduler
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay
    )
    
    # Calculate total steps
    num_epochs = ceil(config.num_steps / len(train_loader))
    print(f"Training for {config.num_steps} steps (~{num_epochs} epochs)")
    
    # Create checkpoint directory
    if config.save_model:
        os.makedirs(config.save_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        global_step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            config=config,
            device=device,
            global_step=global_step,
            total_steps=config.num_steps
        )
        
        # Evaluate
        evaluate(
            model=model,
            val_loader=val_loader,
            config=config,
            device=device,
            epoch=epoch
        )
        
        # Save checkpoint
        if config.save_model and (epoch + 1) % (config.save_every // len(train_loader)) == 0:
            save_checkpoint(model, optimizer, config, epoch, global_step)
        
        # Stop if reached target steps
        if global_step >= config.num_steps:
            break
    
    # Final evaluation and statistics
    print("\n" + "="*50)
    print("Training complete!")
    final_stats = count_nonzero_params(model)
    print(f"Final nonzero params: {final_stats['nonzero']:,} / {final_stats['total']:,}")
    print(f"Final sparsity: {(1 - final_stats['sparsity_ratio']):.2%}")
    print(f"Compression ratio: {final_stats['total'] / final_stats['nonzero']:.1f}x")
    
    # Save final model
    if config.save_model:
        final_path = os.path.join(config.save_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'stats': final_stats
        }, final_path)
        print(f"Saved final model to {final_path}")
    
    if config.use_wandb:
        wandb.finish()


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    config: SparseTrainingConfig,
    device: torch.device,
    global_step: int,
    total_steps: int
) -> int:
    """
    Train for one epoch with sparsity enforcement.
    
    Returns:
        Updated global_step
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for batch in train_loader:
        if global_step >= total_steps:
            break
        
        # Move data to device
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        
        # Calculate current L0 target
        current_L0 = get_target_L0(
            current_step=global_step,
            total_steps=total_steps,
            initial_L0=config.initial_L0,
            final_L0=config.final_L0,
            anneal_end_ratio=config.anneal_end_ratio
        )
        
        # Calculate current learning rate
        current_lr = get_sparse_lr(
            base_lr=config.learning_rate,
            current_step=global_step,
            total_steps=total_steps,
            current_L0=current_L0,
            warmup_ratio=config.warmup_ratio,
            use_cosine_decay=config.use_cosine_decay
        )
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Forward pass
        optimizer.zero_grad()
        output = model(inputs)[-1, :, :]  # Last position only
        loss = criterion(output, labels)
        acc = (torch.argmax(output, dim=1) == labels).float().mean()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients by RMS
        grad_rms = clip_grad_rms(model.parameters(), max_rms=config.grad_clip_rms)
        
        # Optimizer step
        optimizer.step()
        
        # === CRITICAL: Enforce weight sparsity after optimizer step ===
        sparsity_stats = enforce_weight_sparsity(
            model=model,
            target_L0=current_L0,
            min_connections=config.min_connections,
            excluded_params=config.enforce_sparsity_on
        )
        
        # Logging
        if global_step % config.log_every == 0:
            metrics = {
                "training/loss": loss.item(),
                "training/accuracy": acc.item(),
                "training/learning_rate": current_lr,
                "training/grad_rms": grad_rms,
                "sparsity/target_L0": current_L0,
                "sparsity/actual_sparsity": sparsity_stats['nonzero_params'] / sparsity_stats['total_params'],
                "sparsity/nonzero_params": sparsity_stats['nonzero_params'],
                "step": global_step
            }
            
            if config.use_wandb:
                wandb.log(metrics)
            else:
                if global_step % (config.log_every * 10) == 0:
                    print(f"Step {global_step}: loss={loss.item():.4f}, "
                          f"acc={acc.item():.4f}, L0={current_L0:.4f}")
        
        global_step += 1
    
    return global_step


def evaluate(
    model: nn.Module,
    val_loader,
    config: SparseTrainingConfig,
    device: torch.device,
    epoch: int
):
    """Evaluate model on validation set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            
            # Forward pass
            output = model(inputs)[-1, :, :]
            loss = criterion(output, labels)
            
            # Accumulate metrics
            total_loss += loss.item() * len(labels)
            total_correct += (torch.argmax(output, dim=1) == labels).sum().item()
            total_samples += len(labels)
    
    # Calculate averages
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    metrics = {
        "validation/loss": avg_loss,
        "validation/accuracy": avg_acc,
        "epoch": epoch
    }
    
    if config.use_wandb:
        wandb.log(metrics, commit=False)
    
    print(f"Epoch {epoch}: val_loss={avg_loss:.4f}, val_acc={avg_acc:.4f}")


def save_checkpoint(
    model: nn.Module,
    optimizer,
    config: SparseTrainingConfig,
    epoch: int,
    step: int
):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(
        config.save_dir,
        f"checkpoint_epoch{epoch}_step{step}.pt"
    )
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def apply_custom_settings(config: SparseTrainingConfig, settings: dict) -> SparseTrainingConfig:
    """Apply custom settings to override default configuration."""
    for key, value in settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown setting '{key}' ignored")
    return config


if __name__ == "__main__":
    print("="*60)
    print("SPARSE TRANSFORMER TRAINING")
    print("="*60)
    
    # Load base configuration
    print(f"\n📋 Loading base configuration: '{BASE_CONFIG}'")
    config = get_config(BASE_CONFIG)
    
    # Apply custom settings
    if CUSTOM_SETTINGS:
        print(f"🔧 Applying {len(CUSTOM_SETTINGS)} custom setting(s)")
        config = apply_custom_settings(config, CUSTOM_SETTINGS)
    
    # Display key settings
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Task: {config.operation} mod {config.prime}")
    print(f"Model: {config.num_layers} layers × {config.dim_model} dim × {config.num_heads} heads")
    print(f"Target sparsity: {(1-config.final_L0)*100:.1f}% (L0={config.final_L0})")
    print(f"Training steps: {config.num_steps:,}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(f"Wandb: {config.wandb_mode}")
    
    # Check CUDA
    if config.device == "cuda":
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ CUDA not available, falling back to CPU")
            config.device = "cpu"
    
    print("="*60)
    print()
    
    # Confirm before starting
    input("Press Enter to start training (or Ctrl+C to cancel)...")
    print()
    
    # Start training
    main(config)
