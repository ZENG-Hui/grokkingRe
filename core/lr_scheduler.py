"""
Unified Learning Rate Scheduler for Dense and Sparse Training

Implements: warmup → plateau → cosine decay schedule
Compatible with both L0-scaled and non-scaled learning rates
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_unified_lr_scheduler(optimizer, config, total_steps):
    """
    Create unified learning rate scheduler.
    
    Schedule: warmup → plateau → cosine decay
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration with:
            - warmup_ratio: warmup steps = total_steps * warmup_ratio
            - use_cosine_decay: enable cosine annealing
            - min_lr_ratio: minimum lr = base_lr * min_lr_ratio
            - use_L0_lr_scaling: apply 1/√L0 scaling (for sparse training)
        total_steps: Total training steps
    
    Returns:
        LambdaLR scheduler
    """
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.01))
    
    def lr_lambda(current_step):
        # Phase 1: Warmup (linear increase)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Phase 2: Main training
        if not config.get('use_cosine_decay', True):
            # Plateau: constant LR
            return 1.0
        
        # Phase 3: Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_lr_ratio = config.get('min_lr_ratio', 0.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def get_L0_scaled_lr(base_lr, current_L0, config):
    """
    Apply L0 scaling to learning rate: lr * 1/√L0
    
    Args:
        base_lr: Base learning rate from scheduler
        current_L0: Current L0 ratio (e.g., 0.1 for 10% sparsity)
        config: Config with use_L0_lr_scaling flag
    
    Returns:
        Scaled learning rate
    """
    if not config.get('use_L0_lr_scaling', False):
        return base_lr
    
    # Avoid division by zero, minimum L0 = 0.01
    L0_safe = max(current_L0, 0.01)
    scaling_factor = 1.0 / math.sqrt(L0_safe)
    
    return base_lr * scaling_factor


def calculate_current_L0(step, config, total_steps):
    """
    Calculate current L0 target based on annealing schedule.
    
    Args:
        step: Current training step
        config: Config with L0 params:
            - initial_L0: Starting L0 (e.g., 1.0)
            - final_L0: Target L0 (e.g., 0.1)
            - anneal_end_ratio: Anneal over first X% of steps
        total_steps: Total training steps
    
    Returns:
        Current L0 target value
    """
    initial_L0 = config.get('initial_L0', 1.0)
    final_L0 = config.get('final_L0', 0.1)
    anneal_end_ratio = config.get('anneal_end_ratio', 0.5)
    
    anneal_end_step = int(total_steps * anneal_end_ratio)
    
    if step >= anneal_end_step:
        # Annealing finished
        return final_L0
    
    # Linear annealing
    progress = float(step) / float(max(1, anneal_end_step))
    current_L0 = initial_L0 - (initial_L0 - final_L0) * progress
    
    return current_L0


# Gradient clipping utility
def clip_grad_rms(parameters, max_rms):
    """
    Clip gradients based on RMS (Root Mean Square).
    
    Args:
        parameters: Model parameters
        max_rms: Maximum RMS threshold
    
    Returns:
        Actual RMS before clipping
    """
    # Calculate RMS of all gradients
    total_norm_sq = 0.0
    num_params = 0
    
    for p in parameters:
        if p.grad is not None:
            param_norm_sq = p.grad.data.pow(2).sum().item()
            total_norm_sq += param_norm_sq
            num_params += p.grad.numel()
    
    if num_params == 0:
        return 0.0
    
    grad_rms = math.sqrt(total_norm_sq / num_params)
    
    # Clip if necessary
    if grad_rms > max_rms:
        scale = max_rms / (grad_rms + 1e-8)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)
    
    return grad_rms
