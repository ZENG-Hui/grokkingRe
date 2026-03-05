"""
Utility functions for sparse training following the circuit sparsity paper.

This module implements the core techniques for enforcing weight sparsity during training:
- Top-K weight selection
- L0 annealing schedule
- Sparse learning rate adjustment
- RMS gradient clipping
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional


def enforce_weight_sparsity(
    model: nn.Module,
    target_L0: float,
    min_connections: int = 4,
    excluded_params: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Enforce weight sparsity by keeping only top-k weights by absolute value.
    
    This is the core mechanism from the paper: after each optimizer step,
    zero out all but the largest magnitude entries in each weight matrix.
    
    Args:
        model: The model to enforce sparsity on
        target_L0: Target fraction of nonzero weights (e.g., 0.001 for 1/1000)
        min_connections: Minimum number of nonzero values per neuron/channel
        excluded_params: List of parameter name patterns to exclude from sparsification
        
    Returns:
        Dictionary with sparsity statistics
    """
    stats = {
        'total_params': 0,
        'nonzero_params': 0,
        'num_matrices': 0
    }
    
    excluded_params = excluded_params or []
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip if parameter should be excluded
            if any(exclude in name for exclude in excluded_params):
                continue
                
            stats['total_params'] += param.numel()
            stats['num_matrices'] += 1
            
            # Calculate number of nonzero elements to keep
            num_nonzero = max(
                int(param.numel() * target_L0),
                min_connections if param.dim() >= 2 else 1  # At least min_connections for weight matrices
            )
            
            # Get the threshold value (kth largest absolute value)
            flat_weights = param.abs().flatten()
            if flat_weights.numel() <= num_nonzero:
                # If we want to keep all or more weights than exist, keep all
                continue
                
            # Find threshold using kthvalue
            k = flat_weights.numel() - num_nonzero
            if k > 0:
                threshold = torch.kthvalue(flat_weights, k + 1).values
                
                # Create binary mask
                mask = param.abs() >= threshold
                
                # Apply mask to zero out small weights
                param.mul_(mask.float())
                
                stats['nonzero_params'] += mask.sum().item()
            else:
                stats['nonzero_params'] += param.numel()
    
    return stats


def get_target_L0(
    current_step: int,
    total_steps: int,
    initial_L0: float = 1.0,
    final_L0: float = 0.001,
    anneal_end_ratio: float = 0.5
) -> float:
    """
    Calculate target L0 (fraction of nonzero weights) with linear annealing.
    
    From the paper: "We anneal the L0 linearly over the first 50% of training"
    
    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        initial_L0: Initial L0 (1.0 = fully dense)
        final_L0: Final target L0 (e.g., 0.001)
        anneal_end_ratio: Fraction of training when annealing ends (default 0.5)
        
    Returns:
        Current target L0
    """
    anneal_end_step = int(total_steps * anneal_end_ratio)
    
    if current_step >= anneal_end_step:
        return final_L0
    
    # Linear interpolation
    progress = current_step / anneal_end_step
    current_L0 = initial_L0 + (final_L0 - initial_L0) * progress
    
    return current_L0


def get_sparse_lr(
    base_lr: float,
    current_step: int,
    total_steps: int,
    current_L0: float,
    warmup_ratio: float = 0.01,
    use_cosine_decay: bool = True
) -> float:
    """
    Calculate learning rate with warmup, decay, and L0-dependent scaling.
    
    From the paper: "Our lr schedule is defined by the product of a normal 
    warmup-decay schedule, and a factor of 1/√L0"
    
    Args:
        base_lr: Base learning rate
        current_step: Current training step
        total_steps: Total training steps
        current_L0: Current L0 norm
        warmup_ratio: Fraction of training for warmup (default 0.01 = 1%)
        use_cosine_decay: Whether to use cosine decay after warmup
        
    Returns:
        Adjusted learning rate
    """
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Warmup phase
    if current_step < warmup_steps:
        warmup_factor = current_step / warmup_steps
    else:
        # Decay phase
        if use_cosine_decay:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            warmup_factor = 0.5 * (1 + math.cos(math.pi * progress))
        else:
            warmup_factor = 1.0
    
    # L0-dependent scaling: lr ∝ 1/√L0
    # Avoid division by zero and numerical issues
    L0_factor = 1.0 / math.sqrt(max(current_L0, 1e-8))
    
    return base_lr * warmup_factor * L0_factor


def clip_grad_rms(
    parameters,
    max_rms: float = 1.0
) -> float:
    """
    Clip gradients by root mean square to the specified maximum.
    
    From the paper: "We clip the root-mean-square of the gradient to 1"
    
    Args:
        parameters: Model parameters (from model.parameters())
        max_rms: Maximum RMS value (default 1.0)
        
    Returns:
        The gradient RMS before clipping
    """
    # Convert to list to avoid multiple iterations
    params_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(params_with_grad) == 0:
        return 0.0
    
    # Calculate total squared norm and total number of elements
    total_norm_sq = sum((p.grad ** 2).sum().item() for p in params_with_grad)
    total_numel = sum(p.grad.numel() for p in params_with_grad)
    
    # Calculate RMS
    rms = math.sqrt(total_norm_sq / total_numel)
    
    # Clip if necessary
    if rms > max_rms:
        scale = max_rms / (rms + 1e-8)  # Add epsilon for numerical stability
        for p in params_with_grad:
            p.grad.mul_(scale)
    
    return rms


def count_nonzero_params(model: nn.Module) -> Dict[str, int]:
    """
    Count nonzero parameters in the model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with total and nonzero parameter counts
    """
    total = 0
    nonzero = 0
    
    for param in model.parameters():
        total += param.numel()
        nonzero += (param != 0).sum().item()
    
    return {
        'total': total,
        'nonzero': nonzero,
        'sparsity_ratio': nonzero / total if total > 0 else 0.0,
        'zeros': total - nonzero
    }


def get_param_stats_by_layer(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """
    Get detailed sparsity statistics for each layer.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary mapping layer names to their sparsity stats
    """
    stats = {}
    
    for name, param in model.named_parameters():
        total = param.numel()
        nonzero = (param != 0).sum().item()
        
        stats[name] = {
            'total': total,
            'nonzero': nonzero,
            'sparsity_ratio': nonzero / total if total > 0 else 0.0,
            'shape': tuple(param.shape)
        }
    
    return stats


def apply_min_connections_constraint(
    param: torch.Tensor,
    min_connections: int = 4
) -> torch.Tensor:
    """
    Ensure each neuron/channel has at least min_connections nonzero weights.
    
    This helps prevent dead neurons during sparse training.
    
    Args:
        param: Weight tensor (assumed to be 2D or higher)
        min_connections: Minimum number of nonzero connections per neuron
        
    Returns:
        Mask tensor indicating which weights should be kept
    """
    if param.dim() < 2:
        # For 1D parameters (biases), just keep everything nonzero
        return (param != 0).float()
    
    # For weight matrices, ensure minimum connections per row/column
    # This is a simplified implementation - the paper's actual approach may be more sophisticated
    
    # Count nonzero per neuron (first dimension)
    nonzero_per_neuron = (param != 0).sum(dim=1)
    
    # Create mask
    mask = param.abs() > 0
    
    # For neurons with too few connections, keep the top-k by absolute value
    for i in range(param.shape[0]):
        if nonzero_per_neuron[i] < min_connections:
            # Get top-k values for this neuron
            _, indices = param[i].abs().topk(min(min_connections, param.shape[1]))
            # Update mask
            mask[i, :] = False
            mask[i, indices] = True
    
    return mask.float()
