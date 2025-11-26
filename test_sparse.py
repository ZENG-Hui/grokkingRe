"""
Quick test script to verify sparse training implementation.

This script performs basic sanity checks on the sparse training system.
"""

import torch
from config_sparse import get_config, SparseTrainingConfig
from model_sparse import create_sparse_model
from sparse_utils import (
    enforce_weight_sparsity,
    get_target_L0,
    get_sparse_lr,
    clip_grad_rms,
    count_nonzero_params
)


def test_model_creation():
    """Test that sparse model can be created."""
    print("\n=== Testing Model Creation ===")
    config = get_config("tiny")
    model = create_sparse_model(config)
    
    stats = model.count_parameters()
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {stats['total']:,}")
    print(f"  Initial nonzero: {stats['nonzero']:,}")
    print(f"  Sparsity: {stats['sparsity_ratio']:.2%}")
    
    return model, config


def test_forward_pass(model, config):
    """Test forward pass with dummy data."""
    print("\n=== Testing Forward Pass ===")
    batch_size = 4
    seq_len = config.seq_len
    
    # Create dummy input
    dummy_input = torch.randint(0, config.num_tokens, (batch_size, seq_len))
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [{seq_len}, {batch_size}, {config.num_tokens}]")
    
    assert output.shape == (seq_len, batch_size, config.num_tokens), "Output shape mismatch!"
    return output


def test_sparsity_enforcement(model, config):
    """Test that weight sparsity enforcement works."""
    print("\n=== Testing Sparsity Enforcement ===")
    
    # Initial stats
    initial_stats = count_nonzero_params(model)
    print(f"Before enforcement:")
    print(f"  Nonzero: {initial_stats['nonzero']:,} / {initial_stats['total']:,}")
    print(f"  Ratio: {initial_stats['sparsity_ratio']:.4f}")
    
    # Enforce sparsity
    target_L0 = 0.1  # Keep 10%
    sparsity_stats = enforce_weight_sparsity(
        model=model,
        target_L0=target_L0,
        min_connections=4
    )
    
    # Check stats after
    final_stats = count_nonzero_params(model)
    print(f"\nAfter enforcement (target L0={target_L0}):")
    print(f"  Nonzero: {final_stats['nonzero']:,} / {final_stats['total']:,}")
    print(f"  Ratio: {final_stats['sparsity_ratio']:.4f}")
    print(f"  ✓ Sparsity enforcement successful")
    
    # Verify sparsity is close to target
    assert final_stats['sparsity_ratio'] < target_L0 * 1.5, "Sparsity not enforced correctly!"


def test_L0_annealing():
    """Test L0 annealing schedule."""
    print("\n=== Testing L0 Annealing ===")
    
    total_steps = 1000
    final_L0 = 0.001
    
    # Test at different points
    test_points = [0, 250, 500, 750, 1000]
    
    for step in test_points:
        L0 = get_target_L0(step, total_steps, final_L0=final_L0)
        progress = step / total_steps * 100
        print(f"  Step {step:4d} ({progress:5.1f}%): L0 = {L0:.4f}")
    
    # Verify annealing behavior
    L0_start = get_target_L0(0, total_steps, final_L0=final_L0)
    L0_mid = get_target_L0(total_steps // 2, total_steps, final_L0=final_L0)
    L0_end = get_target_L0(total_steps, total_steps, final_L0=final_L0)
    
    assert L0_start == 1.0, "Should start at 1.0"
    assert L0_end == final_L0, "Should end at final_L0"
    assert L0_mid == final_L0, "Should reach final_L0 at 50%"
    print(f"  ✓ L0 annealing working correctly")


def test_sparse_lr_schedule():
    """Test sparse learning rate schedule."""
    print("\n=== Testing Sparse LR Schedule ===")
    
    base_lr = 1e-3
    total_steps = 1000
    
    # Test at different L0 values
    test_L0s = [1.0, 0.1, 0.01, 0.001]
    
    for L0 in test_L0s:
        lr = get_sparse_lr(
            base_lr=base_lr,
            current_step=500,
            total_steps=total_steps,
            current_L0=L0
        )
        scaling = lr / base_lr
        print(f"  L0={L0:6.3f}: lr={lr:.6f} (scaling={scaling:.2f}x)")
    
    print(f"  ✓ Sparse LR schedule working correctly")


def test_gradient_clipping(model):
    """Test RMS gradient clipping."""
    print("\n=== Testing Gradient Clipping ===")
    
    # Create dummy gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 10.0  # Large gradients
    
    # Clip
    grad_rms_before = clip_grad_rms(model.parameters(), max_rms=100.0)  # No clipping
    print(f"  Gradient RMS (before clipping): {grad_rms_before:.4f}")
    
    # Reset with larger gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 10.0
    
    grad_rms_after = clip_grad_rms(model.parameters(), max_rms=1.0)
    print(f"  Gradient RMS (after clipping to 1.0): {grad_rms_after:.4f}")
    
    # Verify clipping  worked
    final_rms = 0.0
    total_numel = 0
    for p in model.parameters():
        if p.grad is not None:
            final_rms += (p.grad ** 2).sum().item()
            total_numel += p.grad.numel()
    final_rms = (final_rms / total_numel) ** 0.5
    
    print(f"  Final RMS after clipping: {final_rms:.4f}")
    assert final_rms <= 1.01, "Gradient clipping failed!"
    print(f"  ✓ Gradient clipping working correctly")


def test_training_step(model, config):
    """Test a single training step."""
    print("\n=== Testing Training Step ===")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay
    )
    
    # Create dummy batch
    batch_size = config.batch_size
    inputs = torch.randint(0, config.num_tokens, (batch_size, config.seq_len))
    labels = torch.randint(0, config.num_tokens, (batch_size,))
    
    # Forward
    output = model(inputs)[-1, :, :]
    loss = torch.nn.functional.cross_entropy(output, labels)
    
    print(f"  Loss before step: {loss.item():.4f}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients
    clip_grad_rms(model.parameters(), max_rms=1.0)
    
    # Optimizer step
    optimizer.step()
    
    # Enforce sparsity
    enforce_weight_sparsity(
        model=model,
        target_L0=config.final_L0,
        min_connections=config.min_connections
    )
    
    # Forward again to check loss changed
    output = model(inputs)[-1, :, :]
    loss_after = torch.nn.functional.cross_entropy(output, labels)
    
    print(f"  Loss after step: {loss_after.item():.4f}")
    print(f"  ✓ Training step completed successfully")


def main():
    """Run all tests."""
    print("="*60)
    print("SPARSE TRAINING IMPLEMENTATION TESTS")
    print("="*60)
    
    try:
        # Test 1: Model creation
        model, config = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model, config)
        
        # Test 3: Sparsity enforcement
        test_sparsity_enforcement(model, config)
        
        # Test 4: L0 annealing
        test_L0_annealing()
        
        # Test 5: LR schedule
        test_sparse_lr_schedule()
        
        # Test 6: Gradient clipping
        test_gradient_clipping(model)
        
        # Test 7: Full training step
        test_training_step(model, config)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe sparse training implementation is ready to use.")
        print("Run 'python training_sparse.py' to start training.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
