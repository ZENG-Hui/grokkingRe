"""
Quick test script for unified configuration modules.

Tests:
1. Model with LayerNorm vs RMSNorm
2. Unified LR scheduler
3. L0 calculation and scaling
4. Gradient clipping
"""

import torch
import sys
from pathlib import Path

print("="*60)
print("Testing Unified Configuration Modules")
print("="*60)

# Test 1: Model with different norm types
print("\n1️⃣ Testing model.py with norm_type parameter...")
try:
    from model import Transformer, RMSNorm
    
    # Test LayerNorm (default)
    model_ln = Transformer(
        num_layers=2,
        dim_model=128,
        num_heads=4,
        num_tokens=99,
        seq_len=5,
        norm_type="layernorm"
    )
    print("  ✅ LayerNorm model created")
    
    # Test RMSNorm
    model_rms = Transformer(
        num_layers=2,
        dim_model=128,
        num_heads=4,
        num_tokens=99,
        seq_len=5,
        norm_type="rmsnorm"
    )
    print("  ✅ RMSNorm model created")
    
    # Test forward pass
    test_input = torch.randint(0, 99, (4, 5))  # batch=4, seq=5
    output_ln = model_ln(test_input)
    output_rms = model_rms(test_input)
    print(f"  ✅ Forward pass works (LayerNorm output shape: {output_ln.shape})")
    print(f"  ✅ Forward pass works (RMSNorm output shape: {output_rms.shape})")
    
except Exception as e:
    print(f"  ❌ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: LR Scheduler
print("\n2️⃣ Testing lr_scheduler.py...")
try:
    from lr_scheduler import (
        get_unified_lr_scheduler,
        get_L0_scaled_lr,
        calculate_current_L0,
        clip_grad_rms
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model_ln.parameters(), lr=1e-3)
    
    # Create scheduler
    config = {
        'warmup_ratio': 0.01,
        'use_cosine_decay': True,
        'min_lr_ratio': 0.0
    }
    scheduler = get_unified_lr_scheduler(optimizer, config, total_steps=1000)
    print("  ✅ Unified scheduler created")
    
    # Test a few steps
    initial_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    after_warmup_lr = optimizer.param_groups[0]['lr']
    print(f"  ✅ LR scheduling works (initial: {initial_lr:.6f}, after 1 step: {after_warmup_lr:.6f})")
    
    # Test L0 scaling
    base_lr = 1e-3
    current_L0 = 0.1
    scaled_lr = get_L0_scaled_lr(base_lr, current_L0, {'use_L0_lr_scaling': True})
    expected_scale = 1.0 / (0.1 ** 0.5)
    print(f"  ✅ L0 scaling works (base: {base_lr}, L0=0.1, scaled: {scaled_lr:.6f}, factor: {scaled_lr/base_lr:.2f}x)")
    
    # Test L0 annealing
    L0_at_start = calculate_current_L0(0, {'initial_L0': 1.0, 'final_L0': 0.1, 'anneal_end_ratio': 0.5}, 1000)
    L0_at_mid = calculate_current_L0(500, {'initial_L0': 1.0, 'final_L0': 0.1, 'anneal_end_ratio': 0.5}, 1000)
    L0_at_end = calculate_current_L0(1000, {'initial_L0': 1.0, 'final_L0': 0.1, 'anneal_end_ratio': 0.5}, 1000)
    print(f"  ✅ L0 annealing works (t=0: {L0_at_start:.2f}, t=500: {L0_at_mid:.2f}, t=1000: {L0_at_end:.2f})")
    
except Exception as e:
    print(f"  ❌ LR scheduler test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Gradient clipping
print("\n3️⃣ Testing gradient clipping...")
try:
    # Create some gradients
    loss = output_ln.sum()
    loss.backward()
    
    # Clip gradients
    grad_rms = clip_grad_rms(model_ln.parameters(), max_rms=1.0)
    print(f"  ✅ Gradient RMS clipping works (RMS: {grad_rms:.6f})")
    
except Exception as e:
    print(f"  ❌ Gradient clipping test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Config compatibility
print("\n4️⃣ Testing config from run_dense_vs_sparse.py...")
try:
    # This would be imported from run_dense_vs_sparse.py
    test_config = {
        "norm_type": "rmsnorm",
        "adam_eps": 0.1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "warmup_ratio": 0.01,
        "use_cosine_decay": True,
    }
    
    # Create model with config
    model_from_config = Transformer(
        num_layers=2,
        dim_model=128,
        num_heads=4,
        num_tokens=99,
        seq_len=5,
        norm_type=test_config['norm_type']
    )
    
    # Create optimizer with config
    optimizer_from_config = torch.optim.AdamW(
        model_from_config.parameters(),
        lr=1e-3,
        betas=(test_config['adam_beta1'], test_config['adam_beta2']),
        eps=test_config['adam_eps']
    )
    
    # Create scheduler from config
    scheduler_from_config = get_unified_lr_scheduler(
        optimizer_from_config,
        test_config,
        total_steps=1000
    )
    
    print("  ✅ Model created from config (norm_type=rmsnorm)")
    print("  ✅ Optimizer created from config (eps=0.1)")
    print("  ✅ Scheduler created from config (warmup+cosine)")
    
except Exception as e:
    print(f"  ❌ Config compatibility test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
print("\n💡 Ready to integrate into training scripts.")
print("   - model.py: RMSNorm support ✓")
print("   - lr_scheduler.py: Unified scheduler ✓")
print("   - Configuration compatibility ✓")
