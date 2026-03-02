"""
Automated script to run dense vs sparse comparison experiments with matched hyperparameters.

## 📝 "相同配置"的含义:
- ✅ 模型架构相同 (num_layers, dim_model, num_heads) → 总参数量相同
- ✅ 任务设置相同 (operation, prime, training_fraction)
- ✅ 训练设置相同 (num_steps, batch_size, learning_rate)
- ❌ 稀疏性设置不同 (final_L0, activation_sparsity等，这是实验变量)

## 🎯 对比目标:
在相同模型容量下，稀疏化是否能保持或提升性能

Usage:
    python run_dense_vs_sparse.py --sequential
"""

import sys
import argparse
import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# 📝 共有配置区 - 在这里修改实验参数
# ============================================================

SHARED_CONFIG = {
    #=== 数据配置 ===
    "operation": "x+y",
    "prime": 97,
    "training_fraction": 0.5,
    
    # === 模型架构 (必须相同！决定总参数量) ===
    "num_layers": 2,       # 层数
    "dim_model": 128,      # 维度
    "num_heads": 4,        # 注意力头数
    # → 总参数量 ~422K
    
    # === Normalization选择 ===
    # 调整：改用LayerNorm以加快小模型收敛
    "norm_type": "layernorm",  # layernorm更适合小规模任务
    
    # === 训练配置 ===
    "batch_size": 512,
    "num_steps": 4000,   # 充足的训练步数
    "device": "cpu",
    
    # === 优化器配置 (AdamW) ===
    # 调整：使用更小的eps以加快收敛
    "learning_rate": 1e-3,     # 基础学习率
    "adam_beta1": 0.9,         # Adam β1
    "adam_beta2": 0.98,        # Adam β2（原始设置）
    "adam_eps": 1e-8,          # Adam ε（PyTorch默认值，与原始一致）
    
    # === 学习率调度 ===
    # 保留warmup和cosine decay（有助于稳定训练）
    "warmup_ratio": 0.01,      # Warmup阶段比例（1%）
    "use_cosine_decay": True,  # 使用cosine衰减
    "min_lr_ratio": 0.0,       # 最小学习率比例
    
    # === 梯度裁剪 ===
    "grad_clip_rms": 1.0,      # RMS梯度裁剪阈值
    
    # === 验证配置 ===
    "eval_every": 5,           # 每5个epoch验证一次
}

# ============================================================
# 🧪 实验2D：对齐测试 - L2 vs L2-TopK(L0=1.0)
# 目标：验证L0=1.0的L2-TopK完全等价于纯L2
# ============================================================

# Dense配置 - 纯L2 baseline
DENSE_SPECIFIC = {
    # === 正则化方法 ===
    "regularization_type": "l2",  # 纯L2正则化（通过weight_decay）
    "weight_decay": 0.30,          # L2正则化强度
    
    # === TopK权重稀疏配置（Dense不使用）===
    "final_L0": 1.0,              # 100%权重非零（不稀疏）
    "initial_L0": 1.0,            # 起始L0值
    "anneal_end_ratio": 0.0,      # L0退火结束位置（0=不退火）
    "use_L0_lr_scaling": False,   # 学习率缩放：lr×1/√L0（L0=1.0时无影响）
    "min_connections": 1,         # 每个神经元最小连接数
}

# Sparse配置 - 测试完整架构（但不稀疏）
SPARSE_SPECIFIC = {
    # === 正则化方法 ===
    "regularization_type": "l2-topk",  # L2 + Top-K权重稀疏
    "weight_decay": 0.4,               # L2部分（与Dense一致）
    
    # === TopK权重稀疏配置 ===
    "final_L0": 0.40,              # 目标L0：100%权重非零（对齐测试）
    "initial_L0": 1.0,            # 起始L0：100%非零
    "anneal_end_ratio": 0.30,      # L0退火：前X%步数从initial_L0降到final_L0（0=不退火）
    "use_L0_lr_scaling": True,   # 学习率动态缩放：lr × 1/√L0
                                  # - False: lr保持不变
                                  # - True:  L0=1.0时lr×1, L0=0.1时lr×3.16, L0=0.01时lr×10
                                  # - 当前L0=1.0，开启也无影响，保持False更清晰
    
    # === AbsTopK激活稀疏配置 ===
    "use_activation_sparsity": True,   # 启用AbsTopK层（测试层本身）
    "activation_sparsity_ratio": 1.0,  # 保留比例：1.0=保留100%（实际不稀疏）
                                       # - 训练和推理都生效
                                       # - 后续实验改为0.25=只保留top 25%激活
    "min_connections": 1,              # 每个神经元最小非零权重数
}

# ============================================================
# 📝 后续实验：逐步启用稀疏化
# ============================================================
# 
# 实验3A：轻度稀疏（权重+激活都稀疏）
# SPARSE_SPECIFIC["final_L0"] = 0.5              # 50%权重非零
# SPARSE_SPECIFIC["anneal_end_ratio"] = 0.5       # 前50%步数退火
# SPARSE_SPECIFIC["use_L0_lr_scaling"] = False   # 轻度稀疏不需要lr缩放
# SPARSE_SPECIFIC["activation_sparsity_ratio"] = 0.25  # top 25%激活
#
# 实验3B：重度稀疏（论文推荐配置）
# SPARSE_SPECIFIC["final_L0"] = 0.01             # 1%权重非零（论文目标）
# SPARSE_SPECIFIC["anneal_end_ratio"] = 0.5       # 前50%步数退火
# SPARSE_SPECIFIC["use_L0_lr_scaling"] = True    # 启用lr缩放（lr×10补偿）
# SPARSE_SPECIFIC["activation_sparsity_ratio"] = 0.25  # top 25%激活
# SPARSE_SPECIFIC["min_connections"] = 4          # 最小4个连接（防止死神经元）



# Wandb配置
WANDB_CONFIG = {
    "project": "sparse_vs_dense",
    "mode": "online",
    # API key从环境变量读取,更安全
    "api_key": os.environ.get("WANDB_API_KEY", "2695076dd0f87f2dba081c03fdc9cfa84acef643")
}

# ============================================================


def run_dense_training(run_id: str):
    """Run dense baseline with wandb logging.
    
    Args:
        run_id: Unique identifier for this experiment batch
    """
    print("\n" + "="*60)
    print("🔷 DENSE BASELINE")
    print("="*60)
    print("Starting dense training...", flush=True)
    
    from training import main as dense_main
    from argparse import Namespace
    
    args = Namespace(**SHARED_CONFIG, **DENSE_SPECIFIC)
    print(f"Config: {args.num_layers}L×{args.dim_model}D×{args.num_heads}H", flush=True)
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}", flush=True)
    print(f"Run ID: {run_id}", flush=True)
    print("Training in progress (this may take several minutes)...", flush=True)
    
    import training
    orig = training.wandb.init
    def custom_init(*a, **k):
        # 添加时间戳和run_id标识,避免重名覆盖
        k.update({
            'project': WANDB_CONFIG['project'], 
            'name': f'dense-baseline-{run_id}',  # 唯一名称
            'tags': ['dense', run_id],  # 通过tag关联同批次实验
            'mode': WANDB_CONFIG['mode']
        })
        training.wandb.login(key=WANDB_CONFIG['api_key'])
        return orig(*a, **k)
    training.wandb.init = custom_init
    
    try:
        print("\n[DENSE] Training started...\n", flush=True)
        dense_main(args)
        print("\n[DENSE] ✅ Training completed!", flush=True)
        # Finish wandb run to separate from sparse run
        training.wandb.finish()
        return True
    except Exception as e:
        print(f"\n[DENSE] ❌ Failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        training.wandb.finish()  # Clean up even on error
        return False
    finally:
        training.wandb.init = orig


def run_sparse_training(run_id: str):
    """Run sparse training with wandb logging.
    
    Args:
        run_id: Unique identifier for this experiment batch
    """
    print("\n" + "="*60)
    print("[SPARSE] SPARSE TRAINING")
    print("="*60)
    print("Starting sparse training...", flush=True)
    
    from training_sparse import main as sparse_main
    from config_sparse import SparseTrainingConfig
    import training_sparse
    
    config = SparseTrainingConfig(
        **SHARED_CONFIG, **SPARSE_SPECIFIC,
        use_wandb=True,
        wandb_mode=WANDB_CONFIG['mode'],
        wandb_project=WANDB_CONFIG['project'],
    )
    print(f"Config: {config.num_layers}L×{config.dim_model}D×{config.num_heads}H", flush=True)
    print(f"L0={config.final_L0}, Regularization: {config.regularization_type}", flush=True)
    print(f"Run ID: {run_id}", flush=True)
    print("Training in progress (this may take several minutes)...", flush=True)
    
    # 统一wandb初始化逻辑 - 添加login和name/tags
    orig_init = training_sparse.wandb.init
    def custom_init(*a, **k):
        # 添加唯一run name和tags
        k.update({
            'name': f'sparse-{run_id}',  # 唯一名称
            'tags': k.get('tags', []) + ['sparse', run_id]  # 通过tag关联
        })
        training_sparse.wandb.login(key=WANDB_CONFIG['api_key'])
        return orig_init(*a, **k)
    training_sparse.wandb.init = custom_init
    
    try:
        print("\n[SPARSE] Training started...\n", flush=True)
        sparse_main(config)
        print("\n[SPARSE] ✅ Training completed!", flush=True)
        # 确保正确关闭wandb run
        training_sparse.wandb.finish()
        return True
    except Exception as e:
        print(f"\n[SPARSE] ❌ Failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        training_sparse.wandb.finish()  # Clean up even on error
        return False
    finally:
        training_sparse.wandb.init = orig_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['both', 'dense', 'sparse'], default='both')
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()
    
    # 生成唯一的run_id来标识同一批次的对比实验
    run_id = time.strftime("%Y%m%d_%H%M%S")
    
    print("="*60, flush=True)
    print("[COMPARISON] DENSE VS SPARSE", flush=True)
    print("="*60, flush=True)
    print(f"\n  Run ID: {run_id}", flush=True)
    print(f"  Mode: {args.mode}", flush=True)
    print(f"  Project: {WANDB_CONFIG['project']}", flush=True)
    print(f"  URL: https://wandb.ai/zengh17/{WANDB_CONFIG['project']}", flush=True)
    print(f"\n  [Shared] {SHARED_CONFIG['operation']} mod {SHARED_CONFIG['prime']}, "
          f"{SHARED_CONFIG['num_layers']}L×{SHARED_CONFIG['dim_model']}D, "
          f"{SHARED_CONFIG['num_steps']} steps", flush=True)
    print(f"  [Dense]  WD={DENSE_SPECIFIC['weight_decay']}, L0={DENSE_SPECIFIC.get('final_L0', 1.0)}", flush=True)
    print(f"  [Sparse] WD={SPARSE_SPECIFIC['weight_decay']}, L0={SPARSE_SPECIFIC['final_L0']}", flush=True)
    print(flush=True)
    
    results = {}
    
    if args.mode in ['both', 'dense']:
        if not args.sequential and input("\n[DENSE] Run dense? (y/n): ").lower() != 'y':
            results['dense'] = 'skipped'
        else:
            print("\n[1/2] Running dense baseline experiment...", flush=True)
            results['dense'] = run_dense_training(run_id)
    
    if args.mode in ['both', 'sparse']:
        if not args.sequential and input("\n[SPARSE] Run sparse? (y/n): ").lower() != 'y':
            results['sparse'] = 'skipped'
        else:
            print("\n[2/2] Running sparse training experiment...", flush=True)
            results['sparse'] = run_sparse_training(run_id)
    
    print("\n" + "="*60, flush=True)
    print("[SUMMARY] Results", flush=True)
    print("="*60, flush=True)
    for name, result in results.items():
        status = "[OK]" if result is True else "[FAIL]" if result is False else "[SKIP]"
        print(f"{name:8s}: {status}", flush=True)
    print(f"\n[WEB] View at: https://wandb.ai/zengh17/{WANDB_CONFIG['project']}", flush=True)


if __name__ == "__main__":
    main()
