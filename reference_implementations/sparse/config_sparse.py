"""
Configuration for sparse training experiments.

This module centralizes all hyperparameters for sparse model training,
making it easy to run different experiments and reproduce results.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SparseTrainingConfig:
    """Configuration for sparse training following the circuit sparsity paper."""
    
    # ===== Data Configuration =====
    operation: str = "x+y"  # Modular operation: "x+y", "x-y", or "x/y"
    prime: int = 97  # Modulus (should be prime)
    training_fraction: float = 0.5  # Fraction of data to use for training
    batch_size: int = 512
    
    # ===== Model Architecture =====
    num_layers: int = 2  # Number of decoder blocks
    dim_model: int = 128  # Model dimension
    num_heads: int = 4  # Number of attention heads
    seq_len: int = 5  # Sequence length (fixed for modular arithmetic)
    
    # ===== Sparsity Configuration =====
    # Target L0 (fraction of nonzero weights)
    final_L0: float = 0.01  # Target: 1% nonzero weights (99% sparse)
    initial_L0: float = 1.0  # Start fully dense
    anneal_end_ratio: float = 0.5  # Anneal L0 over first 50% of training
    
    # Activation sparsity (AbsTopK)
    activation_sparsity_ratio: float = 0.25  # Keep top 25% of activations
    use_activation_sparsity: bool = True
    
    # Weight sparsity constraints
    min_connections: int = 4  # Minimum nonzero weights per neuron
    enforce_sparsity_on: list = None  # Which params to sparsify (None = all)
    
    # ===== Training Configuration =====
    num_steps: int = 50000  # Total training steps
    learning_rate: float = 1e-3  # Base learning rate
    weight_decay: float = 0.1  # AdamW weight decay
    
    # AdamW betas (from paper: β1=0.9, β2=0.95)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 0.1  # Much larger than default 1e-8!
    
    # Learning rate schedule
    warmup_ratio: float = 0.01  # 1% warmup
    use_cosine_decay: bool = True
    use_L0_lr_scaling: bool = True  # Scale lr by 1/√L0
    
    # Gradient clipping
    grad_clip_rms: float = 1.0  # Clip gradient RMS to this value
    
    # ===== System Configuration =====
    device: str = "cuda"  # "cuda" or "cpu"
    seed: Optional[int] = 42
    
    # Logging
    log_every: int = 1              # Log training metrics every N steps
    eval_every: int = 10              # Evaluate on validation set every N epochs
    use_wandb: bool = True           # Use Weights & Biases logging
    wandb_project: str = "sparse_grokking"  # WandB project name
    wandb_mode: str = "offline"      # WandB mode: online, offline, or disabled
    
    # Model saving
    save_model: bool = True
    save_dir: str = "./checkpoints"
    save_every: int = 10000  # Save checkpoint every N steps
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Compute total vocab size
        self.num_tokens = self.prime + 2  # numbers + eq_token + op_token
        
        # Compute total training steps if not specified
        if self.enforce_sparsity_on is None:
            # By default, sparsify all parameters except embeddings (optional)
            self.enforce_sparsity_on = []  # Empty list means sparsify all
        
        # Compute number of epochs (approximate)
        samples_per_epoch = int(self.prime ** 2 * self.training_fraction)
        batches_per_epoch = samples_per_epoch // self.batch_size
        self.num_epochs = max(1, self.num_steps // batches_per_epoch)
    
    def to_dict(self):
        """Convert to dictionary for wandb logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# Predefined configurations for different experiment scales

TINY_CONFIG = SparseTrainingConfig(
    # Very small model for quick testing
    num_layers=1,
    dim_model=64,
    num_heads=2,
    final_L0=0.05,  # Less aggressive sparsity
    num_steps=5000,
)

SMALL_CONFIG = SparseTrainingConfig(
    # Small model for initial experiments
    num_layers=2,
    dim_model=128,
    num_heads=4,
    final_L0=0.01,  # 1% nonzero (99% sparse)
    num_steps=50000,
)

MEDIUM_CONFIG = SparseTrainingConfig(
    # Medium model 
    num_layers=4,
    dim_model=256,
    num_heads=8,
    final_L0=0.005,  # 0.5% nonzero (99.5% sparse)
    num_steps=100000,
)

LARGE_CONFIG = SparseTrainingConfig(
    # Larger model (like paper's setup)
    num_layers=8,
    dim_model=512,
    num_heads=8,
    final_L0=0.001,  # 0.1% nonzero (99.9% sparse)
    num_steps=200000,
)


def get_config(name: str = "small") -> SparseTrainingConfig:
    """
    Get a predefined configuration by name.
    
    Args:
        name: Configuration name ("tiny", "small", "medium", "large")
        
    Returns:
        SparseTrainingConfig instance
    """
    configs = {
        "tiny": TINY_CONFIG,
        "small": SMALL_CONFIG,
        "medium": MEDIUM_CONFIG,
        "large": LARGE_CONFIG,
    }
    
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Choose from {list(configs.keys())}")
    
    return configs[name]
