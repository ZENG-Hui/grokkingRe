"""
Sparse Transformer model following the circuit sparsity paper.

This module implements a weight-sparse Transformer with:
- RMSNorm instead of LayerNorm
- AbsTopK activation sparsity
- Support for tracking weight sparsity
"""

from einops import rearrange, repeat
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    From the paper: "To ensure that zero values have a privileged meaning 
    in the residual stream, we use RMSNorm instead of LayerNorm."
    
    RMSNorm doesn't subtract the mean, only divides by RMS.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class AbsTopK(nn.Module):
    """
    Activation sparsity: keep only top-k activations by absolute value.
    
    From the paper: "We also enforce mild activation sparsity at all node 
    locations, with 1 in 4 nonzero activations."
    """
    def __init__(self, k_ratio: float = 0.25):
        """
        Args:
            k_ratio: Fraction of activations to keep (default 0.25 = keep 25%)
        """
        super().__init__()
        self.k_ratio = k_ratio
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training and self.k_ratio >= 1.0:
            # During evaluation, optionally skip sparsification
            return x
        
        # Calculate k for this specific input
        dim = x.shape[-1]
        k = max(1, int(dim * self.k_ratio))
        
        if k >= dim:
            return x
        
        # Get top-k indices by absolute value
        _, indices = torch.topk(x.abs(), k, dim=-1)
        
        # Create mask
        mask = torch.zeros_like(x)
        mask.scatter_(-1, indices, 1.0)
        
        return x * mask


class SparseDecoderBlock(nn.Module):
    """
    Sparse version of decoder block with RMSNorm and optional AbsTopK.
    """
    def __init__(
        self, 
        dim_model: int, 
        n_heads: int,
        activation_sparsity: float = 0.25,
        use_activation_sparsity: bool = True
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, batch_first=False)
        self.self_attn_norm = RMSNorm(dim_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Linear(dim_model * 4, dim_model)
        )
        self.ffn_norm = RMSNorm(dim_model)
        
        # Activation sparsity
        self.use_activation_sparsity = use_activation_sparsity
        if use_activation_sparsity:
            self.attn_topk = AbsTopK(activation_sparsity)
            self.ffn_topk = AbsTopK(activation_sparsity)
    
    def forward(self, x: Tensor) -> Tensor:
        # Create causal attention mask
        seq_len = x.size(0)
        attn_mask = torch.full(
            (seq_len, seq_len), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        
        # Apply activation sparsity after attention
        if self.use_activation_sparsity:
            attn_out = self.attn_topk(attn_out)
        
        x = self.self_attn_norm(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        
        # Apply activation sparsity after FFN
        if self.use_activation_sparsity:
            ffn_out = self.ffn_topk(ffn_out)
        
        x = self.ffn_norm(x + ffn_out)
        
        return x


class SparseTransformer(nn.Module):
    """
    Sparse Transformer for modular arithmetic task.
    
    Key differences from standard Transformer:
    - Uses RMSNorm instead of LayerNorm
    - Optional AbsTopK activation sparsity
    - Designed to work with weight sparsity enforcement during training
    """
    def __init__(
        self, 
        num_layers: int, 
        dim_model: int, 
        num_heads: int, 
        num_tokens: int, 
        seq_len: int,
        activation_sparsity: float = 0.25,
        use_activation_sparsity: bool = True,
        use_positional_encoding: bool = True
    ):
        super().__init__()
        
        self.use_positional_encoding = use_positional_encoding
        
        # Embeddings
        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        if use_positional_encoding:
            self.position_embeddings = nn.Embedding(seq_len, dim_model)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            SparseDecoderBlock(
                dim_model, 
                num_heads,
                activation_sparsity,
                use_activation_sparsity
            ) 
            for _ in range(num_layers)
        ])
        
        # Output
        self.output_norm = RMSNorm(dim_model)
        self.output_proj = nn.Linear(dim_model, num_tokens)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: Token indices [batch_size, seq_len]
            
        Returns:
            Logits [seq_len, batch_size, num_tokens]
        """
        batch_size, context_len = inputs.shape
        
        # Token embeddings
        token_embedding = self.token_embeddings(inputs)
        
        # Position embeddings (if used)
        if self.use_positional_encoding:
            positions = repeat(
                torch.arange(context_len, device=inputs.device), 
                "p -> b p", 
                b=batch_size
            )
            position_embedding = self.position_embeddings(positions)
            embedding = token_embedding + position_embedding
        else:
            embedding = token_embedding
        
        # Rearrange to [seq_len, batch_size, dim_model] for attention
        x = rearrange(embedding, 'b s d -> s b d')
        
        # Apply decoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    def count_parameters(self) -> dict:
        """Count total and nonzero parameters."""
        total = sum(p.numel() for p in self.parameters())
        nonzero = sum((p != 0).sum().item() for p in self.parameters())
        
        return {
            'total': total,
            'nonzero': nonzero,
            'sparsity_ratio': nonzero / total if total > 0 else 0.0,
            'compression': total / nonzero if nonzero > 0 else float('inf')
        }


def create_sparse_model(config) -> SparseTransformer:
    """
    Create a sparse transformer from configuration.
    
    Args:
        config: SparseTrainingConfig instance
        
    Returns:
        SparseTransformer model
    """
    model = SparseTransformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=config.num_tokens,
        seq_len=config.seq_len,
        activation_sparsity=config.activation_sparsity_ratio,
        use_activation_sparsity=config.use_activation_sparsity,
        use_positional_encoding=True  # Can be made configurable
    )
    
    return model
