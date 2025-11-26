from einops import rearrange, repeat
import torch
from torch import nn, Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm doesn't subtract the mean, only divides by RMS.
    This ensures zero values have privileged meaning in the residual stream.
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


class DecoderBlock(torch.nn.Module):
    def __init__(self, dim_model: int, n_heads: int, norm_type: str = "layernorm"):
        """
        Args:
            dim_model: Model dimension
            n_heads: Number of attention heads
            norm_type: "layernorm" or "rmsnorm"
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        
        # Select normalization type
        if norm_type == "rmsnorm":
            self.self_attn_norm = RMSNorm(dim_model)
            self.ffn_norm = RMSNorm(dim_model)
        else:  # layernorm
            self.self_attn_norm = nn.LayerNorm(dim_model)
            self.ffn_norm = nn.LayerNorm(dim_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Linear(dim_model * 4, dim_model)
        )

    def forward(self, x: Tensor):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        
        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm(x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_norm(a1 + a2)

        return a2


class Transformer(torch.nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        dim_model: int, 
        num_heads: int, 
        num_tokens: int, 
        seq_len: int,
        norm_type: str = "layernorm"
    ):
        """
        Args:
            num_layers: Number of decoder blocks
            dim_model: Model dimension
            num_heads: Number of attention heads
            num_tokens: Vocabulary size
            seq_len: Sequence length
            norm_type: "layernorm" or "rmsnorm"
        """
        super().__init__()

        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        
        # Create decoder blocks with specified norm
        blocks = [DecoderBlock(dim_model, num_heads, norm_type) for _ in range(num_layers)]
        
        # Final norm and projection
        if norm_type == "rmsnorm":
            final_norm = RMSNorm(dim_model)
        else:
            final_norm = nn.LayerNorm(dim_model)
        
        self.model = nn.Sequential(
            *blocks,
            final_norm,
            nn.Linear(dim_model, num_tokens)
        )

    def forward(self, inputs: Tensor):
        batch_size, context_len = inputs.shape

        token_embedding = self.token_embeddings(inputs)

        positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b=batch_size)
        position_embedding = self.position_embeddings(positions)

        embedding = token_embedding + position_embedding

        embedding = rearrange(embedding, 'b s d -> s b d')

        return self.model(embedding)
