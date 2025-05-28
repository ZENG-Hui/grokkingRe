from einops import rearrange, repeat # 提供张量形状操作的高级函数
import torch
from torch import nn, Tensor

class DecoderBlock(torch.nn.Module):
  def __init__(self, dim_model: int, n_heads: int):
    super().__init__()

    self.self_attn = nn.MultiheadAttention(dim_model, n_heads) # 多头自注意力，允许模型关注序列中的不同位置
    self.self_attn_norm = nn.LayerNorm(dim_model) # 层归一化，标准化每个样本的特征
    self.ffn = nn.Sequential( # 前馈神经网络，包含两个线性层和一个激活函数
        nn.Linear(dim_model, dim_model * 4),
        nn.GELU(),
        nn.Linear(dim_model * 4, dim_model)
    )
    self.ffn_norm = nn.LayerNorm(dim_model)

  def forward(self, x: Tensor):
    attn_mask = torch.full( # 创建上三角矩阵（对角线上方为1，其余为0）
        (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
    )
    attn_mask = torch.triu(attn_mask, diagonal=1)
    
    # 应用自注意力，然后进行残差连接和层归一化
    a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
    a1 = self.self_attn_norm (x + a1)
    # 应用前馈神经网络，然后进行残差连接和层归一化
    a2 = self.ffn(a1)
    a2 = self.ffn_norm(a1 + a2)

    return a2

class Transformer(torch.nn.Module):
  def __init__(self, 
               num_layers: int, # 编码器层数
               dim_model: int, # 模型维度
               num_heads: int, # 注意力头数
               num_tokens: int, # 词汇表大小
               seq_len: int # 序列长度（上下文长度）
              ):
    super().__init__()

    # 输入标记的嵌入和位置嵌入
    self.token_embeddings = nn.Embedding(num_tokens, dim_model)
    self.position_embeddings = nn.Embedding(seq_len, dim_model)
    
    # 创建num_layers个DecoderBlock
    # 最后添加层归一化和线性投影层
    # 线性层将隐藏表示映射回词汇表大小，用于最终预测
    self.model = nn.Sequential(
        *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
        nn.LayerNorm(dim_model),
        nn.Linear(dim_model, num_tokens)
    )

  def forward(self, inputs: Tensor):
    batch_size, context_len = inputs.shape
    
    token_embedding = self.token_embeddings(inputs)

    # 创建位置索引(0,1,2,3)，并重复batch_size次
    positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b = batch_size)
    position_embedding = self.position_embeddings(positions)

    embedding = token_embedding + position_embedding

    #重排张量维度，从[batch, seq, dim]变为[seq, batch, dim]，适配PyTorch注意力层的输入格式
    embedding = rearrange(embedding, 'b s d -> s b d')

    return self.model(embedding)
