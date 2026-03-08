import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomModules.config import BaseViTConfig
from CustomModules.basic import FeedForward

class Attention(nn.Module):
    def __init__(self, config: BaseViTConfig, inner_dim: int):
        super().__init__()
        self.inner_dim = inner_dim
        self.keys = nn.Linear(config.d_model, inner_dim, bias=False)
        self.queries = nn.Linear(config.d_model, inner_dim, bias=False)
        self.values = nn.Linear(config.d_model, inner_dim, bias=False)

        self.drop = nn.Dropout(config.dropout_rate)

    def forward(self, input):
        B, T, C = input.shape # C is attention head size

        k = self.keys(input) # (B, T, C)
        q = self.queries(input) # (B, T, C)
        v = self.values(input) # (B, T, C) 

        wei = (q @ k.transpose(dim0=1, dim1=2)) * (C**-0.5) # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        wei = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return wei

class MultiHeadAttention(nn.Module):
    def __init__(self, config: BaseViTConfig):
        super().__init__()
        self.att_heads =nn.ModuleList(
            [Attention(config, config.d_model // config.n_heads) for _ in range(config.n_heads)]) # n_heads x (B, T, C/n_heads)
        self.linear_projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.att_heads], dim=-1) # (B, T, C) cat over channels
        x = self.linear_projection(x)
        
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: BaseViTConfig):
        super().__init__()
        self.lnorm_1 = nn.LayerNorm(config.d_model)
        self.att = MultiHeadAttention(config)
        self.lnorm_2 = nn.LayerNorm(config.d_model)
        self.ffw = FeedForward(config)

    def forward(self, x):
        x = x + self.att(self.lnorm_1(x))
        x = x + self.ffw(self.lnorm_2(x))
        return x
