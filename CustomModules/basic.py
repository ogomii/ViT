import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomModules.config import BaseViTConfig

class PatchEmbed(nn.Module):
    def __init__(self, 
                config: BaseViTConfig,
                bias=True):
        
        """
        Args:
            img_size: Expected Image Shape (img_size x img_size)
            patch_size: Wanted size for each patch
            in_chans: Number of channels in image (3 for RGB)
            embed_dim: Transformer embedding dimension
        
        """
        super(PatchEmbed, self).__init__()
        assert config.height % config.patch_size == 0
        self.img_size = config.height
        self.patch_size = config.patch_size
        self.in_chans = config.channels
        self.embed_dim = config.d_model
        self.num_patches = config.n_patches**2

        self.proj = nn.Conv2d(in_channels=self.in_chans,
                              out_channels=self.embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size,
                              bias=bias)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x

class FeedForward(nn.Module):
    def __init__(self, config: BaseViTConfig):
        super().__init__()
        self.l1 = nn.Linear(config.d_model, config.d_model)
        self.drop = nn.Dropout(config.dropout_rate)
        self.relu = nn.GELU()
        self.l2 = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.drop(x)
        out = self.l2(x)
        return out


