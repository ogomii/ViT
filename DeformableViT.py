import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeformableViT_config:
    def __init__(self, 
                _channels = 3,
                _height = 32, 
                _width = 32,
                _n_patches = 4,
                _window_size = 2,
                _d_model = 1024,
                _n_heads = 16,
                _n_layers = 24,
                _dropout_rate = 0.2):
        self.channels = _channels
        self.height = _height
        self.width = _width
        self.n_patches = _n_patches # number of patches in one dimension, so total patches = n_patches^2
        self.patch_size = int(_height/_n_patches)
        self.window_size = _window_size # size of local attention window
        self.d_model = _d_model
        self.n_heads = _n_heads
        self.n_layers = _n_layers
        self.dropout_rate = _dropout_rate
    
    def to_dict(self):
        return {
            'channels': self.channels,
            'height': self.height,
            'width': self.width,
            'n_patches': self.n_patches,
            'window_size': self.window_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'dropout_rate': self.dropout_rate
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            _channels=config_dict['channels'],
            _height=config_dict['height'],
            _width=config_dict['width'],
            _n_patches=config_dict['n_patches'],
            _window_size=config_dict['window_size'],
            _d_model=config_dict['d_model'],
            _n_heads=config_dict['n_heads'],
            _n_layers=config_dict['n_layers'],
            _dropout_rate=config_dict['dropout_rate']
        )


class PatchEmbed(nn.Module):
    def __init__(self, 
                config: DeformableViT_config,
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
        assert config.width % config.patch_size == 0 # given that stride == patch_size
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


class Attention(nn.Module):
    def __init__(self, config: DeformableViT_config, inner_dim: int):
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
    def __init__(self, config: DeformableViT_config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.att_heads =nn.ModuleList(
            [Attention(config, config.d_model // config.n_heads) for _ in range(config.n_heads)]) # n_heads x (B, T, C/n_heads)
        self.linear_projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.att_heads], dim=-1) # (B, T, C) cat over channels
        x = self.linear_projection(x)
        
        x = self.dropout(x)
        return x
    

class FeedForward(nn.Module):
    def __init__(self, config: DeformableViT_config):
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


class W_MSA(nn.Module): # Window Multi-Head Self Attention
    def __init__(self, config: DeformableViT_config):
        super().__init__()
        self.window_size = config.window_size
        self.n_patches = config.n_patches # number of patches in H or W
        self.lnorm_1 = nn.LayerNorm(config.d_model)
        self.att_local = MultiHeadAttention(config)
        self.lnorm_2 = nn.LayerNorm(config.d_model)
        self.ffw = FeedForward(config)

    def forward(self, x):
        B, T, C = x.shape # Batch, Patches, Embedded dim
        assert T % self.window_size == 0
        x = self.lnorm_1(x) # (B, T, C)
        height = self.n_patches//self.window_size
        width = self.n_patches//self.window_size
        x = x\
            .reshape(B, height, self.window_size, width, self.window_size, C) \
            .permute(0,1,3,2,4,5) \
            .reshape(B,height*width,self.window_size,self.window_size,C) # (B,T,C) -> (B, H*W, Wnd, Wnd, C)
        
        # Process all windows in parallel, dims 0 and 1 flattened due to attn impl. requirimg 3 dim tensors
        num_windows = height * width
        x_flat = x.reshape(B*num_windows, self.window_size*self.window_size, C)  # (B*num_windows, window_size*window_size, C)
        att_out = self.att_local(x_flat)  # (B*num_windows, window_size*window_size, C)
        x = (x.reshape(B*num_windows, self.window_size*self.window_size, C) + att_out) \
            .reshape(B, num_windows, self.window_size, self.window_size, C)
        
        #switch back to (B, T, C)
        x = x\
            .reshape(B, height, width, self.window_size, self.window_size, C)\
            .permute(0,1,3,2,4,5)\
            .reshape(B, self.n_patches, self.n_patches, C)\
            .flatten(start_dim=1,end_dim=2)

        out = x + self.ffw(self.lnorm_2(x))
        return out

class SW_MSA(nn.Module): # Shifted Window Multi-Head Self Attention
    def __init__(self, config: DeformableViT_config):
        super().__init__()
        self.lnorm_1 = nn.LayerNorm(config.d_model)
        self.att = MultiHeadAttention(config)
        self.lnorm_2 = nn.LayerNorm(config.d_model)
        self.ffw = FeedForward(config)

    def forward(self, x):
        x = x + self.att(self.lnorm_1(x))
        x = x + self.ffw(self.lnorm_2(x))
        return x


class DeformableViT(nn.Module):
    def __init__(self, config: DeformableViT_config, n_classes=10):
        super().__init__()
        self.config = config
        self.patchAndEmbed = PatchEmbed(config)
        self.blocks = nn.Sequential(*[W_MSA(config) for _ in range(config.n_layers)])
        self.out_mlp = nn.Linear(config.d_model, n_classes, bias=False)
        # self.output_row = nn.Parameter(torch.rand(1,1,config.d_model)) # Parameter makes sure that Tensor will me added to ViT.parameters()
        self.out_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        B, C, H, W = x.shape # (B, C, H, W) -> cut squares of P x P from channels
        x = self.patchAndEmbed(x) # linear projection image -> patches -> (B, n_patches^2, d_model)
        # add addtitional token for output TODO: does it matter if it is the last or first row ?
        # expanded to be indifferent to examples (each example the same output row, so it can train to be a good output row)
        x = self.blocks(x)
        x = self.out_layer_norm(x)
        x = F.avg_pool1d(x.permute(0,2,1), kernel_size=self.config.n_patches**2).reshape(B, -1)
        out = self.out_mlp(x)
        return out