import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomModules.config import DeformableViT_config
from CustomModules.attention import MultiHeadAttention
from CustomModules.basic import FeedForward, PatchEmbed

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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