import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomModules.config import ViT_config
from CustomModules.attention import PatchEmbed, Block

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ViT(nn.Module):
    def __init__(self, config: ViT_config, n_classes=10):
        super().__init__()
        self.patchAndEmbed = PatchEmbed(config)
        self.pos_embed = nn.Embedding(config.n_patches * config.n_patches + 1, config.d_model)
        self.postion_indexes = torch.tensor([_ for _ in range(config.n_patches*config.n_patches+1)]).to(device)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.out_mlp = nn.Linear(config.d_model, n_classes)
        self.output_row = nn.Parameter(torch.rand(1,1,config.d_model)) # Parameter makes sure that Tensor will me added to ViT.parameters()
        self.out_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        B, C, H, W = x.shape # (B, C, H, W) -> cut squares of P x P from channels
        x = self.patchAndEmbed(x) # linear projection image -> patches -> (B, n_patches^2, d_model)
        # add addtitional token for output TODO: does it matter if it is the last or first row ?
        # expanded to be indifferent to examples (each example the same output row, so it can train to be a good output row)
        x = torch.cat([self.output_row.expand(x.shape[0],-1,-1), x], dim=1)
        x = x + self.pos_embed(self.postion_indexes)
        x = self.blocks(x)
        x = self.out_layer_norm(x)
        out = self.out_mlp(x[:, 0, :])
        return out