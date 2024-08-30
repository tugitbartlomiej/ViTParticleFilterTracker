import torch
import torch.nn as nn
from einops import repeat

from attention import Attention
from feed_forward import FeedForward
from patch_embedding import PatchEmbedding
from prenorm import PreNorm
from residual import ResidualAdd


class VisionTransformer(nn.Module):
    def __init__(self, img_size=144, patch_size=8, emb_size=128, num_layers=6, num_heads=4, num_classes=37, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size, emb_size=emb_size)

        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

        self.layers = nn.ModuleList([
            nn.Sequential(
                ResidualAdd(PreNorm(emb_size, Attention(emb_size, num_heads, dropout))),
                ResidualAdd(PreNorm(emb_size, FeedForward(emb_size, emb_size * 2, dropout)))
            )
            for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :n + 1]

        for layer in self.layers:
            x = layer(x)

        return self.mlp_head(x[:, 0])