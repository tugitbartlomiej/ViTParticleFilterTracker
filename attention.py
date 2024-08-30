import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn_output, _ = self.att(q, k, v)
        return attn_output