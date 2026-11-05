
import math

import numpy as np
import torch
from einops import rearrange
from torch import nn


class PositionalEmbedding(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        B, L = pos_seq.shape
        sinusoid_inp = torch.ger(rearrange(pos_seq, "B L -> (B L)"), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return rearrange(pos_emb, "(B L) E -> B L E", B=B, L=L)


class FourierEncode(nn.Module):

    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size)).float(), requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1.0 / embed_size)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(-1)
        encode = torch.cos(x * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1))
        return self.div_term * encode
