import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


class graphmodule(nn.Module):
    def __init__(self, C_in, C_out, dim_k=32, heads=8):
        super().__init__()
        self.heads = heads
        self.k = dim_k

        assert (C_out % heads) == 0, 'values dimension must be integer'
        dim_v = C_out // heads

        self.conv_q = nn.Conv1d(C_in, dim_k * heads, 1, bias=False)
        self.conv_k = nn.Conv1d(C_in, dim_k, 1, bias=False)
        self.conv_v = nn.Conv1d(C_in, dim_v, 1, bias=False)

        self.norm_q = nn.BatchNorm1d(dim_k * heads)
        self.norm_v = nn.BatchNorm1d(dim_v)

        self.blocker = nn.BatchNorm1d(C_out)
        self.skip = nn.Conv1d(C_out, C_out, 1)

        # multi-dimensional adjacency matrix
        self.A = nn.Parameter(torch.randn(dim_v, dim_v, dim_k), requires_grad=True)

    def forward(self, x):
        '''
        :param x: [B, C_in, N]
        :return: out: [B, C_out, N]
        '''
        query = self.conv_q(x)  # [B, head * C_k, N]
        key = self.conv_k(x)  # [B, C_k, N]
        value = self.conv_v(x)  # [B, C_v, N]

        # normalization
        query = self.norm_q(query)
        value = self.norm_v(value)
        key = key.softmax(dim=-1)

        query = rearrange(query, 'b (h k) n -> b h k n', h=self.heads)  # [B, head, C_k, N]
        k_v_attn = einsum('b k n, b v n -> b k v', key, value)  # [B, C_k, C_v]
        Yc = einsum('b h k n, b k v -> b n h v', query, k_v_attn)  # [B, N, head, C_v]

        
        G = einsum('b v n, w v k -> b n k w', value, self.A).contiguous()  # A*x: [B, N, C_k, C_v]
        value = rearrange(value, 'b v n -> b n (1) v').contiguous()
        G = F.relu(G + value)  # [B, N, C_k, C_v]
        Yp = einsum('b h k n, b n k v -> b n h v', query, G)  # [B, N, head, C_v]

        out = Yc + Yp
        out = rearrange(out, 'b n h v -> b n (h v)')
        out = rearrange(out, 'b n c -> b c n')
        out = self.blocker(self.skip(out))

        return F.relu(out + x)
