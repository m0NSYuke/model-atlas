import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
def precompute_freqs(dim, T, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(T).float()
    f = torch.outer(t, freqs)
    return torch.cos(f), torch.sin(f)
def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)
class Block(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.h = h; self.hd = d // h
        self.norm1 = RMSNorm(d); self.norm2 = RMSNorm(d)
        self.wq = nn.Linear(d, d, bias=False); self.wk = nn.Linear(d, d, bias=False); self.wv = nn.Linear(d, d, bias=False); self.wo = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, 172, bias=False); self.w2 = nn.Linear(172, d, bias=False); self.w3 = nn.Linear(d, 172, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        h = self.norm1(x)
        q = self.wq(h).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(h).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.wv(h).view(B, T, self.h, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(a.transpose(1, 2).contiguous().view(B, T, C))
        h = self.norm2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))
if __name__ == "__main__":
    cos, sin = precompute_freqs(16, 12)
    print(Block()(torch.randn(2, 12, 64), cos, sin).shape)
