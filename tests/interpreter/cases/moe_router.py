import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d=32, n=4, k=2):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, n)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d)) for _ in range(n)])
    def forward(self, x):
        w, idx = torch.topk(F.softmax(self.router(x), -1), self.k, dim=-1)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = (idx == e).any(-1)
            if mask.any():
                out[mask] += expert(x[mask])
        return out
if __name__ == "__main__":
    print(MoE()(torch.randn(10, 32)).shape)
