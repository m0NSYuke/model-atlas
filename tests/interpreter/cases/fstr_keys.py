import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.blocks = nn.ModuleDict({f"b{i}": nn.Linear(8, 8) for i in range(n)})
        for i in range(n):
            setattr(self, f"layer{i}", nn.Linear(8, 8))
    def forward(self, x):
        for i in range(3):
            x = getattr(self, f"layer{i}")(self.blocks[f"b{i}"](x))
        return x
if __name__ == "__main__":
    M()(torch.randn(2, 8))
