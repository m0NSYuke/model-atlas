import torch
import torch.nn as nn
import torch.nn.functional as F

class Base(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d)
    def forward(self, x):
        return self.proj(x)
class Child(Base):
    def __init__(self, d):
        super(Child, self).__init__(d)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        return self.norm(super().forward(x))
if __name__ == "__main__":
    print(Child(32)(torch.randn(2, 5, 32)).shape)
