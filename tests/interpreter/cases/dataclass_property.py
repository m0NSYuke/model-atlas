import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
@dataclass
class Cfg:
    dim: int = 64
    hidden: List[int] = field(default_factory=lambda: [128, 64])
class Net(nn.Module):
    def __init__(self, cfg: Cfg, *, act: Optional[str] = None):
        super().__init__()
        dims = [cfg.dim] + cfg.hidden
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
    @property
    def depth(self) -> int:
        return len(self.layers)
    @staticmethod
    def act(x):
        return F.gelu(x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.depth - 1 else layer(x)
        return x
if __name__ == "__main__":
    m = Net(Cfg())
    y = m(torch.randn(2, 64))
    print(f"{y.shape}")
