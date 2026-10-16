import torch
import torch.nn as nn
import torch.nn.functional as F

import math
class Net(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.blocks = nn.ModuleDict({f"b{i}": nn.Linear(8, 8) for i in range(n)})
        self.extra = nn.Identity()
    def forward(self, x):
        try:
            for k in sorted(self.blocks.keys()):
                x = self.blocks[k](x)
        finally:
            pass
        assert x.dim() == 2, "bad"
        if hasattr(self, "extra") and isinstance(self.extra, nn.Module):
            x = getattr(self, "extra")(x)
        if (n := x.shape[-1]) > 4:
            x = x * math.sqrt(n)
        return x if 0 < n <= 8 else x * 0
if __name__ == "__main__":
    print(Net()(torch.randn(2, 8)).shape)
