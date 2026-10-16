import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(8, 8)
    @torch.jit.ignore
    def helper(self, x):
        return x
    def forward(self, x):
        return checkpoint(self.l, x, use_reentrant=False)
if __name__ == "__main__":
    print(Net()(torch.randn(2, 8)).shape)
