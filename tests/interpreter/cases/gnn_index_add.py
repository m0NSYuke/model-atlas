import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 32)
    def forward(self, x, edge_index):
        src, dst = edge_index
        msg = self.lin(x)[src]
        out = torch.zeros(x.size(0), 32).index_add_(0, dst, msg)
        return out
if __name__ == "__main__":
    print(GCN()(torch.randn(5, 16), torch.randint(0, 5, (2, 12))).shape)
