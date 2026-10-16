import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(64, 4, batch_first=True)
        self.mha2 = nn.MultiheadAttention(64, 4)
    def forward(self, x):
        y, w = self.mha(x, x, x, need_weights=True)
        z, _ = self.mha2(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        return y + z.transpose(0, 1), w
if __name__ == "__main__":
    print([o.shape for o in Net()(torch.randn(2, 10, 64))])
