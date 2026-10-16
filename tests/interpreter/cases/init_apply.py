import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.c = nn.Conv2d(3, 8, 3)
        self.apply(self._init)
        nn.init.trunc_normal_(self.a.weight, std=0.02)
        nn.init.zeros_(self.a.bias)
        nn.init.kaiming_normal_(self.c.weight, mode="fan_out", nonlinearity="relu")
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.c(x).mean(dim=(2, 3)) @ self.a.weight
if __name__ == "__main__":
    m = Net()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(n, m(torch.randn(1, 3, 10, 10)).shape)
