import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms = nn.RMSNorm(32)
        self.acts = nn.Sequential(nn.GELU(approximate="tanh"), nn.SiLU(), nn.Mish(), nn.Softplus(), nn.LeakyReLU(0.1), nn.PReLU(), nn.ELU(), nn.Tanh(), nn.Hardswish())
        self.sm = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.1)
        self.fl = nn.Unflatten(1, (4, 8))
    def forward(self, x):
        return self.fl(self.sm(self.drop(self.acts(self.rms(x)))))
if __name__ == "__main__":
    print(Net()(torch.randn(2, 32)).shape)
