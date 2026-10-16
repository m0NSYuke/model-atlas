import torch
import torch.nn as nn
import torch.nn.functional as F

def make_mlp(*dims, act=nn.ReLU, **kw):
    layers = []
    for a, b in zip(dims, dims[1:]):
        layers += [nn.Linear(a, b, **kw), act()]
    return nn.Sequential(*layers[:-1])
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = make_mlp(16, 32, 8, bias=False)
        scale = 2.0
        def scaled(x):
            return x * scale
        self.scaled = scaled
    def forward(self, x):
        a, *rest = x.chunk(4, dim=-1)
        y = self.f(torch.cat([a, *rest], dim=-1))
        return self.scaled(y)
if __name__ == "__main__":
    print(Net()(torch.randn(3, 16)).shape)
