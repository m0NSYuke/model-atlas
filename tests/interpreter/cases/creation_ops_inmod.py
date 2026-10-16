import torch
import torch.nn as nn
import torch.nn.functional as F
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1, 1)
    def forward(self, dummy):
        a = torch.arange(0, 10, 2); b = torch.linspace(0, 1, 5); c = torch.full((2, 3), 7.0)
        d = torch.eye(4); e = torch.ones_like(d); f = torch.rand(3, 3); g = torch.randn_like(f)
        h = torch.zeros(2, 3, dtype=torch.long); i = torch.tensor([1.0, 2.0, 3.0]); j = torch.empty(5)
        k = torch.cumsum(i, 0); l = torch.cat([i, i]); m = torch.meshgrid(a.float(), b, indexing="ij")
        print(a.shape, b.shape, c.shape, d.shape, e.shape, g.shape, h.shape, i.shape, j.shape, k.shape, l.shape, m[0].shape)
        return dummy
if __name__ == "__main__":
    M()(torch.randn(1))
