import torch
import torch.nn as nn
import torch.nn.functional as F
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1, 1)
    def forward(self, dummy):
        x = torch.randn(2, 3, 4, 5)
        a = x.view(2, -1).reshape(2, 3, 20).permute(0, 2, 1).contiguous()
        b = x.flatten(1, 2).unsqueeze(1).squeeze(1).transpose(-1, -2)
        c = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1).repeat(1, 1, 2, 1)
        d = torch.stack([x, x], dim=0).unbind(0)[0]
        e1, e2 = x.split([1, 2], dim=1)
        f = x.expand_as(x).sum(-1)
        g = x.new_zeros(2, 3).float().half().to(torch.float32)
        print(a.shape, b.shape, c.shape, d.shape, e1.shape, e2.shape, f.shape, g.shape, x.size(-1), x.numel())
        return dummy
if __name__ == "__main__":
    M()(torch.randn(1))
