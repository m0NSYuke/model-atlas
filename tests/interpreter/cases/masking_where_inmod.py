import torch
import torch.nn as nn
import torch.nn.functional as F
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1, 1)
    def forward(self, dummy):
        T = 6
        att = torch.randn(2, 4, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = torch.where(att > 0.1, att, torch.zeros_like(att))
        z = torch.clamp(y, min=0.0, max=1.0).log_softmax(-1)
        print(z.shape)
        return dummy
if __name__ == "__main__":
    M()(torch.randn(1))
