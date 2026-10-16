import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 16)
        self.out = nn.Linear(16, 50, bias=False)
        self.out.weight = self.emb.weight
        for p in self.emb.parameters():
            p.requires_grad = False
        self.emb.requires_grad_(False)
    def forward(self, ids):
        return self.out(self.emb(ids))
if __name__ == "__main__":
    m = Net(); m.eval(); m.train()
    print(m(torch.randint(0, 50, (2, 4))).shape, sum(p.numel() for p in m.parameters()))
