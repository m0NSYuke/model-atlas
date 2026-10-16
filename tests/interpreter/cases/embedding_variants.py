import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(100, 16, padding_idx=0)
        self.bag = nn.EmbeddingBag(100, 16, mode="mean")
        self.pos = nn.Parameter(torch.zeros(1, 10, 16))
        self.register_buffer("mask", torch.tril(torch.ones(10, 10)))
        self.plist = nn.ParameterList([nn.Parameter(torch.randn(16)) for _ in range(2)])
        self.pdict = nn.ParameterDict({"a": nn.Parameter(torch.ones(16))})
    def forward(self, ids):
        x = self.e(ids) + self.pos[:, : ids.shape[1]]
        b = self.bag(ids)
        return x * self.plist[0] + self.pdict["a"], b
if __name__ == "__main__":
    print([o.shape for o in Net()(torch.randint(0, 100, (2, 10)))])
