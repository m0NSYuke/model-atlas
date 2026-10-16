import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
class Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_patch = nn.Sequential(Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=4, p2=4), nn.Linear(48, 64))
        self.cls = nn.Parameter(torch.randn(1, 1, 64))
    def forward(self, img):
        x = self.to_patch(img)
        cls = repeat(self.cls, "1 1 d -> b 1 d", b=x.shape[0])
        x = torch.cat([cls, x], dim=1)
        return reduce(x, "b n d -> b d", "mean"), rearrange(x, "b n (h d) -> b h n d", h=4)
if __name__ == "__main__":
    print([o.shape for o in Patch()(torch.randn(2, 3, 32, 32))])
