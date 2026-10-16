import torch
import torch.nn as nn
import torch.nn.functional as F
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1, 1)
    def forward(self, dummy):
        q = torch.randn(2, 4, 10, 16)
        o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        img = torch.randn(1, 3, 15, 15)
        a = F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False)
        b = F.interpolate(img, size=(7, 9), mode="nearest")
        c = F.pad(img, (1, 2, 3, 4))
        d = F.normalize(q, dim=-1)
        e = F.avg_pool2d(img, 3, stride=2, padding=1)
        f = F.max_pool2d(img, 2)
        g = F.dropout(q, 0.1, training=True)
        h = F.one_hot(torch.randint(0, 5, (4,)), num_classes=5)
        print(o.shape, a.shape, b.shape, c.shape, d.shape, e.shape, f.shape, g.shape, h.shape)
        return dummy
if __name__ == "__main__":
    M()(torch.randn(1))
