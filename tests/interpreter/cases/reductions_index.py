import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    x = torch.randn(4, 10)
    v, i = x.max(dim=-1)
    tv, ti = torch.topk(x, 3, dim=-1)
    am = x.argmax(-1)
    g = torch.gather(x, 1, ti)
    s = torch.zeros(4, 10).scatter(1, ti, 1.0)
    sel = x[torch.arange(4), am]
    bm = x[x > 0]
    idx = x[:, [0, 2, 4]]
    ell = torch.randn(2, 3, 4)[..., :2]
    non = x[None, :, None]
    step = x[:, ::2]
    print(v.shape, i.shape, tv.shape, am.shape, g.shape, s.shape, sel.shape, idx.shape, ell.shape, non.shape, step.shape)
