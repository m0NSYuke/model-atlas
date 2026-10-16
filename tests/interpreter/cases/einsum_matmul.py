import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    q = torch.randn(2, 4, 8, 16); k = torch.randn(2, 4, 8, 16)
    a = torch.einsum("bhqd,bhkd->bhqk", q, k)
    b = torch.einsum("bij,bjk->bik", torch.randn(3, 4, 5), torch.randn(3, 5, 6))
    c = torch.einsum("i,j->ij", torch.randn(4), torch.randn(5))
    d = torch.einsum("...d,de->...e", q, torch.randn(16, 32))
    e = torch.bmm(torch.randn(3, 4, 5), torch.randn(3, 5, 2))
    f = q @ k.transpose(-2, -1)
    g = torch.outer(torch.randn(3), torch.randn(4))
    print(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape, g.shape)
