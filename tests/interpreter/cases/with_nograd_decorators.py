import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)
    @torch.no_grad()
    def ema(self, x: Tensor) -> Tensor:
        return self.l(x)
    def forward(self, x):
        with torch.no_grad():
            y = self.l(x)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            z = self.l(x)
        return y + z.float() + self.ema(x)
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = Net().to(device)
    print(m(torch.randn(2, 4, device=device)).shape)
