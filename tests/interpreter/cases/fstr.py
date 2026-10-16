import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)
    def forward(self, x):
        y = self.l(x)
        n = 3
        print(f"{n}", f"{n + 1}", f"{y.shape}", f"{tuple(y.shape)}", f"{n:>4d}", f"a{{b}}", f'{"q"}', f"{y.shape[0]=}")
        return y
if __name__ == "__main__":
    M()(torch.randn(2, 4))
