import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 4); self.b = nn.Linear(8, 2)
    def forward(self, x, *, return_dict=True, **kwargs):
        out = {"cls": self.a(x), "box": self.b(x)}
        return out if return_dict else (out["cls"], out["box"])
if __name__ == "__main__":
    o = Net()(x=torch.randn(3, 8))
    print({k: v.shape for k, v in o.items()})
