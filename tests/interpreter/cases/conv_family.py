import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(8, 16, 3, padding=1)
        self.c3 = nn.Conv3d(3, 8, (3, 3, 3), stride=(1, 2, 2), padding=1)
        self.ct = nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1)
        self.dw = nn.Conv2d(4, 4, 3, padding=1, groups=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.gn = nn.GroupNorm(2, 4)
        self.inn = nn.InstanceNorm2d(4, affine=True)
        self.ps = nn.PixelShuffle(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, a, v, img):
        a = self.bn1(self.c1(a))
        v = self.c3(v)
        img = self.inn(self.gn(self.dw(self.ct(img))))
        img = self.ps(img)
        img = self.up(img)
        return a, v, self.pool(img).flatten(1)
if __name__ == "__main__":
    out = Net()(torch.randn(2, 8, 50), torch.randn(1, 3, 4, 16, 16), torch.randn(1, 8, 8, 8))
    print([o.shape for o in out])
