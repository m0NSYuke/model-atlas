import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3×3 convolutions plus a shortcut. When the shape changes, a 1×1 convolution projects the shortcut.
    zh: 两个 3×3 卷积加一条捷径；当形状改变时，用 1×1 卷积投影捷径。"""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)        # residual connection · 残差连接
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet for 32×32 images (He et al., 2015). num_blocks=(2, 2, 2, 2) is ResNet-18.
    zh: 面向 32×32 图像的 ResNet（He 等，2015）。num_blocks=(2, 2, 2, 2) 即 ResNet-18。"""

    def __init__(self, num_blocks=(2, 2, 2, 2), widths=(64, 128, 256, 512), num_classes=10):
        super().__init__()
        self.in_planes = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        )
        self.stages = nn.ModuleList()
        for i, (n, width) in enumerate(zip(num_blocks, widths)):
            self.stages.append(self._make_stage(width, n, stride=1 if i == 0 else 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], num_classes)

    def _make_stage(self, planes, n, stride):
        blocks = []
        for s in [stride] + [1] * (n - 1):
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    model = ResNet(num_blocks=(2, 2, 2, 2))
    x = torch.randn(1, 3, 32, 32)
    logits = model(x)
