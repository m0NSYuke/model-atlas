import torch
import torch.nn as nn

WIDTHS = [32, 64, 128, 256]   # one entry per resolution level · 每个分辨率层级一个宽度


class DoubleConv(nn.Module):
    """(Conv 3×3 → BatchNorm → ReLU) × 2.
    zh: （3×3 卷积 → 批归一化 → ReLU）× 2。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Halve the resolution with max pooling, then DoubleConv.
    zh: 先用最大池化把分辨率减半，再做 DoubleConv。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Double the resolution, concatenate the matching encoder feature (skip connection), then DoubleConv.
    zh: 分辨率加倍，拼接对应编码器特征（跳跃连接），再做 DoubleConv。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015): a contracting path, an expanding path and skip connections between them.
    zh: U-Net（Ronneberger 等，2015）：收缩路径、扩张路径，以及两者之间的跳跃连接。"""

    def __init__(self, in_ch=3, num_classes=2, widths=WIDTHS):
        super().__init__()
        self.inc = DoubleConv(in_ch, widths[0])
        self.downs = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.ups = nn.ModuleList([Up(widths[i + 1], widths[i]) for i in reversed(range(len(widths) - 1))])
        self.outc = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for up in self.ups:
            x = up(x, skips.pop())
        return self.outc(x)


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 3, 128, 128)
    mask_logits = model(x)
