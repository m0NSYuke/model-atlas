# ── Model Atlas import template · 导入模板 ─────────────────────────────────────────
# model.py (required · 必需): only the network. Every nn.Module class becomes a block in 3D.
# zh: 这里只写网络本身。每个 nn.Module 类都会变成 3D 里的一个可展开模块。
#
# Rules · 规则
#   1. Use torch / torch.nn / torch.nn.functional / einops. Other libraries are not available.
#      zh: 只用 torch、torch.nn、torch.nn.functional、einops，其它库（timm、transformers…）用不了。
#   2. Sizes must be plain numbers or come from config values, not from real data.
#      zh: 尺寸必须是数字或来自配置，不能依赖真实数据的数值。
#   3. Keep the example call under `if __name__ == "__main__":` so train.py can reuse this file.
#      zh: 示例调用放在 `if __name__ == "__main__":` 里，train.py 才能复用这个文件。
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
IMAGE_SIZE = 32


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool: halves height and width.
    zh: 卷积 → 批归一化 → ReLU → 最大池化：高宽减半。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class SmallCNN(nn.Module):
    """A small image classifier: three ConvBlocks, global pooling, then a linear head.
    zh: 小型图像分类网络：三个卷积块、全局池化，再接线性分类头。"""

    def __init__(self, widths=(32, 64, 128), num_classes=NUM_CLASSES):
        super().__init__()
        chans = [3, *widths]
        self.features = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):                      # x: [B, 3, 32, 32]
        x = self.features(x)                   # [B, 128, 4, 4]
        x = self.pool(x).flatten(1)            # [B, 128]
        return self.head(x)                    # [B, num_classes] logits


if __name__ == "__main__":
    # Example call · 示例调用: one fake input is enough to draw the network.
    model = SmallCNN()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    logits = model(x)
