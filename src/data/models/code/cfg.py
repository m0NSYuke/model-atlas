import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classifier-free guidance around a small class-conditional denoiser.
# zh: 在一个小型类别条件去噪器外面包一层无分类器引导。
IMAGE_SIZE = 32
WIDTH = 128
NUM_CLASSES = 10
GUIDANCE = 4.0


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Residual block modulated by the condition c = time + class.
    zh: 由条件 c（时间 + 类别）调制的残差块。"""

    def __init__(self, ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.cond = nn.Linear(WIDTH, ch)
        self.norm2 = nn.GroupNorm(32, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x))) + self.cond(c)[:, :, None, None]
        return x + self.conv2(F.silu(self.norm2(h)))


class Denoiser(nn.Module):
    """ε_θ(x_t, t, y). The extra class index NUM_CLASSES means "no condition" (∅).
    zh: ε_θ(x_t, t, y)。额外的类别编号 NUM_CLASSES 表示“无条件”（∅）。"""

    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.label = nn.Embedding(NUM_CLASSES + 1, WIDTH)
        self.inc = nn.Conv2d(3, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(WIDTH) for _ in range(2)])
        self.out = nn.Conv2d(WIDTH, 3, 3, padding=1)

    def forward(self, x, t, y):
        c = self.time(timestep_embedding(t, WIDTH)) + self.label(y)
        h = self.inc(x)
        for block in self.blocks:
            h = block(h, c)
        return self.out(h)


class ClassifierFreeGuidance(nn.Module):
    """CFG (Ho & Salimans, 2022): run the same denoiser with and without the label and push the
    prediction away from the unconditional one: ε = ε_∅ + w·(ε_y − ε_∅).
    zh: 无分类器引导（Ho & Salimans，2022）：同一个去噪器分别带标签、不带标签各跑一次，
    再把预测推离无条件结果：ε = ε_∅ + w·(ε_y − ε_∅)。"""

    def __init__(self):
        super().__init__()
        self.denoiser = Denoiser()

    def forward(self, x_t, t, y):
        eps_cond = self.denoiser(x_t, t, y)
        eps_uncond = self.denoiser(x_t, t, torch.full_like(y, NUM_CLASSES))
        return eps_uncond + GUIDANCE * (eps_cond - eps_uncond)


if __name__ == "__main__":
    model = ClassifierFreeGuidance()
    x_t = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    t = torch.randint(0, 1000, (1,))
    y = torch.randint(0, NUM_CLASSES, (1,))
    eps = model(x_t, t, y)
