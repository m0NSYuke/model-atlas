import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# DDIM sampling, unrolled for a few steps so every call of the noise predictor is visible.
# zh: 把 DDIM 采样展开几步，噪声预测器的每次调用都清晰可见。
IMAGE_SIZE = 32
WIDTH = 128
TRAIN_STEPS = 1000
SAMPLE_STEPS = [999, 666, 333, 0]   # a 3-jump schedule · 只跳 3 步的采样时间表

# ᾱ_t of the linear β schedule the noise predictor was trained with · 训练时线性 β 调度对应的 ᾱ_t
ALPHA_BAR = []
product = 1.0
for i in range(TRAIN_STEPS):
    product = product * (1 - (1e-4 + (0.02 - 1e-4) * i / (TRAIN_STEPS - 1)))
    ALPHA_BAR.append(product)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class NoisePredictor(nn.Module):
    """A DDPM-trained ε_θ(x_t, t), shrunk to three convolutions.
    zh: 用 DDPM 方式训练好的 ε_θ(x_t, t)，这里缩成三层卷积。"""

    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.SiLU())
        self.inc = nn.Conv2d(3, WIDTH, 3, padding=1)
        self.mid = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.out = nn.Conv2d(WIDTH, 3, 3, padding=1)

    def forward(self, x, t):
        h = F.silu(self.inc(x) + self.time(timestep_embedding(t, WIDTH))[:, :, None, None])
        return self.out(F.silu(self.mid(h)))


class DDIMSampler(nn.Module):
    """DDIM (Song et al., 2021): a deterministic sampler (η = 0) that can skip steps. Each step predicts
    the noise, estimates x₀, then jumps straight to the next timestep.
    zh: DDIM（Song 等，2021）：可以跳步的确定性采样器（η = 0）。每一步先预测噪声、估计 x₀，再直接跳到下一个时间步。"""

    def __init__(self):
        super().__init__()
        self.eps_model = NoisePredictor()

    def forward(self, x):
        for step in range(len(SAMPLE_STEPS) - 1):
            t, t_next = SAMPLE_STEPS[step], SAMPLE_STEPS[step + 1]
            a, a_next = ALPHA_BAR[t], ALPHA_BAR[t_next]
            eps = self.eps_model(x, torch.full((1,), t))
            x0 = (x - math.sqrt(1 - a) * eps) / math.sqrt(a)             # predicted x₀ · 预测的 x₀
            x = math.sqrt(a_next) * x0 + math.sqrt(1 - a_next) * eps     # jump to t_next · 跳到下一时间步
        return x


if __name__ == "__main__":
    model = DDIMSampler()
    x_T = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    sample = model(x_T)
