import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rectified flow on 2D points: learn the velocity that carries noise to data.
# zh: 二维点上的整流流：学习把噪声搬运到数据的速度场。
POINTS = 256
DIM = 2
HIDDEN = 256
EULER_STEP = 0.25


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
    args = t[:, None] * 1000 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class VelocityNet(nn.Module):
    """v_θ(x_t, t): an MLP that reads the point and the time and returns a velocity.
    zh: v_θ(x_t, t)：读取点坐标与时间、输出速度的多层感知机。"""

    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(64, HIDDEN), nn.SiLU())
        self.inp = nn.Linear(DIM, HIDDEN)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, DIM))

    def forward(self, x, t):
        return self.net(self.inp(x) + self.time(time_embedding(t, 64)))


class FlowMatching(nn.Module):
    """Flow matching (Lipman et al., 2023; Liu et al., 2023): interpolate on a straight line between noise
    and data, regress the constant velocity data − noise, then integrate with Euler steps.
    zh: 流匹配（Lipman 等 2023；Liu 等 2023）：在噪声与数据之间做直线插值，回归恒定速度“数据 − 噪声”，
    采样时用欧拉步积分。"""

    def __init__(self):
        super().__init__()
        self.velocity = VelocityNet()

    def forward(self, x1, x0, t):
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1         # point on the path · 路径上的点
        v = self.velocity(x_t, t)
        loss = F.mse_loss(v, x1 - x0)                         # target velocity · 目标速度
        x_next = x_t + EULER_STEP * v                         # one Euler step · 一步欧拉积分
        return loss, x_next


if __name__ == "__main__":
    model = FlowMatching()
    x1 = torch.randn(POINTS, DIM)        # data, e.g. two moons · 数据（如双月形）
    x0 = torch.randn(POINTS, DIM)        # noise · 噪声
    t = torch.rand(POINTS)
    loss, x_next = model(x1, x0, t)
