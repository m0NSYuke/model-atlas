import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# A small DDPM on 32×32 RGB images · 32×32 彩色图像上的小型 DDPM
IMAGE_SIZE = 32
WIDTHS = [64, 128, 256]   # one entry per resolution level · 每个分辨率层级一个通道宽度
TIME_DIM = 256
STEPS = 1000              # diffusion steps T · 扩散步数 T


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Two 3×3 convolutions with GroupNorm and SiLU; the timestep embedding is added in between.
    zh: 两个 3×3 卷积（GroupNorm + SiLU），中间加入时间步嵌入。"""

    def __init__(self, in_ch, out_ch, time_dim=TIME_DIM):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(emb)[:, :, None, None]       # inject t · 注入时间步
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over all pixels of a low-resolution feature map.
    zh: 在低分辨率特征图的所有像素之间做单头自注意力。"""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.proj(h.transpose(1, 2).reshape(B, C, H, W))


class UNet(nn.Module):
    """Noise predictor ε_θ(x_t, t): encoder, attention bottleneck, decoder with skip connections.
    zh: 噪声预测器 ε_θ(x_t, t)：编码路径、带注意力的瓶颈、带跳跃连接的解码路径。"""

    def __init__(self, widths=WIDTHS):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(widths[0], TIME_DIM), nn.SiLU(), nn.Linear(TIME_DIM, TIME_DIM))
        self.inc = nn.Conv2d(3, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([ResBlock(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.mid = ResBlock(widths[-1], widths[-1])
        self.mid_attn = SelfAttention(widths[-1])
        self.up_blocks = nn.ModuleList([ResBlock(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(len(widths) - 1))])
        self.out = nn.Sequential(nn.GroupNorm(32, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], 3, 3, padding=1))

    def forward(self, x, t):
        emb = self.time_mlp(timestep_embedding(t, WIDTHS[0]))
        h = self.inc(x)
        skips = []
        for block in self.down_blocks:
            skips.append(h)
            h = F.avg_pool2d(h, 2)
            h = block(h, emb)
        h = self.mid_attn(self.mid(h, emb))
        for block in self.up_blocks:
            h = F.interpolate(h, scale_factor=2)
            h = block(torch.cat([h, skips.pop()], dim=1), emb)   # skip connection · 跳跃连接
        return self.out(h)


class DDPM(nn.Module):
    """DDPM training step (Ho et al., 2020): noise a clean image to step t, then predict that noise.
    zh: DDPM 训练一步（Ho 等，2020）：把干净图像加噪到第 t 步，再预测所加的噪声。"""

    def __init__(self):
        super().__init__()
        self.eps_model = UNet()
        betas = torch.linspace(1e-4, 0.02, STEPS)
        self.register_buffer("alpha_bar", torch.exp(torch.cumsum(torch.log(1 - betas), dim=0)))   # ᾱ_t

    def forward(self, x0, t, noise):
        a = self.alpha_bar[t][:, None, None, None]
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise         # q(x_t | x_0) · 前向加噪
        eps = self.eps_model(x_t, t)
        return F.mse_loss(eps, noise)                         # predict the noise · 预测噪声


if __name__ == "__main__":
    model = DDPM()
    x0 = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    t = torch.randint(0, STEPS, (1,))
    noise = torch.randn_like(x0)
    loss = model(x0, t, noise)
