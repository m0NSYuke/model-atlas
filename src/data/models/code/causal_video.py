import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Block-causal video continuation: earlier frames are clean context, the newest frame is being denoised.
# zh: 分块因果的视频续写：前面的帧是干净的上下文，最新一帧正在去噪。
CONTEXT_FRAMES = 3
LATENT_CH = 8
GRID = 8             # latent frames are 8×8 · 每帧潜变量为 8×8
PATCH = 2
DIM = 256
DEPTH = 3


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
    args = t[:, None].float() * 1000 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class CausalBlock(nn.Module):
    """Transformer block whose attention mask is block-causal over frames: a token sees its own frame
    and every earlier frame, never a later one.
    zh: 注意力掩码按帧分块因果的 Transformer 块：token 能看到本帧和之前所有帧，看不到之后的帧。"""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.proj = nn.Linear(DIM, DIM)
        self.norm2 = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))

    def forward(self, x, mask, cond):
        q, k, v = self.qkv(self.norm1(x + cond)).chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(DIM)
        scores = scores.masked_fill(mask == 0, float("-inf"))   # hide future frames · 遮住未来帧
        x = x + self.proj(scores.softmax(dim=-1) @ v)
        return x + self.mlp(self.norm2(x))


class CausalVideoGenerator(nn.Module):
    """Autoregressive video diffusion (diffusion-forcing / self-forcing style): clean past frames and one
    noisy new frame share a block-causal Transformer, which predicts the new frame's velocity.
    zh: 自回归视频扩散（Diffusion Forcing / Self-Forcing 思路）：干净的历史帧与一帧带噪新帧共用一个分块因果
    Transformer，由它预测新帧的速度。"""

    def __init__(self):
        super().__init__()
        self.patchify = nn.Conv2d(LATENT_CH, DIM, PATCH, stride=PATCH)
        self.frame_position = nn.Parameter(torch.zeros(1, CONTEXT_FRAMES + 1, 1, DIM))
        self.time_mlp = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.blocks = nn.ModuleList([CausalBlock() for _ in range(DEPTH)])
        self.out = nn.Linear(DIM, PATCH * PATCH * LATENT_CH)

    def forward(self, context, noisy, t):
        B = noisy.shape[0]
        frames = torch.cat([context, noisy[:, None]], dim=1)                     # [B, F, C, H, W]
        F_, N = CONTEXT_FRAMES + 1, (GRID // PATCH) ** 2
        x = self.patchify(frames.reshape(B * F_, LATENT_CH, GRID, GRID)).flatten(2).transpose(1, 2)
        x = (x.reshape(B, F_, N, DIM) + self.frame_position).reshape(B, F_ * N, DIM)
        # Frame-level lower-triangular mask, expanded to tokens · 帧级下三角掩码，展开到 token
        mask = torch.tril(torch.ones(F_, F_))[:, None, :, None].repeat(1, N, 1, N).reshape(F_ * N, F_ * N)
        cond = self.time_mlp(timestep_embedding(t, DIM))[:, None, :]
        for block in self.blocks:
            x = block(x, mask, cond)
        new_frame = self.out(x[:, -N:])                                          # only the newest frame · 只取最新一帧
        h = GRID // PATCH
        new_frame = new_frame.reshape(B, h, h, PATCH, PATCH, LATENT_CH)
        return torch.einsum("bhwpqc->bchpwq", new_frame).reshape(B, LATENT_CH, GRID, GRID)


if __name__ == "__main__":
    model = CausalVideoGenerator()
    context = torch.randn(1, CONTEXT_FRAMES, LATENT_CH, GRID, GRID)   # cached past latents · 缓存的历史潜变量
    noisy = torch.randn(1, LATENT_CH, GRID, GRID)
    t = torch.rand(1)
    velocity = model(context, noisy, t)
