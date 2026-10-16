import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# A tiny text-to-video model: 3D VAE + spatiotemporal DiT trained with flow matching.
# zh: 迷你文生视频模型：3D VAE + 时空 DiT，用流匹配训练。
FRAMES = 16
SIZE = 64
LATENT_CH = 8
PATCH = 2
DIM = 256
HEADS = 4
DEPTH = 2
TEXT_DIM = 256
TOKENS = 16
VOCAB = 1000


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
    args = t[:, None].float() * 1000 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class VideoEncoder(nn.Module):
    """3D VAE encoder: compresses time 4× and space 4× with strided 3D convolutions.
    zh: 3D VAE 编码器：用带步长的三维卷积把时间压缩 4 倍、空间压缩 4 倍。"""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(3, 64, 3, padding=1)
        self.down1 = nn.Conv3d(64, 128, 3, stride=2, padding=1)
        self.down2 = nn.Conv3d(128, 128, 3, stride=2, padding=1)
        self.to_latent = nn.Conv3d(128, LATENT_CH, 1)

    def forward(self, video):
        h = F.silu(self.conv_in(video))
        h = F.silu(self.down1(h))
        h = F.silu(self.down2(h))
        return self.to_latent(h)


class VideoDecoder(nn.Module):
    """3D VAE decoder: latent video back to frames.
    zh: 3D VAE 解码器：把潜空间视频还原成帧。"""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(LATENT_CH, 128, 3, padding=1)
        self.up1 = nn.ConvTranspose3d(128, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)
        self.conv_out = nn.Conv3d(64, 3, 3, padding=1)

    def forward(self, z):
        h = F.silu(self.conv_in(z))
        h = F.silu(self.up1(h))
        h = F.silu(self.up2(h))
        return self.conv_out(h)


class STBlock(nn.Module):
    """Spatiotemporal DiT block: attention within each frame, then across frames at each position,
    then cross-attention to the prompt; adaLN injects the flow time.
    zh: 时空 DiT 块：先在每帧内部做注意力，再在同一位置跨帧做注意力，然后交叉注意提示词；adaLN 注入流时间。"""

    def __init__(self):
        super().__init__()
        self.ada = nn.Linear(DIM, 6 * DIM)
        self.norm_s = nn.LayerNorm(DIM, elementwise_affine=False)
        self.spatial = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.norm_t = nn.LayerNorm(DIM, elementwise_affine=False)
        self.temporal = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.norm_c = nn.LayerNorm(DIM)
        self.cross = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.norm_m = nn.LayerNorm(DIM, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))

    def forward(self, x, cond, text, T, N):
        B = x.shape[0]
        shift1, scale1, shift2, scale2, gate1, gate2 = self.ada(cond)[:, None, :].chunk(6, dim=-1)
        # Spatial attention: tokens of one frame · 空间注意力：同一帧内的 token
        h = (self.norm_s(x) * (1 + scale1) + shift1).reshape(B * T, N, DIM)
        x = x + gate1 * self.spatial(h, h, h)[0].reshape(B, T * N, DIM)
        # Temporal attention: one position across frames · 时间注意力：同一位置跨帧
        h = self.norm_t(x).reshape(B, T, N, DIM).transpose(1, 2).reshape(B * N, T, DIM)
        h = self.temporal(h, h, h)[0].reshape(B, N, T, DIM).transpose(1, 2).reshape(B, T * N, DIM)
        x = x + h
        # Text cross-attention · 文本交叉注意力
        x = x + self.cross(self.norm_c(x), text, text)[0]
        h = self.norm_m(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class VideoDiT(nn.Module):
    """Text-to-video generation (Sora / Open-Sora style): encode the clip, add noise along a flow,
    predict the velocity with a spatiotemporal Transformer, decode the one-step estimate.
    zh: 文生视频（Sora / Open-Sora 思路）：编码视频片段，沿流加噪，用时空 Transformer 预测速度，再解码一步估计。"""

    def __init__(self):
        super().__init__()
        self.encoder = VideoEncoder()
        self.text = nn.Sequential(nn.Embedding(VOCAB, TEXT_DIM), nn.TransformerEncoderLayer(TEXT_DIM, 4, 4 * TEXT_DIM, batch_first=True))
        self.patchify = nn.Conv3d(LATENT_CH, DIM, kernel_size=(1, PATCH, PATCH), stride=(1, PATCH, PATCH))
        T, H = FRAMES // 4, SIZE // 4 // PATCH
        self.position = nn.Parameter(torch.zeros(1, T * H * H, DIM))
        self.time_mlp = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.blocks = nn.ModuleList([STBlock() for _ in range(DEPTH)])
        self.norm_out = nn.LayerNorm(DIM)
        self.unpatchify = nn.Linear(DIM, PATCH * PATCH * LATENT_CH)
        self.decoder = VideoDecoder()

    def forward(self, video, ids, noise, t):
        z0 = self.encoder(video)                                         # [B, C, T, H, W]
        text = self.text(ids)
        z_t = t[:, None, None, None, None] * noise + (1 - t[:, None, None, None, None]) * z0
        B, C, T, H, W = z_t.shape
        x = self.patchify(z_t).flatten(2).transpose(1, 2) + self.position  # spacetime patches · 时空 patch
        cond = self.time_mlp(timestep_embedding(t, DIM))
        for block in self.blocks:
            x = block(x, cond, text, T, (H // PATCH) * (W // PATCH))
        x = self.unpatchify(self.norm_out(x))
        x = x.reshape(B, T, H // PATCH, W // PATCH, PATCH, PATCH, C)
        v = torch.einsum("bthwpqc->bcthpwq", x).reshape(B, C, T, H, W)  # predicted velocity · 预测速度
        loss = F.mse_loss(v, noise - z0)
        return loss, self.decoder(z_t - t[:, None, None, None, None] * v)


if __name__ == "__main__":
    model = VideoDiT()
    video = torch.randn(1, 3, FRAMES, SIZE, SIZE)
    ids = torch.randint(0, VOCAB, (1, TOKENS))
    noise = torch.randn(1, LATENT_CH, FRAMES // 4, SIZE // 4, SIZE // 4)
    t = torch.rand(1)
    loss, frames = model(video, ids, noise, t)
