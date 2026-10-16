import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# A miniature Stable Diffusion: 128×128 images ↔ 4×16×16 latents · 迷你 Stable Diffusion：128×128 图像 ↔ 4×16×16 潜变量
IMAGE_SIZE = 128
LATENT_CH = 4
WIDTHS = [64, 128, 128]     # U-Net widths on the latent grid · 潜空间 U-Net 各层宽度
TEXT_DIM = 256
TOKENS = 16
VOCAB = 1000
STEPS = 1000
SCALE = 0.18215             # latent scaling factor · 潜变量缩放系数


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class VAEEncoder(nn.Module):
    """Frozen autoencoder encoder: 8× spatial compression into a 4-channel latent.
    zh: 冻结的自编码器编码器：空间压缩 8 倍，得到 4 通道潜变量。"""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.to_moments = nn.Conv2d(128, 2 * LATENT_CH, 1)

    def forward(self, x):
        h = F.silu(self.conv_in(x))
        h = F.silu(self.down1(h))
        h = F.silu(self.down2(h))
        h = F.silu(self.down3(h))
        mean, logvar = self.to_moments(h).chunk(2, dim=1)
        return mean * SCALE                          # use the mean as z₀ · 取均值作为 z₀


class VAEDecoder(nn.Module):
    """Frozen autoencoder decoder: latent back to pixels.
    zh: 冻结的自编码器解码器：把潜变量还原为像素。"""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(LATENT_CH, 128, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, z):
        h = F.silu(self.conv_in(z / SCALE))
        h = F.silu(self.up1(h))
        h = F.silu(self.up2(h))
        h = F.silu(self.up3(h))
        return torch.tanh(self.conv_out(h))


class TextEncoder(nn.Module):
    """A small CLIP-style text Transformer: token ids → contextual embeddings for cross-attention.
    zh: 小型 CLIP 式文本 Transformer：把 token 编号变成供交叉注意力使用的上下文向量。"""

    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(VOCAB, TEXT_DIM)
        self.position = nn.Parameter(torch.zeros(1, TOKENS, TEXT_DIM))
        self.layers = nn.Sequential(*[nn.TransformerEncoderLayer(TEXT_DIM, 4, 4 * TEXT_DIM, batch_first=True) for _ in range(2)])
        self.norm = nn.LayerNorm(TEXT_DIM)

    def forward(self, ids):
        return self.norm(self.layers(self.token(ids) + self.position))


class ResBlock(nn.Module):
    """Residual block on the latent grid with the timestep added in the middle.
    zh: 潜空间上的残差块，中间加入时间步嵌入。"""

    def __init__(self, in_ch, out_ch, time_dim=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Every latent pixel queries the text tokens: Q from the image, K and V from the prompt.
    zh: 每个潜空间像素查询文本 token：Q 来自图像，K、V 来自提示词。"""

    def __init__(self, ch, context_dim=TEXT_DIM):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.to_q = nn.Linear(ch, ch)
        self.to_k = nn.Linear(context_dim, ch)
        self.to_v = nn.Linear(context_dim, ch)
        self.proj = nn.Linear(ch, ch)

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)              # [B, HW, C]
        out = F.scaled_dot_product_attention(self.to_q(h), self.to_k(context), self.to_v(context))
        return x + self.proj(out).transpose(1, 2).reshape(B, C, H, W)


class Stage(nn.Module):
    """ResBlock followed by text cross-attention.
    zh: 残差块后接文本交叉注意力。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch)
        self.attn = CrossAttention(out_ch)

    def forward(self, x, emb, context):
        return self.attn(self.res(x, emb), context)


class UNet(nn.Module):
    """Text-conditioned denoiser ε_θ(z_t, t, text) on the latent grid.
    zh: 在潜空间网格上、以文本为条件的去噪器 ε_θ(z_t, t, text)。"""

    def __init__(self, widths=WIDTHS):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(widths[0], 256), nn.SiLU(), nn.Linear(256, 256))
        self.inc = nn.Conv2d(LATENT_CH, widths[0], 3, padding=1)
        self.downs = nn.ModuleList([Stage(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.mid = Stage(widths[-1], widths[-1])
        self.ups = nn.ModuleList([Stage(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(len(widths) - 1))])
        self.out = nn.Conv2d(widths[0], LATENT_CH, 3, padding=1)

    def forward(self, z, t, context):
        emb = self.time_mlp(timestep_embedding(t, WIDTHS[0]))
        h = self.inc(z)
        skips = []
        for stage in self.downs:
            skips.append(h)
            h = stage(F.avg_pool2d(h, 2), emb, context)
        h = self.mid(h, emb, context)
        for stage in self.ups:
            h = stage(torch.cat([F.interpolate(h, scale_factor=2), skips.pop()], dim=1), emb, context)
        return self.out(h)


class LatentDiffusion(nn.Module):
    """Latent diffusion (Rombach et al., 2022): encode, add noise in latent space, denoise with text, decode.
    zh: 潜空间扩散（Rombach 等，2022）：编码、在潜空间加噪、以文本为条件去噪、再解码。"""

    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.text_encoder = TextEncoder()
        self.unet = UNet()
        self.decoder = VAEDecoder()
        betas = torch.linspace(1e-4, 0.02, STEPS)
        self.register_buffer("alpha_bar", torch.exp(torch.cumsum(torch.log(1 - betas), dim=0)))

    def forward(self, image, ids, t, noise):
        z0 = self.encoder(image)                                  # pixels → latent · 像素 → 潜变量
        context = self.text_encoder(ids)                          # prompt → tokens · 提示词 → 上下文
        a = self.alpha_bar[t][:, None, None, None]
        z_t = a.sqrt() * z0 + (1 - a).sqrt() * noise              # forward noising · 前向加噪
        eps = self.unet(z_t, t, context)
        loss = F.mse_loss(eps, noise)
        z0_hat = (z_t - (1 - a).sqrt() * eps) / a.sqrt()          # one-step estimate of z₀ · 一步估计 z₀
        return loss, self.decoder(z0_hat)


if __name__ == "__main__":
    model = LatentDiffusion()
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    ids = torch.randint(0, VOCAB, (1, TOKENS))
    t = torch.randint(0, STEPS, (1,))
    noise = torch.randn(1, LATENT_CH, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
    loss, preview = model(image, ids, t, noise)
