import math

import torch
import torch.nn as nn

# DiT-S/2-like settings on a 32×32×4 latent · 类 DiT-S/2，输入 32×32×4 潜变量
INPUT_SIZE = 32
PATCH = 2
IN_CH = 4
HIDDEN = 384
DEPTH = 4
HEADS = 6
NUM_CLASSES = 10


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of the diffusion step t followed by an MLP.
    zh: 扩散时间步 t 的正弦编码，再经过 MLP。"""

    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class DiTBlock(nn.Module):
    """Transformer block whose LayerNorms are modulated by the condition c (adaLN-Zero).
    zh: 由条件 c 调制层归一化（adaLN-Zero）的 Transformer 块。"""

    def __init__(self, hidden, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(h, h, h)[0]
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class DiT(nn.Module):
    """Diffusion Transformer (Peebles & Xie, 2022): predicts noise for latent patches, conditioned on timestep and class.
    zh: 扩散 Transformer（Peebles & Xie，2022）：以时间步和类别为条件，预测潜变量 patch 的噪声。"""

    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Conv2d(IN_CH, HIDDEN, kernel_size=PATCH, stride=PATCH)
        self.t_embedder = TimestepEmbedder(HIDDEN)
        self.y_embedder = nn.Embedding(NUM_CLASSES + 1, HIDDEN)   # +1 for classifier-free guidance
        num_patches = (INPUT_SIZE // PATCH) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, HIDDEN))
        self.blocks = nn.ModuleList([DiTBlock(HIDDEN, HEADS) for _ in range(DEPTH)])
        self.final_norm = nn.LayerNorm(HIDDEN, elementwise_affine=False)
        self.final_linear = nn.Linear(HIDDEN, PATCH * PATCH * IN_CH * 2)   # noise + variance

    def forward(self, x, t, y):
        x = self.x_embedder(x).flatten(2).transpose(1, 2) + self.pos_embed   # [N, T, D]
        c = self.t_embedder(t) + self.y_embedder(y)                         # condition
        for block in self.blocks:
            x = block(x, c)
        x = self.final_linear(self.final_norm(x))
        h = w = INPUT_SIZE // PATCH
        x = x.reshape(x.shape[0], h, w, PATCH, PATCH, IN_CH * 2)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], IN_CH * 2, h * PATCH, w * PATCH)


if __name__ == "__main__":
    model = DiT()
    latent = torch.randn(1, IN_CH, INPUT_SIZE, INPUT_SIZE)
    t = torch.randint(0, 1000, (1,))
    y = torch.randint(0, NUM_CLASSES, (1,))
    noise_pred = model(latent, t, y)
