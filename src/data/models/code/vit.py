import torch
import torch.nn as nn

IMG_SIZE = 32
PATCH = 4
DIM = 192
DEPTH = 6
HEADS = 3
NUM_CLASSES = 10


class PatchEmbed(nn.Module):
    """Cuts the image into P×P patches and linearly projects each patch to a token.
    zh: 把图像切成 P×P 的小块，并把每块线性投影成一个 token。"""

    def __init__(self, img_size, patch_size, in_chans, dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                        # [B, D, H/P, W/P]
        return x.flatten(2).transpose(1, 2)     # [B, N, D]


class Block(nn.Module):
    """Transformer encoder block with multi-head self-attention and an MLP.
    zh: Transformer 编码块：多头自注意力 + MLP，各带残差连接。"""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (Dosovitskiy et al., 2020): an image is a sequence of patch tokens plus a [CLS] token.
    zh: 视觉 Transformer（Dosovitskiy 等，2020）：把图像看成 patch token 序列，再加一个 [CLS] token。"""

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(IMG_SIZE, PATCH, 3, DIM)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, DIM))
        self.blocks = nn.Sequential(*[Block(DIM, HEADS) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)          # prepend [CLS]
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])               # classify from [CLS]


if __name__ == "__main__":
    model = VisionTransformer()
    images = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    logits = model(images)
