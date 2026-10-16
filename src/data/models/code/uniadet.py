import torch
import torch.nn as nn
import torch.nn.functional as F

# UniADet at toy scale. The official code is not released yet (github.com/gaobb/UniADet), so this file
# follows the paper (arXiv 2601.05552, Sec. 3). The paper's best setting freezes DINOv2-R ViT-L/14
# (24 layers, width 1024) at 518×518 and reads blocks 12/15/18/21/24; here 8 layers, width 256, blocks 4–8.
# Assumed where the paper is silent: τ = 0.07 (as in AnomalyCLIP) and the layer maps are averaged.
# Class-Aware Augmentation (CAA) is a training-time data step and is not drawn.
# zh: 缩小版 UniADet。官方代码尚未开源，本文件按论文（arXiv 2601.05552 第 3 节）复现。
# 论文最佳配置冻结 DINOv2-R ViT-L/14（24 层、宽 1024），输入 518×518，取第 12/15/18/21/24 块；这里用 8 层、宽 256、取第 4–8 块。
# 论文未写明的部分按假设实现：τ = 0.07（同 AnomalyCLIP），各层异常图取平均。训练阶段的类感知增强（CAA）属于数据处理，未画出。
IMAGE_SIZE = 112
PATCH = 14
WIDTH = 256
DEPTH = 8
HEADS = 4
REGISTERS = 4                   # DINOv2 with registers · 带寄存器 token 的 DINOv2
FEATURES_LIST = [4, 5, 6, 7, 8]
TAU = 0.07
LAMBDA_F = 0.5                  # few-shot map weight λ_f · 少样本异常图权重
LAMBDA_P = 0.5                  # image-score fusion weight λ_p · 图像分数融合权重
SHOTS = 1                       # K normal prompt images · 正常提示图数量


class Block(nn.Module):
    """Frozen DINOv2 block: pre-norm attention and MLP, each scaled by LayerScale.
    zh: 冻结的 DINOv2 块：前置归一化的注意力和 MLP，各自带 LayerScale 缩放。"""

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ls1 = nn.Parameter(torch.ones(dim) * 1e-5)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ls2 = nn.Parameter(torch.ones(dim) * 1e-5)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.ls1 * self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class FoundationModel(nn.Module):
    """Frozen ViT backbone. Returns, for each selected block l, the global token x^l and the patch tokens F^l.
    zh: 冻结的 ViT 主干。对每个选中的块 l，输出全局 token x^l 与 patch token F^l。"""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, WIDTH, kernel_size=PATCH, stride=PATCH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.register_tokens = nn.Parameter(torch.zeros(1, REGISTERS, WIDTH))
        self.pos_embed = nn.Parameter(torch.zeros(1, (IMAGE_SIZE // PATCH) ** 2 + 1, WIDTH))
        self.blocks = nn.ModuleList([Block(WIDTH, HEADS) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)              # [B, N, C]
        b = x.shape[0]
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1) + self.pos_embed
        x = torch.cat([x[:, :1], self.register_tokens.expand(b, -1, -1), x[:, 1:]], dim=1)
        global_tokens = []
        patch_tokens = []
        idx = 0
        for block in self.blocks:
            idx += 1
            x = block(x)
            if idx in FEATURES_LIST:
                h = self.norm(x)
                global_tokens.append(h[:, 0])                            # x^l: [B, C]
                patch_tokens.append(h[:, 1 + REGISTERS:])                # F^l: [B, N, C]
        return global_tokens, patch_tokens


class DecoupledHead(nn.Module):
    """The only learnable part: per-layer weights W_cls^l and W_seg^l, each d × 2 (normal, anomaly),
    used in place of text embeddings. DCS = separate cls / seg weights; DHF = separate weights per layer.
    ŷ = Softmax(⟨W_cls, x⟩ / τ), Ŷ = Softmax(⟨W_seg, F⟩ / τ), with ⟨·,·⟩ the cosine similarity.
    zh: 唯一可学习的部分：每层一组 W_cls^l 与 W_seg^l，形状都是 d × 2（正常、异常），用来替代文本嵌入。
    DCS 指分类与分割权重分开；DHF 指每层权重分开。⟨·,·⟩ 为余弦相似度。"""

    def __init__(self, dim):
        super().__init__()
        self.w_cls = nn.Parameter(torch.randn(dim, 2) * 0.02)
        self.w_seg = nn.Parameter(torch.randn(dim, 2) * 0.02)

    def forward(self, global_token, patch_tokens):
        w_cls = F.normalize(self.w_cls, dim=0)
        w_seg = F.normalize(self.w_seg, dim=0)
        cls_logits = F.normalize(global_token, dim=-1) @ w_cls / TAU     # [B, 2]
        seg_logits = F.normalize(patch_tokens, dim=-1) @ w_seg / TAU     # [B, N, 2]
        cls_prob = F.softmax(cls_logits, dim=-1)[:, 1]                   # P(anomaly) · 异常概率
        seg_prob = F.softmax(seg_logits, dim=-1)[:, :, 1]
        return cls_prob, seg_prob


def to_map(patch_scores):
    side = int(patch_scores.shape[1] ** 0.5)
    grid = patch_scores.reshape(patch_scores.shape[0], 1, side, side)
    return F.interpolate(grid, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False).squeeze(1)


class ZeroShotHead(nn.Module):
    """Zero-shot branch: one DecoupledHead per selected layer; scores and maps are averaged over layers.
    zh: 零样本分支：每个选中层一个 DecoupledHead，图像分数和异常图在各层间取平均。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoupledHead(WIDTH) for _ in FEATURES_LIST])

    def forward(self, global_tokens, patch_tokens):
        cls_scores = []
        seg_maps = []
        for i in range(len(FEATURES_LIST)):
            cls_prob, seg_prob = self.layers[i](global_tokens[i], patch_tokens[i])
            cls_scores.append(cls_prob)
            seg_maps.append(to_map(seg_prob))
        return torch.stack(cls_scores).mean(dim=0), torch.stack(seg_maps).mean(dim=0)    # ŷ_z, Ŷ_z


class NormalMemory(nn.Module):
    """Few-shot branch (no learnable weights): the K prompt images' patches form a memory M^l per layer;
    a query patch scores 1 − its max cosine similarity to M^l (Eq. 9), averaged over layers (Eq. 10).
    zh: 少样本分支（没有可学习参数）：K 张正常提示图的 patch 组成每层的记忆库 M^l；
    query patch 的异常分 = 1 − 与 M^l 的最大余弦相似度（式 9），再在各层间平均（式 10）。"""

    def forward(self, query_patches, prompt_patches):
        maps = []
        for i in range(len(FEATURES_LIST)):
            memory = F.normalize(prompt_patches[i].reshape(-1, WIDTH), dim=-1)                # [K·N, C]
            sim = F.normalize(query_patches[i], dim=-1) @ memory.transpose(0, 1)              # [B, N, K·N]
            maps.append(to_map(1 - torch.max(sim, dim=-1)[0]))
        return torch.stack(maps).mean(dim=0)


class UniADet(nn.Module):
    """UniADet (Gao & Wang, 2026): one frozen foundation model, no language. Zero-shot uses only the
    decoupled per-layer weights (4·d per layer); few-shot adds a nearest-neighbour normal memory and
    fuses both maps (Eq. 11) and the image score with the map peak (Eq. 12).
    Training (not drawn): cross-entropy for W_cls, focal + dice for W_seg, on auxiliary data.
    zh: UniADet（Gao 与 Wang，2026）：一个冻结的基础模型，不用语言。零样本只靠逐层解耦的权重（每层 4·d 个参数）；
    少样本再加最近邻的正常记忆库，融合两张异常图（式 11），图像分数再与异常图峰值融合（式 12）。
    训练（未画出）：W_cls 用交叉熵，W_seg 用 focal + dice，在辅助数据集上训练。"""

    def __init__(self):
        super().__init__()
        self.backbone = FoundationModel()
        self.zero_shot = ZeroShotHead()
        self.memory = NormalMemory()

    def forward(self, query, prompt):
        q_global, q_patches = self.backbone(query)
        _, p_patches = self.backbone(prompt)                 # same frozen model · 同一个冻结模型
        y_z, Y_z = self.zero_shot(q_global, q_patches)
        Y_m = self.memory(q_patches, p_patches)
        Y_f = (1 - LAMBDA_F) * Y_z + LAMBDA_F * Y_m          # Eq. 11
        peak = torch.max(Y_f.reshape(Y_f.shape[0], -1), dim=-1)[0]
        y_f = (1 - LAMBDA_P) * y_z + LAMBDA_P * peak         # Eq. 12
        return Y_f, y_f


if __name__ == "__main__":
    model = UniADet()
    query = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)          # image to inspect · 待检测图像
    prompt = torch.randn(SHOTS, 3, IMAGE_SIZE, IMAGE_SIZE)     # K normal images · K 张正常图
    anomaly_map, score = model(query, prompt)
