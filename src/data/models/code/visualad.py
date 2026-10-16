import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from the official code, github.com/7HHHHH/VisualAD (VisualAD_lib/VisualAD.py, utils/, train.py).
# zh: 改写自官方代码 github.com/7HHHHH/VisualAD（VisualAD_lib/VisualAD.py、utils/、train.py）。
# VisualAD at toy scale: the paper freezes CLIP ViT-L/14@336px (24 layers, width 1024) at 518×518
# and reads layers 6/12/18/24. Here: 8 layers, width 256, layers 2/4/6/8.
# zh: 缩小版 VisualAD：论文冻结 CLIP ViT-L/14@336px（24 层、宽 1024），输入 518×518，取第 6/12/18/24 层。
# 这里用 8 层、宽 256，取第 2/4/6/8 层。
IMAGE_SIZE = 112
PATCH = 14
WIDTH = 256
LAYERS = 8
HEADS = 4                       # vision_width // 64 · 头数 = 宽度 // 64
FEATURES_LIST = [2, 4, 6, 8]    # layers whose tokens are read out · 读出 token 的层
NUM_ANCHORS = 4
TOPK_RATIO = 0.01


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Frozen CLIP Transformer block (sequence-first MultiheadAttention, QuickGELU MLP).
    zh: 冻结的 CLIP Transformer 块（序列维在前的多头注意力 + QuickGELU MLP）。"""

    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), QuickGELU(), nn.Linear(d_model * 4, d_model))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionTransformer(nn.Module):
    """CLIP visual encoder with two extra learnable tokens, [anomaly] and [normal], placed before [CLS].
    They attend to the patches in every frozen layer, so they end up encoding abnormality and normality.
    zh: 在 [CLS] 前插入两个可学习 token：[anomaly] 与 [normal]。它们在每一层冻结的自注意力里与 patch 交互，
    从而学到“异常”和“正常”的视觉原型，不需要文本编码器。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, WIDTH, kernel_size=PATCH, stride=PATCH, bias=False)
        scale = WIDTH ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(WIDTH))
        self.anomaly_token = nn.Parameter(scale * torch.randn(WIDTH) * 0.1)   # learnable · 可学习
        self.normal_token = nn.Parameter(scale * torch.randn(WIDTH) * 0.1)    # learnable · 可学习
        num_patches = (IMAGE_SIZE // PATCH) ** 2
        self.positional_embedding_frozen = nn.Parameter(scale * torch.randn(num_patches + 1, WIDTH))  # [CLS] + patches, from CLIP
        self.anomaly_pos = nn.Parameter(scale * torch.randn(1, WIDTH))
        self.normal_pos = nn.Parameter(scale * torch.randn(1, WIDTH))
        self.ln_pre = nn.LayerNorm(WIDTH)
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(WIDTH, HEADS) for _ in range(LAYERS)])
        self.ln_post = nn.LayerNorm(WIDTH)    # fine-tuned · 参与微调

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)       # [B, N, C]
        b = x.shape[0]
        anomaly = self.anomaly_token.unsqueeze(0).expand(b, -1, -1)
        normal = self.normal_token.unsqueeze(0).expand(b, -1, -1)
        cls = self.class_embedding.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([anomaly, normal, cls, x], dim=1)                   # [B, 3 + N, C]
        pos = torch.cat([self.anomaly_pos, self.normal_pos, self.positional_embedding_frozen], 0)
        x = self.ln_pre(x + pos)
        x = x.permute(1, 0, 2)                                            # sequence-first · 序列维在前
        patch_tokens = []
        idx = 0
        for block in self.resblocks:
            idx += 1
            x = block(x)
            if idx in FEATURES_LIST:
                patch_tokens.append(self.ln_post(x.permute(1, 0, 2)))
        x = x.permute(1, 0, 2)
        anomaly_features = self.ln_post(x[:, 0, :])
        normal_features = self.ln_post(x[:, 1, :])
        return anomaly_features, normal_features, patch_tokens


class SpatialBottleneckAttention(nn.Module):
    """SCA: a few learnable anchor queries cross-attend to one layer's patches (keys carry 2D positions);
    a gate computed from the token weights the anchors, and the result is added back with a small residual scale.
    zh: SCA 空间感知交叉注意力：少量可学习锚点 query 去看某一层的 patch（key 带位置编码），
    由 token 自己算出的门控给各锚点加权，再以很小的残差系数加回 token。"""

    def __init__(self, embed_dim, num_anchors, max_patches=1536, res_scale_init=0.01):
        super().__init__()
        self.anchor_queries = nn.Parameter(torch.randn(num_anchors, embed_dim) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim, num_anchors), nn.Sigmoid())
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.res_scale = nn.Parameter(torch.ones(1) * res_scale_init)

    def forward(self, token_features, patch_tokens):
        B, N, C = patch_tokens.shape
        anchor_q = self.anchor_queries.unsqueeze(0).expand(B, -1, -1)            # [B, A, C]
        Q = self.q_norm(self.q_proj(anchor_q))
        K = self.k_norm(self.k_proj(patch_tokens + self.pos_encoding[:, :N, :]))  # [B, N, C]
        V = self.v_proj(patch_tokens)
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * C ** -0.5, dim=-1)     # [B, A, N]
        aggregated = torch.bmm(self.dropout(attn), V)                            # [B, A, C]
        gate_weights = self.gate(token_features).unsqueeze(-1)                   # [B, A, 1]
        gated_agg = (aggregated * gate_weights).sum(dim=1)                       # [B, C]
        output = self.dropout(self.out_proj(gated_agg))
        return token_features + self.res_scale * output


class LayerAdaptiveCrossAttention(nn.Module):
    """One independent SCA module per readout layer (low layers see texture, high layers see semantics).
    zh: 每个读出层一个独立的 SCA 模块（低层偏纹理、高层偏语义，所以参数不共享）。"""

    def __init__(self):
        super().__init__()
        self.layer_attentions = nn.ModuleList([SpatialBottleneckAttention(WIDTH, NUM_ANCHORS) for _ in FEATURES_LIST])

    def forward(self, anomaly_token, normal_token, patch_tokens_list):
        anomaly_list = []
        normal_list = []
        for i in range(len(FEATURES_LIST)):
            anomaly_list.append(self.layer_attentions[i](anomaly_token, patch_tokens_list[i]))
            normal_list.append(self.layer_attentions[i](normal_token, patch_tokens_list[i]))
        return anomaly_list, normal_list


class FeatureTransformMLP(nn.Module):
    """SAF: per-layer MLP that re-aligns patch features before scoring them against the tokens.
    zh: SAF 自对齐函数：每层一个 MLP，在与 token 比较前重新校准 patch 特征。"""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, dim), nn.Dropout(0.1))

    def forward(self, x):
        return self.mlp(x)


def generate_anomaly_map_from_tokens(anomaly_features, normal_features, patch_tokens, image_size):
    """Patch score = cos(patch, anomaly) − cos(patch, normal), reshaped to a grid and upsampled.
    zh: patch 异常分 = 与 anomaly token 的余弦相似度 − 与 normal token 的余弦相似度，再还原成网格并上采样。"""
    B = anomaly_features.shape[0]
    anomaly_sim = torch.cosine_similarity(F.normalize(patch_tokens, dim=2), F.normalize(anomaly_features, dim=1).unsqueeze(1), dim=2)
    normal_sim = torch.cosine_similarity(F.normalize(patch_tokens, dim=2), F.normalize(normal_features, dim=1).unsqueeze(1), dim=2)
    anomaly_score = anomaly_sim - normal_sim                       # [B, N]
    side = int(math.sqrt(anomaly_score.shape[1]))
    anomaly_map = anomaly_score.reshape(B, side, side)
    return F.interpolate(anomaly_map.unsqueeze(1), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(1)


class LayerAnomalyMaps(nn.Module):
    """SAF + scoring for every readout layer: align the patches with that layer's MLP, then score them
    against that layer's refined tokens.
    zh: 对每个读出层做 SAF + 打分：先用该层 MLP 校准 patch，再与该层增强后的 token 比较。"""

    def __init__(self):
        super().__init__()
        self.layer_transforms = nn.ModuleList([FeatureTransformMLP(WIDTH, WIDTH) for _ in FEATURES_LIST])

    def forward(self, anomaly_list, normal_list, patch_tokens):
        maps = []
        for i in range(len(FEATURES_LIST)):
            patch_feature = self.layer_transforms[i](patch_tokens[i])            # all 3 + N tokens · 含特殊 token
            maps.append(generate_anomaly_map_from_tokens(anomaly_list[i], normal_list[i], patch_feature[:, 3:, :], IMAGE_SIZE))
        return maps


class FocalLoss(nn.Module):
    """Focal loss on the two-channel map [1 − p, p] · 在 [1 − p, p] 两通道图上的 focal 损失。"""

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, prob, target):
        pt = prob * target + (1 - prob) * (1 - target)
        return (-(1 - pt) ** self.gamma * torch.log(pt + 1e-5)).mean()


class BinaryDiceLoss(nn.Module):
    """Dice loss on the anomaly channel · 只在异常通道上算 Dice 损失。"""

    def forward(self, prob, target):
        inter = (prob * target).sum(dim=(1, 2))
        return (1 - (2 * inter + 1) / (prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)).mean()


class SegmentationLoss(nn.Module):
    """compute_segmentation_loss in the official code: focal + dice on every layer's map.
    zh: 对应官方代码的 compute_segmentation_loss：每层异常图各算一次 focal + dice。"""

    def __init__(self):
        super().__init__()
        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

    def forward(self, maps, gt):
        loss = 0
        for anomaly_map in maps:
            prob = torch.sigmoid(anomaly_map)
            loss = loss + self.loss_focal(prob, gt) + self.loss_dice(prob, gt)
        return loss


class VisualAD(nn.Module):
    """VisualAD (CVPR 2026): language-free zero-shot anomaly detection. A frozen ViT carries two learnable
    tokens; SCA refines them per layer, SAF re-aligns patches, and the anomaly map is the per-layer sum of
    cos(patch, anomaly) − cos(patch, normal). The image score is the top-k mean of that map.
    One training step as in train.py: BCE on the image score, focal + dice on every layer's map,
    and a margin loss that pushes the two tokens apart.
    zh: VisualAD（CVPR 2026）：不依赖语言的零样本异常检测。冻结 ViT 中插入两个可学习 token；SCA 逐层增强 token，
    SAF 校准 patch，异常图是各层 cos(patch, anomaly) − cos(patch, normal) 的和，图像分数取异常图 top-k 均值。
    这里画的是 train.py 的一个训练步：图像分数的 BCE、每层异常图的 focal + dice、把两个 token 推开的对比损失。"""

    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer()
        self.cross_attn = LayerAdaptiveCrossAttention()
        self.anomaly_maps = LayerAnomalyMaps()
        self.seg_loss = SegmentationLoss()

    def forward(self, image, label, gt):
        anomaly_features, normal_features, patch_tokens = self.visual(image)
        patch_features_list = [pt[:, 3:, :] for pt in patch_tokens]        # drop [anomaly] [normal] [CLS] · 去掉三个特殊 token
        anomaly_list, normal_list = self.cross_attn(anomaly_features, normal_features, patch_features_list)
        maps = self.anomaly_maps(anomaly_list, normal_list, patch_tokens)
        seg_loss = self.seg_loss(maps, gt)                             # each layer supervised · 每层单独监督
        final_map = torch.stack(maps).sum(dim=0)                       # [B, H, W]
        flat = final_map.reshape(final_map.shape[0], -1)
        k = max(1, int(math.ceil(flat.shape[-1] * TOPK_RATIO)))
        score = torch.topk(flat, k, dim=-1)[0].mean(dim=-1)            # top-k mean → image logit · 图像级 logit
        image_loss = F.binary_cross_entropy_with_logits(score, label)
        # Contrastive margin on the last layer's tokens · 用最后一层的 token 做对比损失
        a = F.normalize(anomaly_list[-1], dim=1)
        n = F.normalize(normal_list[-1], dim=1)
        contrast = torch.clamp(F.cosine_similarity(a, n, dim=1) + 0.5, min=0).mean()
        return image_loss + seg_loss + contrast


if __name__ == "__main__":
    model = VisualAD()
    image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    label = torch.tensor([0.0, 1.0])                               # normal / anomalous · 正常 / 异常
    gt = (torch.rand(2, IMAGE_SIZE, IMAGE_SIZE) > 0.9).float()     # pixel mask · 像素级掩码
    loss = model(image, label, gt)
