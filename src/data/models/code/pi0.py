import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# π0 at toy scale: the real model pairs a 3B PaliGemma with a 300M action expert.
# zh: 缩小版 π0：真实模型是 30 亿参数的 PaliGemma 加 3 亿参数的动作专家。
IMAGE_SIZE = 64
PATCH = 8
WIDTH = 256          # VLM (prefix) width · VLM 前缀宽度
EXPERT = 128         # action expert width · 动作专家宽度
ATTN = 128           # shared attention width · 共享注意力宽度
DEPTH = 3            # joint Transformer layers · 联合 Transformer 层数
VOCAB = 1000
TEXT_TOKENS = 12
STATE_DIM = 14
ACTION_DIM = 14
HORIZON = 50         # actions predicted per chunk · 每块预测的动作数


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
    args = t[:, None].float() * 1000 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class VisionEncoder(nn.Module):
    """SigLIP-style ViT: image patches → tokens, projected to the language model width.
    zh: SigLIP 式视觉编码器：图像切块成 token，再投影到语言模型宽度。"""

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 192, PATCH, stride=PATCH)
        self.position = nn.Parameter(torch.zeros(1, (IMAGE_SIZE // PATCH) ** 2, 192))
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(192, 3, 768, batch_first=True, norm_first=True) for _ in range(2)])
        self.project = nn.Linear(192, WIDTH)

    def forward(self, image):
        x = self.patch(image).flatten(2).transpose(1, 2) + self.position
        return self.project(self.blocks(x))


class JointLayer(nn.Module):
    """One layer of the shared Transformer. Prefix tokens use the VLM weights and attend only to the
    prefix; state and action tokens use the action expert's weights and attend to everything.
    zh: 共享 Transformer 的一层。前缀 token 用 VLM 权重、只看前缀；状态和动作 token 用动作专家的权重、可以看全部 token。"""

    def __init__(self):
        super().__init__()
        self.vlm_norm1 = nn.RMSNorm(WIDTH)
        self.vlm_qkv = nn.Linear(WIDTH, 3 * ATTN)
        self.vlm_out = nn.Linear(ATTN, WIDTH)
        self.vlm_norm2 = nn.RMSNorm(WIDTH)
        self.vlm_mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))
        self.expert_norm1 = nn.RMSNorm(EXPERT)
        self.expert_qkv = nn.Linear(EXPERT, 3 * ATTN)
        self.expert_out = nn.Linear(ATTN, EXPERT)
        self.expert_norm2 = nn.RMSNorm(EXPERT)
        self.expert_mlp = nn.Sequential(nn.Linear(EXPERT, 4 * EXPERT), nn.GELU(), nn.Linear(4 * EXPERT, EXPERT))

    def forward(self, prefix, suffix):
        pq, pk, pv = self.vlm_qkv(self.vlm_norm1(prefix)).chunk(3, dim=-1)
        sq, sk, sv = self.expert_qkv(self.expert_norm1(suffix)).chunk(3, dim=-1)
        # Blockwise-causal attention: the prefix never reads the actions · 分块因果：前缀看不到动作
        prefix_attn = F.scaled_dot_product_attention(pq, pk, pv)
        suffix_attn = F.scaled_dot_product_attention(sq, torch.cat([pk, sk], dim=1), torch.cat([pv, sv], dim=1))
        prefix = prefix + self.vlm_out(prefix_attn)
        prefix = prefix + self.vlm_mlp(self.vlm_norm2(prefix))
        suffix = suffix + self.expert_out(suffix_attn)
        suffix = suffix + self.expert_mlp(self.expert_norm2(suffix))
        return prefix, suffix


class Pi0(nn.Module):
    """π0 (Black et al., 2024): a vision-language model plus an action expert trained with flow matching
    to output a chunk of continuous actions.
    zh: π0（Black 等，2024）：视觉语言模型加动作专家，用流匹配训练，一次输出一段连续动作。"""

    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.text_embed = nn.Embedding(VOCAB, WIDTH)
        self.state_proj = nn.Linear(STATE_DIM, EXPERT)
        self.action_in = nn.Linear(ACTION_DIM, EXPERT)
        self.action_time_mlp = nn.Sequential(nn.Linear(2 * EXPERT, EXPERT), nn.SiLU(), nn.Linear(EXPERT, EXPERT))
        self.layers = nn.ModuleList([JointLayer() for _ in range(DEPTH)])
        self.action_out = nn.Linear(EXPERT, ACTION_DIM)

    def forward(self, image, ids, state, actions, noise, tau):
        # Prefix: image and language tokens · 前缀：图像与语言 token
        prefix = torch.cat([self.vision(image), self.text_embed(ids)], dim=1)
        # Flow matching: x_τ = τ·noise + (1−τ)·actions · 流匹配插值
        x_tau = tau[:, None, None] * noise + (1 - tau[:, None, None]) * actions
        time = time_embedding(tau, EXPERT)[:, None, :].expand(-1, HORIZON, -1)
        action_tokens = self.action_time_mlp(torch.cat([self.action_in(x_tau), time], dim=-1))
        suffix = torch.cat([self.state_proj(state)[:, None, :], action_tokens], dim=1)
        for layer in self.layers:
            prefix, suffix = layer(prefix, suffix)
        velocity = self.action_out(suffix[:, 1:])
        return F.mse_loss(velocity, noise - actions)           # target u = noise − actions · 目标速度场


if __name__ == "__main__":
    model = Pi0()
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    ids = torch.randint(0, VOCAB, (1, TEXT_TOKENS))
    state = torch.randn(1, STATE_DIM)
    actions = torch.randn(1, HORIZON, ACTION_DIM)
    noise = torch.randn_like(actions)
    tau = torch.rand(1)
    loss = model(image, ids, state, actions, noise, tau)
