import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# π0.5 at toy scale · 缩小版 π0.5
IMAGE_SIZE = 64
PATCH = 8
WIDTH = 256          # VLM width · VLM 宽度
EXPERT = 128         # action expert width · 动作专家宽度
ATTN = 128
DEPTH = 3
VOCAB = 1000
PROMPT_TOKENS = 12   # high-level task prompt · 高层任务指令
STATE_DIM = 14
STATE_BINS = 256     # state is discretized into text tokens · 状态离散化成文本 token
SUBTASK_TOKENS = 6   # predicted subtask, e.g. "pick up the cup" · 预测的子任务文本
ACTION_DIM = 14
HORIZON = 50


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
    args = t[:, None].float() * 1000 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class VisionEncoder(nn.Module):
    """SigLIP-style ViT: image patches → tokens in the language model width.
    zh: SigLIP 式视觉编码器：图像块 → 语言模型宽度的 token。"""

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 192, PATCH, stride=PATCH)
        self.position = nn.Parameter(torch.zeros(1, (IMAGE_SIZE // PATCH) ** 2, 192))
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(192, 3, 768, batch_first=True, norm_first=True) for _ in range(2)])
        self.project = nn.Linear(192, WIDTH)

    def forward(self, image):
        x = self.patch(image).flatten(2).transpose(1, 2) + self.position
        return self.project(self.blocks(x))


class AdaRMSNorm(nn.Module):
    """RMSNorm whose scale, shift and gate come from the flow time (π0.5 conditions the expert this way).
    zh: 缩放、平移和门控都由流时间生成的 RMSNorm（π0.5 用它给动作专家注入时间）。"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.modulation = nn.Linear(EXPERT, 3 * dim)

    def forward(self, x, cond):
        scale, shift, gate = self.modulation(cond)[:, None, :].chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


class JointLayer(nn.Module):
    """Shared Transformer layer: VLM weights for the prefix, action-expert weights (with AdaRMSNorm) for
    the noisy actions; the actions attend to the whole prefix.
    zh: 共享 Transformer 层：前缀用 VLM 权重，带噪动作用动作专家权重（含 AdaRMSNorm）；动作可以读取整个前缀。"""

    def __init__(self):
        super().__init__()
        self.vlm_norm1 = nn.RMSNorm(WIDTH)
        self.vlm_qkv = nn.Linear(WIDTH, 3 * ATTN)
        self.vlm_out = nn.Linear(ATTN, WIDTH)
        self.vlm_norm2 = nn.RMSNorm(WIDTH)
        self.vlm_mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))
        self.expert_norm1 = AdaRMSNorm(EXPERT)
        self.expert_qkv = nn.Linear(EXPERT, 3 * ATTN)
        self.expert_out = nn.Linear(ATTN, EXPERT)
        self.expert_norm2 = AdaRMSNorm(EXPERT)
        self.expert_mlp = nn.Sequential(nn.Linear(EXPERT, 4 * EXPERT), nn.GELU(), nn.Linear(4 * EXPERT, EXPERT))

    def forward(self, prefix, suffix, cond):
        pq, pk, pv = self.vlm_qkv(self.vlm_norm1(prefix)).chunk(3, dim=-1)
        h, gate = self.expert_norm1(suffix, cond)
        sq, sk, sv = self.expert_qkv(h).chunk(3, dim=-1)
        prefix_attn = F.scaled_dot_product_attention(pq, pk, pv)
        suffix_attn = F.scaled_dot_product_attention(sq, torch.cat([pk, sk], dim=1), torch.cat([pv, sv], dim=1))
        prefix = prefix + self.vlm_out(prefix_attn)
        prefix = prefix + self.vlm_mlp(self.vlm_norm2(prefix))
        suffix = suffix + gate * self.expert_out(suffix_attn)
        h, gate = self.expert_norm2(suffix, cond)
        suffix = suffix + gate * self.expert_mlp(h)
        return prefix, suffix


class Pi05(nn.Module):
    """π0.5 (Physical Intelligence, 2025): one VLM co-trained on robot, web and semantic data. It predicts a
    high-level subtask as text, and an action expert turns the context into a continuous action chunk.
    zh: π0.5（Physical Intelligence，2025）：在机器人、网页和语义数据上协同训练的 VLM。它先用文本预测高层子任务，
    再由动作专家把上下文变成一段连续动作。"""

    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.text_embed = nn.Embedding(VOCAB + STATE_BINS, WIDTH)   # words and state bins share one table · 词表与状态桶共用
        self.layers = nn.ModuleList([JointLayer() for _ in range(DEPTH)])
        self.lm_head = nn.Linear(WIDTH, VOCAB)                      # subtask text · 子任务文本
        self.time_mlp = nn.Sequential(nn.Linear(EXPERT, EXPERT), nn.SiLU(), nn.Linear(EXPERT, EXPERT))
        self.action_in = nn.Linear(ACTION_DIM, EXPERT)
        self.action_out = nn.Linear(EXPERT, ACTION_DIM)

    def forward(self, image, prompt, state, subtask, actions, noise, tau):
        # Robot state becomes discrete tokens in the prompt · 机器人状态离散成提示词里的 token
        state_ids = VOCAB + ((state.clamp(-1, 1) + 1) / 2 * (STATE_BINS - 1)).long()
        tokens = self.text_embed(torch.cat([prompt, state_ids, subtask], dim=1))
        prefix = torch.cat([self.vision(image), tokens], dim=1)
        # Flow matching on the action chunk; time enters through AdaRMSNorm · 动作块上的流匹配
        x_tau = tau[:, None, None] * noise + (1 - tau[:, None, None]) * actions
        cond = self.time_mlp(time_embedding(tau, EXPERT))
        suffix = self.action_in(x_tau)
        for layer in self.layers:
            prefix, suffix = layer(prefix, suffix, cond)
        # High level: next-token loss on the subtask text · 高层：子任务文本的下一 token 损失
        logits = self.lm_head(prefix[:, -SUBTASK_TOKENS - 1:-1])
        text_loss = F.cross_entropy(logits.reshape(-1, VOCAB), subtask.reshape(-1))
        # Low level: velocity of the action flow · 低层：动作流的速度场
        flow_loss = F.mse_loss(self.action_out(suffix), noise - actions)
        return text_loss + flow_loss


if __name__ == "__main__":
    model = Pi05()
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    prompt = torch.randint(0, VOCAB, (1, PROMPT_TOKENS))
    state = torch.randn(1, STATE_DIM)
    subtask = torch.randint(0, VOCAB, (1, SUBTASK_TOKENS))
    actions = torch.randn(1, HORIZON, ACTION_DIM)
    noise = torch.randn_like(actions)
    tau = torch.rand(1)
    loss = model(image, prompt, state, subtask, actions, noise, tau)
