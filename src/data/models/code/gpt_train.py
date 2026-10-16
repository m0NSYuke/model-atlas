# Next-token training for the GPT in model.py, in the style of nanoGPT's train.py.
# zh: 用 nanoGPT 的方式训练 model.py 里的 GPT：预测下一个 token。
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig

BATCH_SIZE = 32
MAX_ITERS = 5000


class TokenChunks(Dataset):
    """Random windows of a long token stream; y is x shifted by one token.
    zh: 从长 token 序列里截取窗口；y 是 x 向后错一位。"""

    def __init__(self, n_tokens, block_size, vocab_size):
        self.n_tokens = n_tokens
        self.block_size = block_size
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n_tokens - self.block_size

    def __getitem__(self, i):
        x = torch.randint(0, self.vocab_size, (self.block_size,))
        y = torch.randint(0, self.vocab_size, (self.block_size,))
        return x, y


cfg = GPTConfig(n_layer=4, n_head=4, n_embd=128)
model = GPT(cfg)
loader = DataLoader(TokenChunks(1_000_000, 32, cfg.vocab_size), batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS)

for x, y in loader:
    logits = model(x)                                            # [B, T, vocab]
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
