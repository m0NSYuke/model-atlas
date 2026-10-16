# train.py (optional · 可选): loss, optimizer and the training loop. Atlas draws ONE training step:
#   batch → model → loss → backward → optimizer.step
# Every loop around it (epochs, batches) runs once; the step block shows how many times it would repeat.
# zh: 损失、优化器和训练循环。Atlas 只画“一个训练步”，外面的 epoch / batch 循环只跑一次，
#     “更新参数”模块会写明真实训练要重复多少步。
import torch
import torch.nn as nn
from model import SmallCNN
from data import make_loader

EPOCHS = 20
LR = 3e-4

model = SmallCNN()
loader = make_loader()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)                 # forward · 前向
        loss = criterion(logits, labels)       # loss · 损失
        loss.backward()                        # gradients · 反向传播求梯度
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                       # update weights · 更新参数
    scheduler.step()
