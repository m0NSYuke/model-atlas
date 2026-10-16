import torch
import torch.nn as nn

# ── Hyper-parameters · 超参数：修改后 3D 结构会自动刷新 ──
IMAGE_SIZE = 28
HIDDEN = [512, 256]
NUM_CLASSES = 10
DROPOUT = 0.1


class MLP(nn.Module):
    """A multilayer perceptron that classifies 28×28 MNIST digits.
    zh: 对 28×28 MNIST 手写数字分类的多层感知机。"""

    def __init__(self, in_features, hidden, num_classes, dropout):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        for width in hidden:
            layers += [nn.Linear(in_features, width), nn.ReLU(), nn.Dropout(dropout)]
            in_features = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.flatten(x)      # [B, 1, 28, 28] → [B, 784]
        x = self.body(x)
        return self.head(x)      # class logits


if __name__ == "__main__":
    model = MLP(IMAGE_SIZE * IMAGE_SIZE, HIDDEN, NUM_CLASSES, DROPOUT)
    x = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    logits = model(x)
    print("parameters:", sum(p.numel() for p in model.parameters()))
