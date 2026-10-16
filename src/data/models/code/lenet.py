import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """LeNet-5 (LeCun et al., 1998): two convolution stages followed by three fully connected layers.
    zh: LeNet-5（LeCun 等，1998）：两级卷积加三层全连接，是 CNN 的经典起点。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),   # C1: 28×28 → 28×28
            nn.Tanh(),
            nn.AvgPool2d(2),                             # S2: → 14×14
            nn.Conv2d(6, 16, kernel_size=5),             # C3: → 10×10
            nn.Tanh(),
            nn.AvgPool2d(2),                             # S4: → 5×5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),                  # C5
            nn.Tanh(),
            nn.Linear(120, 84),                          # F6
            nn.Tanh(),
            nn.Linear(84, num_classes),                  # output
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = LeNet5()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
