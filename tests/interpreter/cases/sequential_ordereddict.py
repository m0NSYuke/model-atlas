import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(OrderedDict([("fc1", nn.Linear(10, 20)), ("act", nn.ReLU()), ("fc2", nn.Linear(20, 5))]))
        self.head = nn.Linear(5, 5)
        self.head.weight = self.body.fc2.weight if False else self.head.weight
    def forward(self, x):
        return self.head(self.body(x))
if __name__ == "__main__":
    print(Net()(torch.randn(2, 10)).shape)
