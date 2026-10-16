import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(16, 32, num_layers=2, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(64, 32, batch_first=True)
        self.rnn = nn.RNN(32, 8)
    def forward(self, x):
        y, (h, c) = self.lstm(x)
        y, h2 = self.gru(y)
        z, _ = self.rnn(y.transpose(0, 1))
        return z[-1], h[-2:].transpose(0, 1).reshape(x.size(0), -1)
if __name__ == "__main__":
    print([o.shape for o in Net()(torch.randn(3, 7, 16))])
