import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.dec = nn.TransformerDecoderLayer(64, 4, 128, batch_first=True)
    def forward(self, src, tgt):
        mem = self.enc(src)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
        return self.dec(tgt, mem, tgt_mask=mask)
if __name__ == "__main__":
    print(Net()(torch.randn(2, 12, 64), torch.randn(2, 7, 64)).shape)
