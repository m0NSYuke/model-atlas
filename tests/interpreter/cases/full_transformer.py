import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    m = nn.Transformer(d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64, batch_first=True)
    print(m(torch.randn(2, 10, 32), torch.randn(2, 6, 32)).shape)
