import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(784, 400); self.mu = nn.Linear(400, 20); self.lv = nn.Linear(400, 20); self.dec = nn.Sequential(nn.Linear(20, 400), nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid())
    def encode(self, x):
        h = F.relu(self.enc(x))
        return self.mu(h), self.lv(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar
def loss_fn(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon, x.view(-1, 784), reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld
if __name__ == "__main__":
    m = VAE(); x = torch.rand(8, 1, 28, 28)
    r, mu, lv = m(x)
    print(loss_fn(r, x, mu, lv).shape)
