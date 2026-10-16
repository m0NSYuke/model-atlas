import torch
import torch.nn as nn

LATENT_DIM = 32
CHANNELS = [32, 64, 128]


class ConvVAE(nn.Module):
    """A convolutional variational autoencoder for 64×64 images.
    zh: 面向 64×64 图像的卷积变分自编码器。"""

    def __init__(self, channels=CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        enc, c_in = [], 3
        for c in channels:
            enc += [nn.Conv2d(c_in, c, 4, stride=2, padding=1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2)]
            c_in = c
        self.encoder = nn.Sequential(*enc)
        self.side = 64 // 2 ** len(channels)
        flat = channels[-1] * self.side * self.side
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat)
        dec = []
        rev = list(reversed(channels))
        for c, c_out in zip(rev, rev[1:] + [3]):
            dec += [nn.ConvTranspose2d(c, c_out, 4, stride=2, padding=1), nn.ReLU() if c_out != 3 else nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)          # ε ~ N(0, I)
        return mu + eps * std                # z = μ + σ·ε

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        h = self.fc_dec(z).view(-1, CHANNELS[-1], self.side, self.side)
        return self.decoder(h), mu, logvar


if __name__ == "__main__":
    model = ConvVAE()
    x = torch.randn(1, 3, 64, 64)
    recon, mu, logvar = model(x)
