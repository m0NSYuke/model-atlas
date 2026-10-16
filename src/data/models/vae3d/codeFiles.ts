export const codeFiles = {
  'model.py': `import torch
import torch.nn as nn
from blocks import Conv3DBlock, DownsampleCausal3D


class VideoVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channels = 3

        self.encoder = nn.Sequential(
            Conv3DBlock(3, 128, stride=(1, 2, 2)),
            DownsampleCausal3D(
                channels=128, out_channels=256,
                # kernel_size and stride use the block defaults
            ),
            nn.SiLU(),  # mid block
        )
        hidden = self.encoder[-2].conv.out_channels
        self.moments = nn.Conv3d(hidden, hidden * 2, 1)
        self.mu_proj = nn.Identity()
        self.logvar_proj = nn.Identity()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=(2, 4, 4), mode="nearest"),
            nn.Conv3d(hidden, 3, kernel_size=3, padding=1),
        )

    def encode(self, video):
        features = self.encoder(video)
        mu, logvar = self.moments(features).chunk(2, dim=1)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return latent

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, video):
        return self.decode(self.encode(video))
`,
  'blocks.py': `import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    def __init__(self, channels, out_channels, stride=(1, 2, 2)):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, out_channels, kernel_size=3,
            stride=stride, padding=1
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x))


class DownsampleCausal3D(nn.Module):
    """Reduce T/H/W while padding only the past in time."""

    def __init__(self, channels, out_channels,
                 kernel_size=3, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv3d(
            channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 1, 1)
        )

    def forward(self, x):
        # Past-only temporal padding preserves causality.
        temporal_kernel = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
        x = F.pad(x, (0, 0, 0, 0, temporal_kernel - 1, 0))
        return self.conv(x)
`,
} as const
