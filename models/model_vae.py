import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, num_classes=2, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # ===== ENCODER =====
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + num_classes, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(32 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(32 * 16 * 16, latent_dim)

        # ===== DECODER =====
        self.fc_decode = nn.Linear(latent_dim + num_classes, 32 * 16 * 16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, labels):
        z = torch.cat([z, labels], dim=1)

        h = self.fc_decode(z)
        h = h.view(-1, 32, 16, 16)

        return self.decoder(h)

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    def sample(self, z, labels):
        return self.decode(z, labels)