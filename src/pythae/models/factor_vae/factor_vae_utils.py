import torch
import torch.nn as nn


class FactorVAEDiscriminator(nn.Module):
    def __init__(self, latent_dim=16, hidden_units=1000) -> None:

        nn.Module.__init__(self)

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, 2),
        )

    def forward(self, z: torch.Tensor):
        return self.layers(z)
