import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_attributes):
        super(Discriminator, self).__init__()
        # Following the paper, we assume the latent code has a size of 512
        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, n_attributes),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)