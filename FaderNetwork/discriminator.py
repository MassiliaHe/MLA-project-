import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_attributes):
        super(Discriminator, self).__init__()
        self.n_attributes = n_attributes
        # Assuming the flattened latent representation has a size of 2048 (e.g., 512 * 2 * 2)
        self.discriminator = nn.Sequential(
            nn.Linear(2048, 1024),  # Adjust the input features to match the flattened size.
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, self.n_attributes),
            nn.Sigmoid()
        )

    def forward(self, z):
        # Flatten the latent representation if it's not already flat.
        if z.dim() > 2:
            z = z.view(z.size(0), -1)
        return self.discriminator(z)
