import torch
import torchvision
import os 
import numpy as np
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import re
from torch.utils.data import Dataset,DataLoader, Subset,DataLoader, random_split
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn as nn
from torchvision import datasets, transforms




class FaderNetworksEncoder(nn.Module):
    def __init__(self,):
        """
        Initialiser l'encodeur CNN pour Fader Networks.

        :param input_channels: Nombre de canaux de l'image d'entrée (par exemple, 3 pour RGB).
        :param latent_dim: Dimension de l'espace latent.
        """
        super(FaderNetworksEncoder, self).__init__()

        # Définition des couches convolutives
        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            
           

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            


            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),


            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        """
        Propagation avant à travers l'encodeur.

        :param x: Tensor, image d'entrée.
        :return: Tensor, représentation dans l'espace latent.
        """
        x = self.conv_layers(x)
    
        return x


# Exemple d'utilisation

# Afficher le résumé du modèle
