import torch
import torch.nn as nn

class AutoEncoder(nn.Module):

    def __init__(self, n_attributes):
        super(AutoEncoder, self).__init__()
        self.encode = Encoder(n_attributes)
        self.decode = Decoder()

    def forward(self, x, y):
        enc_outputs = self.encode(x)
        dec_outputs = self.decode(enc_outputs, y)
        return enc_outputs, dec_outputs

class Decoder(nn.Module):
    def __init__(self, n_attributes):
        """
        Initialize the CNN decoder for Fader Networks.

        :param output_channels: Number of channels in the output image (e.g., 3 for RGB).
        :param n_attributes: Number of attributes used in the latent code.
        """
        super(Decoder, self).__init__()
        self.n_attributes = n_attributes

        # Calculate the additional channels for the attribute codes
        additional_channels =  2*n_attributes

        # Define the transposed convolutional layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512 + additional_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512 + additional_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256 + additional_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128 + additional_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64 + additional_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32 + additional_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16 + additional_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), 
        )

    def forward(self, x, attributes):
        """
        Propager l'entrée à travers le décodeur.

        :param x: Représentation latente.
        :param attributes: Codes d'attributs pour l'image.
        :return: Image reconstruite.
        """
        # Redimensionner les attributs pour correspondre à la taille du batch de x
        if attributes.dim() == 1:
            attributes = attributes.unsqueeze(0).repeat(x.size(0), 1)

        attributes = attributes.view(x.size(0), -1, 1, 1)

        for layer in self.deconv_layers:
            if isinstance(layer, nn.ConvTranspose2d):
                # Étendre les attributs pour chaque couche de déconvolution
                expanded_attributes = attributes.expand(-1, -1, x.size(2), x.size(3))
                x = torch.cat([x, expanded_attributes], dim=1)

            x = layer(x)

        return x
    
    
class Encoder(nn.Module):
    def __init__(self):
        """
        Initialiser l'encodeur CNN pour Fader Networks.

        :param input_channels: Nombre de canaux de l'image d'entrée (par exemple, 3 pour RGB).
        :param latent_dim: Dimension de l'espace latent.
        """
        super(Encoder, self).__init__()

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