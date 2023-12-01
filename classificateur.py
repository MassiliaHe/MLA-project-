
import torch
import torch.nn as nn

class FaderNetworksClassifier(nn.Module):
    def __init__(self, n_attributes=40):
        super(FaderNetworksClassifier, self).__init__()
        # n_attributes = 40 pour le dataset CelebA car il y a 40 attributs binaires à prédire

        # Création des couches linéaires pour le classificateur
        self.fc_layers = nn.Sequential(

            # La première couche prend 512 entrées (la sortie de l'encodeur)
            nn.Linear(512, 1024),  
             # Fonction d'activation LeakyReLU
            nn.LeakyReLU(0.2, inplace=True), 
            # Dropout pour réduire le surajustement
            nn.Dropout(0.5),  
            # Deuxième couche linéaire pour prédire les attributs
            nn.Linear(1024, n_attributes),  
            # Utilisation de Sigmoid car les attributs sont binaires
            nn.Sigmoid()  
        )

    def forward(self, x):
        # Aplatir les caractéristiques convolutives en un vecteur
        x = x.view(x.size(0), -1)  
        # Passer le vecteur à travers les couches linéaires
        x = self.fc_layers(x)  
        return x

