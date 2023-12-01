"""
Ce code implémente un classificateur PyTorch pour prédire
40 attributs binaires du dataset CelebA à partir des représentations latentes
"""

#importer la bibliothèque PyTorch pour l'apprentissage profond
import torch
#importer le module nn de PyTorch qui contient des classes pour construire des réseaux de neurones
import torch.nn as nn


#définition de la classe FaderNetworksClassifier qui hérite de nn.Module
#nn.Module : classe de base pour tous les modules de réseau neuronal dans PyTorch
class FaderNetworksClassifier(nn.Module):

    #constructeur de la classe FaderNetworksClassifier
    #n_attributes: paramètre stockant le nombre d'attributs avec une valeur par défaut de 40
    # ce qui correspond au nombre d'attributs dans le dataset CelebA.
    def __init__(self, n_attributes=40):

        #appeler le constructeur de la classe mère pour initialiser le module correctement
        super(FaderNetworksClassifier, self).__init__()

        #construire les couches linéaires pour le classificateur
        self.fc_layers = nn.Sequential(

            #la première couche linéaire prend 512 entrées (la sortie de l'encodeur)
            #et les transforme en une dimension intermédiaire (1024)
            nn.Linear(512, 1024),  
            #fonction d'activation LeakyReLU: pour introduire la non-linéarité
            nn.LeakyReLU(0.2, inplace=True), 
            #dropout pour prévenir l'overfitting
            nn.Dropout(0.5),  
            #deuxième couche linéaire pour prédire les attributs 
            nn.Linear(1024, n_attributes),  
            #utilisation de Sigmoid car les attributs sont binaires
            #transformer la sortie en probabilités pour chaque attribu
            nn.Sigmoid()  
        )

    def forward(self, x):
        #aplatir les caractéristiques convolutives en un vecteur
        x = x.view(x.size(0), -1)  
        #passer le vecteur à travers les couches linéaires
        x = self.fc_layers(x)  
        return x

