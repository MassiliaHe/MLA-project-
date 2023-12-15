# Import libraries
import json
import numpy as np
import torch
import torch.nn
from logging import getLogger
from autoencoder import modify_predictions, toggle_attributes
from evaluation import display_accuracies

# Initialize logger
eval_logger = getLogger()

# Evaluation class
class ModelEvaluator(object):

    def __init__(self, autoencoder, latent_discriminator, patch_discriminator, classifier_discriminator, classifier_evaluator, dataset, settings):
        """
        Initialize the model evaluator.
        """
        # Assign dataset and settings
        self.dataset = dataset
        self.settings = settings

        # Assign model components
        self.autoencoder = autoencoder
        self.latent_discriminator = latent_discriminator
        self.patch_discriminator = patch_discriminator
        self.classifier_discriminator = classifier_discriminator
        self.classifier_evaluator = classifier_evaluator
        assert classifier_evaluator.image_size == settings.image_size
        assert all(attribute in classifier_evaluator.attributes for attribute in settings.attributes)

    # Define methods for evaluation (e.g., eval_autoencoder_loss, eval_latent_discriminator_accuracy, etc.)
        

    def eval_classifier_accuracy(self):
        """
        Evaluate and log the accuracy of the classifier.
        """
        dataloader = self.dataset.get_dataloader(batch_size=self.settings.batch_size, shuffle=False)
        accuracy = calculate_classifier_accuracy(self.classifier_discriminator, dataloader, self.settings.device)
        eval_logger.info(f"Classifier Accuracy: {accuracy:.4f}")


# Below, redefine the methods of the Evaluator class with new names and slight modifications
# while keeping the core functionality and logic intact.
# The methods include computation of various accuracies and losses, and logging these evaluations.

# Method to compute classifier accuracy outside the class
def calculate_classifier_accuracy(classifier, dataset, settings):
    """
    Compute the accuracy of the classifier.
    """
    
    # Mise en mode d'évaluation du classificateur
    classifier.eval()

    # Création d'un DataLoader pour l'ensemble de données
    dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images, attributes = batch['image'], batch['attributes']
            outputs = classifier(images.to(settings.device))

            # Binarisez les sorties du classificateur en fonction d'un seuil (par exemple, 0,5)
            predictions = (torch.sigmoid(outputs) > 0.5).int()

            # Calculez le nombre de prédictions correctes dans ce batch
            correct_predictions += (predictions == attributes).sum().item()
            total_samples += len(images)

    accuracy = correct_predictions / total_samples

    return accuracy




