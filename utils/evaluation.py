# Import libraries
import json
import numpy as np
import torch
import torch.nn
from logging import getLogger
# from FaderNetwork.autoencoder import modify_predictions, toggle_attributes

# Initialize logger
eval_logger = getLogger()

# Evaluation class


class ModelEvaluator(object):

    def __init__(self, autoencoder, discriminator, dataset, settings):
        """
        Initialize the model evaluator.
        """
        # Assign dataset and settings
        self.dataset = dataset
        self.settings = settings

        # Assign model components
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        # self.classifier_evaluator = classifier_evaluator
        # assert classifier_evaluator.image_size == settings.image_size
        # assert all(attribute in classifier_evaluator.attributes for attribute in settings.attributes)

    def validate(self):
        print('Accuracy : 0.0')
        return 0.0, 0.0

    # Define methods for evaluation (e.g., eval_autoencoder_loss, eval_latent_discriminator_accuracy, etc.)

    # def eval_classifier_accuracy(self):
    #     """
    #     Evaluate and log the accuracy of the classifier.
    #     """
        # self.dataloader = self.dataset.get_dataloader(batch_size=self.settings.batch_size, shuffle=False)
        # accuracy = self.calculate_classifier_accuracy()
        # eval_logger.info(f"Classifier Accuracy: {accuracy:.4f}")

        # Below, redefine the methods of the Evaluator class with new names and slight modifications
        # while keeping the core functionality and logic intact.
        # The methods include computation of various accuracies and losses, and logging these evaluations.

        # method to compute classifier accuracy outside the class

    # def calculate_classifier_accuracy(self):
    #     """
    #     Compute the accuracy of the classifier.
    #     """

        # #set the classifier in evaluation mode
        # self.classifier.eval()

        # #create a DataLoader for the dataset
        # dataloader = dataloader(self.dataset, batch_size=self.settings.batch_size, shuffle=False)

        # #iInitialize counters for correct predictions and total samples processed
        # correct_predictions = 0
        # total_samples = 0

        # #disabling gradient calculation to save memory and computations during evaluation
        # with torch.no_grad():
        #     for batch in dataloader:
        #         images, attributes = batch['image'], batch['attributes']
        #         outputs = self.classifier(images.to(self.settings.device))

        #         #binarize the classifier outputs based on a threshold (e.g., 0.5)
        #         predictions = (torch.sigmoid(outputs) > 0.5).int()

        #         #calculate the number of correct predictions in this batch
        #         correct_predictions += (predictions == attributes).sum().item()
        #         total_samples += len(images)

        # #calculate and return accuracy as the ratio of correct predictions to total samples
        # accuracy = correct_predictions / total_samples

        # #return the accuracy
        # return accuracy
