# Import all blocks of the model
# All the training functions
# Initialise dataloader
# For loop on epochs
# For loop on dataloader
# Call step function of each part of the model 

import torch
from torch.utils.data import DataLoader

from dataset.dataloader import CelebADataset
from FaderNetwork.autoencoder import AutoEncoder
from FaderNetwork.classifier import Classifier
from FaderNetwork.discriminator import Discriminator
from utils.training import autoencoder_step, classifier_step, discriminator_step


def train():
    # Specify the path to your CelebA dataset
    celeba_root = '/path/to/celeba/dataset'

    # Create an instance of the CelebADataset with specified transformations
    # TODO Instantiate with specific parameters
    celeba_dataset = CelebADataset(root_dir=celeba_root, image_size=(64, 64), normalize=True)

    # Specify batch size and whether to shuffle the data
    batch_size = 64
    shuffle = True

    # Create DataLoader TODO Instantiate with specific parameters
    celeba_dataloader = DataLoader(dataset=celeba_dataset, batch_size=batch_size, shuffle=shuffle)

    # Create instances of your models (Encoder, Decoder, Classifier, Discriminator)
    # TODO Instantiate with specific parameters
    autoencoder = AutoEncoder(n_attributes=1)  
    classifier = Classifier()  
    discriminator = Discriminator() 

    # Define training parameters
    num_epochs = 10  # Adjust as needed
    learning_rate = 0.001  # Adjust as needed

    # Move models to device if using GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    classifier.to(device)
    discriminator.to(device)

    # Define optimizers for each model TODO Each one choose an optimiser for it's model
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch in celeba_dataloader:
            images, attributes = batch['image'].to(device), batch['attributes'].to(device)

            # Training steps for each component
            # TODO When training encoder, the decoder is in val mode vice versa
            autoencoder_loss = autoencoder_step(autoencoder, discriminator, images, attributes, autoencoder_optimizer)
            # The discriminator needs the autoencoder in val mode
            discriminator_loss = discriminator_step(discriminator, autoencoder, images, attributes, discriminator_optimizer)
            classifier_loss = classifier_step(classifier, images, attributes, classifier_optimizer)
            # Print or log the losses if needed
            print(f"Epoch [{epoch+1}/{num_epochs}], autoencoder Loss: {autoencoder_loss}, Classifier Loss: {classifier_loss}, Discriminator Loss: {discriminator_loss}")
        
        # TODO Add validation for autoencoder, Classifier, Discriminator
        # validation for Classifier
        # validation for Autoencoder
        # validation for Discriminator

    # Optionally, save the trained models
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    torch.save(classifier.state_dict(), 'classifier.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

if __name__ == "__main__":
    train()
