import torch
import torch.nn as nn

def classifier_step(classifier, images, attributes, classifier_optimizer):
    """
    Train the classifier.
    """
    classifier.train()
    # TODO
    ## batch / classify
    preds = classifier(images)

    ## loss / optimize

    loss = 0 
    classifier_optimizer.zero_grad()
    loss.backward()
    classifier_optimizer.step()

    return loss

def autoencoder_step(autoencoder, discriminator, images, attributes, autoencoder_optimizer, criterion):
    """
    Train the classifier.
    """
     # Encode and decode images
    encoded_imgs = autoencoder.encode(images)
    decoded_imgs = autoencoder.decode(encoded_imgs, attributes)
    # Calculate loss
    loss = criterion(decoded_imgs, images)
    
    # Backpropagation and optimization
    autoencoder_optimizer.zero_grad()
    loss.backward()
    autoencoder_optimizer.step()

    return loss.item()

def discriminator_step(discriminator, autoencoder, images, attributes, discriminator_optimizer):
    """
    Train the classifier.
    """

    criterion = nn.BCELoss()

    discriminator.train()
    # TODO
    ## batch / classify
    preds = discriminator(images)

    ## loss / optimize
    loss = criterion(preds, attributes.float())
    discriminator_optimizer.zero_grad()
    loss.backward()
    discriminator_optimizer.step() 

    return loss

def step():
    # TODO 
    # Add a step function to train all the model for a batch 
    # (facultatif)
    pass