import torch
import torch.nn as nn

def classifier_step(classifier, images, attributes, classifier_optimizer):

    """
    train the classifier 
    """

    #set the classifier to training mode
    classifier.train() 
    #forward pass: compute the predicted outputs. 
    preds = classifier(images)  
    
    #create a loss function for binary classification
    loss_function = nn.BCEWithLogitsLoss() 
    #compute the loss 
    loss = loss_function(preds, attributes)  
    
    #zero the gradients of the classifier parameters
    classifier_optimizer.zero_grad() 
    #perform backpropagation to calculate gradients
    loss.backward()  
    #update the classifier parameters
    classifier_optimizer.step()  

    #return the loss as a Python float
    return loss.item()  


import torch

def autoencoder_step(autoencoder, discriminator, images, attributes, autoencoder_optimizer, discriminator_optimizer, lambda_val, criterion, step_count):
    """
    Perform a training step for both the autoencoder and the discriminator.

    :param autoencoder: The AutoEncoder model.
    :param discriminator: The Discriminator model.
    :param images: The batch of images.
    :param attributes: The batch of attributes associated with the images.
    :param autoencoder_optimizer: The optimizer for the autoencoder.
    :param discriminator_optimizer: The optimizer for the discriminator.
    :param lambda_val: The weight for the adversarial loss.
    :param criterion: The loss function for reconstruction (e.g., nn.MSELoss()).
    :param step_count: The current step count for adjusting lambda_val.
    :return: The reconstruction loss and adversarial loss.
    """
    autoencoder.train()
    discriminator.train()

    # Forward pass through the autoencoder.
    encoded_imgs, decoded_imgs = autoencoder(images, attributes)

    # Compute the reconstruction loss.
    reconstruction_loss = criterion(decoded_imgs, images)

    # Forward pass through the discriminator.
    attributes_pred = discriminator(encoded_imgs.detach())

    # Compute the adversarial loss.
    adversarial_loss = -torch.mean(torch.log(attributes_pred + 1e-8) * attributes + torch.log(1 - attributes_pred + 1e-8) * (1 - attributes))

    # Update the discriminator.
    discriminator_optimizer.zero_grad()
    adversarial_loss.backward(retain_graph=True)  
    discriminator_optimizer.step()

    # Compute the total loss for the autoencoder.
    total_loss = reconstruction_loss + lambda_val(step_count) * adversarial_loss

    # Update the autoencoder.
    autoencoder_optimizer.zero_grad()
    total_loss.backward()
    autoencoder_optimizer.step()

    return reconstruction_loss.item(), adversarial_loss.item()




def discriminator_step(discriminator, autoencoder, images,  real_images, real_attributes, fake_images, discriminator_optimizer, criterion):
    """
    Train the discriminator.
    """
    discriminator.train()
    # Generate fake attributes using the autoencoder
    with torch.no_grad():
        encoded_imgs = autoencoder.encode(images)
        fake_attributes = discriminator(encoded_imgs)
    # Calculate the discriminator's prediction on real data
    real_predictions = discriminator(real_images)
    real_loss = criterion(real_predictions, real_attributes)
    # Calculate loss
    fake_predictions = discriminator(fake_images.detach())
    fake_loss = criterion(fake_predictions, torch.zeros_like(fake_predictions))
    loss = (real_loss + fake_loss) / 2
    # Backpropagation and optimization
    discriminator_optimizer.zero_grad()
    loss.backward()
    discriminator_optimizer.step()

    return loss.item()

def step(autoencoder,classifier_optimizer, discriminator, images, attributes, autoencoder_optimizer,classifier, criterion, discriminator_optimizer):
    clf_loss = classifier_step(classifier, images, attributes, classifier_optimizer, criterion)
    # Train autoencoder
    ae_loss = autoencoder_step(autoencoder, discriminator, images, attributes, autoencoder_optimizer, criterion)
    # Train discriminator
    dis_loss = discriminator_step(discriminator, autoencoder, images, attributes, discriminator_optimizer, criterion)
    return clf_loss, ae_loss, dis_loss

