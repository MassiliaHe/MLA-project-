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


def autoencoder_step(autoencoder, discriminator, images, attributes, autoencoder_optimizer, criterion):
    """
    Train the autoencoder.
    """
    autoencoder.train()
    # Encode and decode images
    encoded_imgs = autoencoder.encode(images)
    decoded_imgs = autoencoder.decode(encoded_imgs, attributes)
    # Calculate loss
    loss = criterion(decoded_imgs, images)  # Assuming you have a reconstruction loss
    # Backpropagation and optimization
    autoencoder_optimizer.zero_grad()
    loss.backward()
    autoencoder_optimizer.step()

    return loss.item()


def discriminator_step(discriminator, autoencoder, images, attributes, discriminator_optimizer, criterion):
    """
    Train the discriminator.
    """
    discriminator.train()
    # Generate fake attributes using the autoencoder
    with torch.no_grad():
        encoded_imgs = autoencoder.encode(images)
        fake_attributes = discriminator(encoded_imgs)
    # Calculate loss
    real_loss = criterion(discriminator(images), attributes.float())
    fake_loss = criterion(fake_attributes, torch.zeros_like(fake_attributes))
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

