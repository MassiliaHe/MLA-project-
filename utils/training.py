import torch
import torch.nn as nn



def classifier_step(classifier, images, attributes, classifier_optimizer):

    """
    Perform a single training step for the classifier.
    
    Arguments:
    - classifier: the neural network model.
    - images: a batch of input images.
    - attributes: the ground truth binary attributes for the images.
    - classifier_optimizer: optimizer for the classifier model.
    
    Returns:
    - loss.item(): The binary cross-entropy loss for the current batch.
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


def autoencoder_step(autoencoder, discriminator, images, attributes, autoencoder_optimizer):
    """
    Train the classifier.
    """
    autoencoder.train()
    # TODO
    ## batch / classify
    preds = autoencoder(images)

    ## loss / optimize

    loss = 0
    autoencoder_optimizer.zero_grad()
    loss.backward()
    autoencoder_optimizer.step()

    return loss

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

