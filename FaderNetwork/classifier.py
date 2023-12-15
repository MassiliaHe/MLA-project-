"""
This code implements a PyTorch classifier to predict
40 binary attributes from the CelebA dataset based on latent representations

"""

#import the PyTorch library for deep learning
import torch
#import the nn module from PyTorch, which contains classes for building neural networks
import torch.nn as nn

#define the Classifier class that inherits from nn.Module
#nn.Module: base class for all neural network modules in PyTorch

class Classifier(nn.Module):

    #constructor of the Classifier class
    #n_attributes: parameter storing the number of attributes with a default value of 40
    #which corresponds to the number of attributes in the CelebA dataset.
    def __init__(self, n_attributes=40):

        #call the constructor of the parent class to initialize the module correctly
        super(Classifier, self).__init__()

        #build the linear layers for the classifier
        self.fc_layers = nn.Sequential(

            #the first linear layer takes 512 inputs (the output of the encoder)
            #and transforms them into an intermediate dimension (1024)
            nn.Linear(512, 1024),  
            #normalization by batches: to establish the training
            nn.BatchNorm1d(1024),
            #LeakyReLU activation function: to introduce non-linearity
            nn.LeakyReLU(0.2, inplace=True), 
            #Dropout to prevent overfitting
            nn.Dropout(0.5),  
            #second linear layer to predict attributes 
            nn.Linear(1024, n_attributes),  
            #using Sigmoid because the attributes are binary
            #transforming the output into probabilities for each attribute
            nn.Sigmoid()  
        )

    def forward(self, x):
        #flatten the convolutional features into a vector
        x = x.view(x.size(0), -1)  
        #pass the vector through the linear layers
        x = self.fc_layers(x)  
        #return the predicted binary attributes
        return x
