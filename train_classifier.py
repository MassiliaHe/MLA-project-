"""
This script trains a classifier model using the CelebA dataset
"""
#import the OS module for interacting with the operating system
import os
#import the PyTorch library for deep learning tasks
import torch
#import the neural network module from PyTorch
import torch.nn as nn
#import DataLoader for batch loading of dataset
from torch.utils.data import DataLoader

#import CelebADataset class and split_data function from the dataset.dataloader module
from dataset.dataloader import CelebADataset, split_data
#import the Classifier class from the FaderNetwork module
from FaderNetwork.classifier import Classifier  
#import the classifier_step function from the utils.training module
from utils.trining import classifier_step

def train_classifier(base_dir, annotations_file, list_eval_partition, Attr):

    #check if the specified directories exist
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"The base directory {base_dir} was not found.")
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"The annotations file {annotations_file} was not found.")
    
    #path to save the best model
    model_save_path = 'classifier_best.pth'

    #split data into training, validation, and test sets
    dataset_ids = split_data(list_eval_partition)

     #dataLoader for training dataset
    train_dataset = CelebADataset(base_dir, annotations_file, Attr, dataset_ids['train'])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    #dataLoader for validation dataset
    validation_dataset = CelebADataset(base_dir, annotations_file, Attr, dataset_ids['validation'])
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=64, shuffle=False)

    #initialize the model and optimizer
    classifier = Classifier()
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    #set the number of epochs for training
    num_epochs = 10
    #initialize the variable to track the best validation loss
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        #training loop
        classifier.train()
        for batch in train_dataloader:
            #load images and their corresponding attributes to the device
            images, attributes = batch['image'].to(device), batch['attributes'].to(device)
            #perform a training step and get the classifier loss
            classifier_loss = classifier_step(classifier, images, attributes, classifier_optimizer)
            #print the classifier loss for each epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Classifier Loss: {classifier_loss}")

        #validation loop
        classifier.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in validation_dataloader:
                #load validation images and their attributes to the device
                val_images, val_attributes = batch['image'].to(device), batch['attributes'].to(device)
                #get the classifier's output for the validation images
                val_outputs = classifier(val_images)
                #calculate the validation loss
                val_loss = nn.functional.binary_cross_entropy_with_logits(val_outputs, val_attributes)
                #accumulate the validation loss
                total_val_loss += val_loss.item()

        #calculate the average validation loss for the epoch
        avg_val_loss = total_val_loss / len(validation_dataloader)
        #print the average validation loss for the epoch
        print(f"Validation Loss for Epoch {epoch+1}: {avg_val_loss}")

        #save the classifier model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            #create the directory for model_save_path if it doesn't exist
            if not os.path.isdir(os.path.dirname(model_save_path)):
                os.makedirs(os.path.dirname(model_save_path))
            #save the state of the classifier
            torch.save(classifier.state_dict(), model_save_path)

#entry point for the script
if __name__ == "__main__":
    #call the train_classifier function with the specified paths and attribute name
    train_classifier('/path/to/celeba/dataset', 'list_attr_celeba.csv', 'list_eval_partition.csv', 'Attribute_Name')
