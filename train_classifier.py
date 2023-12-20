"""
This script trains a classifier 
"""

#import the PyTorch library for deep learning
import torch
#import the argparse library for parsing command-line arguments
import argparse
#import the itertools module for iteration and combination functions
import itertools
#import tqdm for displaying progress bars during iteration
from tqdm import tqdm


#import a custom function to get data loaders
from dataset.dataloader import get_dataloaders
#import a custom Classifier class
from FaderNetwork.classifier import Classifier
#import custom utility functions
from utils.training import classifier_step, get_classifier_optimizer, save_classifier

from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter for TensorBoard logging.

#this function configures and parses command-line arguments for the Classifier Training script
def configure_arg_parser():
    #create an argument parser object with a description for the script
    parser = argparse.ArgumentParser(description="Classifier Training")
    #add command-line arguments with their respective data types, default values, and help descriptions.
    #argument for specifying the data path (default path provided)
    parser.add_argument("--data_path", type=str, default="C:\\Users\\sabri\\OneDrive\\Bureau\\celebA_dataset", help="C:\\Users\\sabri\\OneDrive\\Bureau\\celebA_dataset")
    #argument for setting the batch size for training (default is 128)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128).")
    #argument for specifying the number of training epochs (default is 10)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    #argument for setting the learning rate for the optimizer (default is 0.001)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer (default: 0.001).")
    #argument for selecting the attribute for training (default is 'Young')
    parser.add_argument("--attr", type=str, default="Young", help="Attribute for training (default: 'Young').")
    #argument for specifying the proportion of the dataset to use for training (default is 2)
    parser.add_argument("--train_slice", type=int, default=2, help="Proportion of the dataset to use for training (default: 2).")
    #argument for specifying the proportion of the dataset to use for validation (default is 2)
    parser.add_argument("--val_slice", type=int, default=2, help="Proportion of the dataset to use for validation (default: 2).")
    #parse the command-line arguments and store them in an 'args' object
    args = parser.parse_args()
    #return the parsed arguments as an object
    return args

def main(args):
    writer = SummaryWriter(log_dir='logs', comment='Classifier')

    train_dataloader, val_dataloader, _ = get_dataloaders(args.data_path, name_attr=args.attr, batch_size=args.batch_size)

    #set the device (GPU if available, otherwise CPU)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #initialize the classifier and move it to the selected device
    classifier = Classifier().to(args.device)
    #get the optimizer for the classifier
    classifier_optimizer = get_classifier_optimizer(classifier, args.learning_rate)
    #create a classifier evaluator
    #evaluator = ClassifierEvaluator(classifier, val_dataloader, args)

    #initialize the best validation loss to positive infinity
    best_val_loss = float('inf')

    #loop through the specified number of training epochs
    for epoch in range(args.num_epochs):
        #set the classifier to training mode
        classifier.train()
        #initialize the total classifier loss
        total_classifier_loss = 0

        num_batches = 0
        for iter, (images, attributes) in tqdm(enumerate(itertools.islice(train_dataloader, args.train_slice)), total=args.train_slice):
            #move data to the device
            images, attributes = images.to(args.device), attributes.to(args.device)
            #compute classifier loss
            classifier_loss = classifier_step(classifier, images, attributes, classifier_optimizer)
            #accumulate the loss
            total_classifier_loss += classifier_loss
            num_batches += 1

        #calculate the average classifier loss
        avg_classifier_loss = total_classifier_loss / num_batches
        print(f"Epoch [{epoch}/{args.num_epochs}] - Classifier Loss: {avg_classifier_loss}")

        #set the classifier to evaluation mode
        classifier.eval()
        #initialize the total validation loss
        total_val_loss = 0

        #disable gradient computation for validation
        with torch.no_grad():
            #iterate through validation data
            for iter, (val_images, val_attributes) in enumerate(val_dataloader):
                #move data to the device
                val_images, val_attributes = val_images.to(args.device), val_attributes.to(args.device)
                #get classifier predictions
                val_outputs = classifier(val_images)

                #print dimensions for debugging
                print("Dimensions des sorties du classificateur: ", val_outputs.shape)
                print("Dimensions des cibles: ", val_attributes.shape)

                #compute validation loss
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(val_outputs, val_attributes)
                #accumulate the loss
                total_val_loss += val_loss.item()

        #calculate the average validation loss
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation Loss for Epoch {epoch}: {avg_val_loss}")

        #check if the current validation loss is better than the best so far
        if avg_val_loss < best_val_loss:
            #update the best validation loss
            best_val_loss = avg_val_loss
            #save the best classifier
            save_classifier(classifier, directory="C:\\Users\\sabri\\OneDrive\\Bureau\\celebA_dataset\\classifier_evaluation", filename="classifier_best.pth")

        #log the classifier loss to TensorBoard
        writer.add_scalar('Classifier Loss', avg_classifier_loss, epoch)
        #log the validation loss to TensorBoard
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
if __name__ == "__main__":
    #parse command-line arguments
    args = configure_arg_parser()
    #call the main function to start training.
    main(args)
