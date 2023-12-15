"""
This script loads pre-trained FaderNetwork models (autoencoders, classifiers, discriminators) to process images
from the CelebA dataset to perform predictions and evaluations and visualize the results.
"""

#mport PyTorch library for deep learning tasks
import torch
#import NumPy for numerical operations
import numpy as np
#import matplotlib for plotting
import matplotlib.pyplot as plt
#import transforms for image preprocessing
from torchvision import transforms
#mport DataLoader for batch loading of dataset
from torch.utils.data import DataLoader
#import CelebADataset for loading CelebA dataset
from dataset.dataloader import CelebADataset

#import the model classes from FaderNetwork module
from FaderNetwork.autoencoder import AutoEncoder  
from FaderNetwork.classifier import Classifier   
from FaderNetwork.discriminator import Discriminator  


# load a pre-trained model from a specified file path
def load_model(model_path, model_class):

    #create an instance of the specified model class
    model = model_class()
    #load the model's state dictionary from the file
    model.load_state_dict(torch.load(model_path))
    #set the model to evaluation mode (no training)
    model.eval()
    #return the loaded model
    return model

def get_dataloader(image_folder, batch_size=64):

    #define image transformations: resize the image and convert it to a tensor
    transform = transforms.Compose([
        #resize the image
        transforms.Resize((256, 256)),  
        #convert the image to a tensor
        transforms.ToTensor(),          
    ])

    #create an instance of CelebADataset with the specified transformations
    dataset = CelebADataset(image_folder, transform=transform)
    # create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def predict(autoencoder, classifier, discriminator, dataloader, device):
    # iitialize lists to store results from autoencoder, classifier, and discriminator
    ae_results = []
    classifier_results = []
    discriminator_results = []

    #disable gradient calculations for prediction
    with torch.no_grad():
        for images, _ in dataloader:

            #move images to the specified device
            images = images.to(device)
            #get the output from the autoencoder
            ae_output = autoencoder(images)
            #get the output from the classifier
            classifier_output = classifier(ae_output)
            #get the output from the discriminator
            discriminator_output = discriminator(ae_output)

            #append the outputs to the respective result lists
            ae_results.append(ae_output.cpu())
            classifier_results.append(classifier_output.cpu())
            discriminator_results.append(discriminator_output.cpu())
    return ae_results, classifier_results, discriminator_results

def display_images(images, ncols=5):

    #calculate the number of rows needed for displaying images
    nrows = len(images) // ncols + (len(images) % ncols > 0)
    #create a figure with the specified number of rows and columns
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))
    for i, img in enumerate(images):
        #get the current axis for the image
        ax = axes.flat[i]
        #rearrange the dimensions for display
        ax.imshow(np.transpose(img, (1, 2, 0)))  
        #turn off the axis
        ax.axis('off')
    #display the images
    plt.show()

if __name__ == "__main__":

    #paths to the saved models
    autoencoder_path = "chemin/vers/votre/autoencoder.pth"
    classifier_path = "chemin/vers/votre/classificateur.pth"
    discriminator_path = "chemin/vers/votre/discriminateur.pth"
    image_folder = "chemin/vers/le/dossier/des/images"

    #set up the device for running models (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #load models onto the device
    autoencoder = load_model(autoencoder_path, AutoEncoder).to(device)
    classifier = load_model(classifier_path, Classifier).to(device)
    discriminator = load_model(discriminator_path, Discriminator).to(device)

    #get a DataLoader for the images
    dataloader = get_dataloader(image_folder)

    #make predictions with the AutoEncoder, Classifier, and Discriminator
    ae_results, classifier_results, discriminator_results = predict(autoencoder, classifier, discriminator, dataloader, device)

    #display some reconstructed images and their predicted attributes
    display_images([batch[0] for batch in ae_results], ncols=3)

    #print the predicted attributes for the first images
    print("Attributs prédits pour les premières images :")
    for batch in classifier_results:
        print(batch[0])

    #print the discriminator's responses for the first images
    print("Réponses du discriminateur pour les premières images :")
    for batch in discriminator_results:
        print(batch[0])
