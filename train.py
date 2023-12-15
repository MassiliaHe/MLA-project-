# Import all blocks of the model
# All the training functions
# Initialise dataloader
# For loop on epochs
# For loop on dataloader
# Call step function of each part of the model 

import torch
from torch.utils.data import DataLoader

from dataset.dataloader import CelebADataset, split_data
from FaderNetwork.autoencoder import AutoEncoder
from FaderNetwork.classifier import Classifier
from FaderNetwork.discriminator import Discriminator
from utils.training import autoencoder_step, classifier_step, discriminator_step
from utils.evaluation import ModelEvaluator

def train(base_dir,annotations_file,list_eval_partition,Attr):
    # Specify the path to your CelebA dataset
    # base_dir = img_align_celeba'#chemin vers les images 
    # annotations_file = list_attr_celeba.csv 
    # Use split_data to get the image IDs
    #Attr attribue qu'on souhaite extraire
    dataset_ids = split_data(list_eval_partition)
    # Créer des instances de CelebADataset pour chaque ensemble
    # Create an instance of the CelebADataset with specified transformations
    # TODO Instantiate with specific parameters
    celeba_dataset = CelebADataset(root_dir=celeba_root, image_size=(64, 64), normalize=True)

    # Specify batch size and whether to shuffle the data
    batch_size = 64


    # Create DataLoader TODO Instantiate with specific parameters
    celeba_dataloader = DataLoader(dataset=celeba_dataset, batch_size=batch_size, shuffle=shuffle)

    # Create instances of your models (Encoder, Decoder, Classifier, Discriminator)
    autoencoder = AutoEncoder(n_attributes=1)  
    classifier = Classifier()  
    discriminator = Discriminator() 
    eval_clf = torch.load(eval).cuda().eval()


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
        for batch in celeba_dataloader_train:
            images, attributes = batch['image'].to(device), batch['attributes'].to(device)

            # Training steps for each component
            # When training encoder, the decoder is in val mode vice versa
            autoencoder_loss = autoencoder_step(autoencoder, discriminator, images, attributes, autoencoder_optimizer)
            # The discriminator needs the autoencoder in val mode
            discriminator_loss = discriminator_step(discriminator, autoencoder, images, attributes, discriminator_optimizer)
            classifier_loss = classifier_step(classifier, images, attributes, classifier_optimizer)
            # Print or log the losses if needed
            print(f"Epoch [{epoch+1}/{num_epochs}], autoencoder Loss: {autoencoder_loss}, Classifier Loss: {classifier_loss}, Discriminator Loss: {discriminator_loss}")

        evaluate = ModelEvaluator(autoencoder, discriminator, classifier, celeba_dataset)
        # TODO Add validation for autoencoder, Classifier, Discriminator
        # validation for Classifier
        # validation for Autoencoder
        # validation for Discriminator

        # validation for Classifier
        #set the classifier to evaluation mode
        classifier.eval()
        #initialize total validation loss for the classifier
        total_val_loss_classifier = 0

        #disable gradient calculation for validation
        with torch.no_grad():
            #iterate through the validation data
            for val_images, val_attributes in celeba_dataloader_val:
                #forward pass: compute the classifier's output for validation images
                val_outputs_classifier = classifier(val_images.to(device))

                #calculate the validation loss using Binary Cross-Entropy Loss
                #this compares the classifier's output with the true attributes
                val_loss_classifier = torch.nn.functional.binary_cross_entropy_with_logits(val_outputs_classifier, val_attributes.to(device))

                #accumulate the validation loss
                total_val_loss_classifier += val_loss_classifier.item()

        #compute the average validation loss for the classifier over all batches
        avg_val_loss_classifier = total_val_loss_classifier / len(celeba_dataloader_val)

        #print the average validation loss for the classifier
        print(f"Validation Loss for Classifier: {avg_val_loss_classifier}")


    # Optionally, save the trained models
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    torch.save(classifier.state_dict(), 'classifier.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

if __name__ == "__main__":
    train()
