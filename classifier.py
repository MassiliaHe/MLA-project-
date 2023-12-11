import torch
from torch.utils.data import DataLoader

from dataset.dataloader import CelebADataset, split_data
from FaderNetwork.classificateur import Classifier
from utils.training import classifier_step


def train_classier(base_dir,annotations_file,list_eval_partition,Attr):
    # Specify the path to your CelebA dataset
    # base_dir = img_align_celeba'#chemin vers les images 
    # annotations_file = list_attr_celeba.csv 
    # Use split_data to get the image IDs
    #Attr attribue qu'on souhaite extraire
    dataset_ids = split_data(list_eval_partition)
    # Créer des instances de CelebADataset pour chaque ensemble
    # Create an instance of the CelebADataset with specified transformations
    # TODO Instantiate with specific parameters
    celeba_dataset =  CelebADataset(base_dir, annotations_file,Attr, dataset_ids['train'])

    # Specify batch size and whether to shuffle the data
    batch_size = 64
    shuffle = True

    # Create DataLoader TODO Instantiate with specific parameters
    celeba_dataloader = DataLoader(dataset=celeba_dataset, batch_size=batch_size, shuffle=shuffle)

    # Create instances of your models (Encoder, Decoder, Classifier, Discriminator)
    # TODO Instantiate with specific parameters
    classifier = Classifier()  

    # Define training parameters
    num_epochs = 10  # Adjust as needed
    learning_rate = 0.001  # Adjust as needed

    # Define optimizers for each model TODO Each one choose an optimiser for it's model
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    # Move models to device if using GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        classifier.train()
        for batch in celeba_dataloader:
            images, attributes = batch['image'].to(device), batch['attributes'].to(device)
            classifier_loss = classifier_step(classifier, classifier_optimizer, images, attributes)
            # Print or log the losses if needed
            print(f"Epoch [{epoch+1}/{num_epochs}], Classifier Loss: {classifier_loss}")
        
        # TODO Add validation 

    # Optionally, save the trained models
    torch.save(classifier.state_dict(), 'classifier.pth')