import torch
from torch.utils.data import DataLoader

from dataset.dataloader import CelebADataset
from FaderNetwork.classifier_ import Classifier
from utils.training import classifier_step

def train_classifier():
    
    #paths and parameters
    celeba_root = '/path/to/celeba/dataset'
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001

    #dataset and dataLoader setup
    celeba_dataset = CelebADataset(root_dir=celeba_root, image_size=(64, 64), normalize=True)
    celeba_dataloader = DataLoader(dataset=celeba_dataset, batch_size=batch_size, shuffle=True)


    """
    #create the DataLoader for the validation dataset
    #path to the validation dataset
    validation_data_root = '/path/to/validation/dataset'  
    validation_batch_size = 64
    #to adjust according to the needs
    shuffle_validation = False  

    validation_dataset = CelebADataset(root_dir=validation_data_root, image_size=(64, 64), normalize=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=validation_batch_size, shuffle=shuffle_validation)
    """

    #Model and Optimizer
    classifier = Classifier()
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #move the model to the device
    classifier.to(device)

    # best validation loss initialization
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0

        #training step
        for batch in celeba_dataloader:
            images, attributes = batch['image'].to(device), batch['attributes'].to(device)
            classifier_loss = classifier_step(classifier, images, attributes, classifier_optimizer)
            total_loss += classifier_loss

        avg_loss = total_loss / len(celeba_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Classifier Loss: {avg_loss}")

        #validation step
        classifier.eval()
        val_loss = 0
        with torch.no_grad():

            #assuming we have the validation dataloader
            for val_images, val_attributes in validation_dataloader:
                val_preds = classifier(val_images.to(device))
                val_loss += torch.nn.functional.binary_cross_entropy_with_logits(val_preds, val_attributes.to(device))

        avg_val_loss = val_loss / len(validation_dataloader)
        print(f"Validation Loss: {avg_val_loss}")

        #save the model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(classifier.state_dict(), 'classifier_best.pth')
            print("Saved Improved Model")

#call the function to start training
train_classifier()

