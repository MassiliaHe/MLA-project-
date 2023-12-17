# Import all blocks of the model
# All the training functions
# Initialise dataloader
# For loop on epochs
# For loop on dataloader
# Call step function of each part of the model

import torch
import argparse

from dataset.dataloader import get_dataloaders
from FaderNetwork.autoencoder import AutoEncoder
from FaderNetwork.discriminator import Discriminator
from utils.training import autoencoder_step, discriminator_step, get_optimizer
from utils.evaluation import ModelEvaluator


def configure_arg_parser():
    parser = argparse.ArgumentParser(description="Fader Networks")

    # Arguments optionnels avec valeurs par défaut
    parser.add_argument("--data_path", type=str, default="E:\\M2_2023\\MLA\\FaderNetworks\\datasets", help="Chemin vers les données")
    parser.add_argument("--classifier_path", type=str, default="models/classifier.pth", help="Chemin vers le classificateur")
    parser.add_argument("--Attr", type=str, default="Young", help="Attribut à utiliser (par défaut: 'Young')")
    parser.add_argument("--batch_size", type=int, default=64, help="Taille du lot (par défaut: 64)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Nombre d'époques (par défaut: 10)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Taux d'apprentissage (par défaut: 0.001)")

    args = parser.parse_args()
    return args


def main(args):

    train_dataloader, val_dataloader, _ = get_dataloaders(
        args.data_path, name_attr=args.Attr, batch_size=args.batch_size)

    # Create instances of your models (Encoder, Decoder, Classifier, Discriminator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device if using GPU
    autoencoder = AutoEncoder(n_attributes=1).to(device)
    discriminator = Discriminator(n_attributes=1).to(device)
    # classifier = torch.load(args.classifier_path).to(device).eval()

    evaluator = ModelEvaluator(autoencoder, discriminator, val_dataloader, args)

    # Define optimizers for each model
    autoencoder_optimizer, discriminator_optimizer = get_optimizer(
        autoencoder, discriminator, args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        for batch in train_dataloader:
            images, attributes = batch['image'].to(device), batch['attributes'].to(device)
            # When training encoder, the decoder is in val mode vice versa
            autoencoder_loss = autoencoder_step(autoencoder, discriminator, images, attributes, autoencoder_optimizer)
            # The discriminator needs the autoencoder in val mode
            discriminator_loss = discriminator_step(
                discriminator, autoencoder, images, attributes, discriminator_optimizer)
            # Print or log the losses if needed
            print(f"Epoch [{epoch+1}/{args.num_epochs}], autoencoder Loss: {autoencoder_loss}, Discriminator Loss: {discriminator_loss}")

        # TODO Add validation for autoencoder, Classifier, Discriminator
        accu_ae, accu_disc = evaluator.validate()
        print(f'accu_ae, {accu_ae} \naccu_disc : {accu_disc}')

    # Optionally, save the trained models
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')


if __name__ == "__main__":
    args = configure_arg_parser()
    main(args)
