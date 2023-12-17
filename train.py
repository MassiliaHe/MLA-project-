# Import all blocks of the model
# All the training functions
# Initialise dataloader
# For loop on epochs
# For loop on dataloader
# Call step function of each part of the model

import torch
import argparse
import itertools
from tqdm import tqdm

from dataset.dataloader import get_dataloaders
from FaderNetwork.autoencoder import AutoEncoder
from FaderNetwork.discriminator import Discriminator
from utils.training import autoencoder_step, discriminator_step, get_optimizer, check_attr, save_models
from utils.evaluation import ModelEvaluator

from torch.utils.tensorboard import SummaryWriter


def configure_arg_parser():
    parser = argparse.ArgumentParser(description="Fader Networks")

    # Arguments optionnels avec valeurs par défaut
    parser.add_argument("--data_path", type=str, default="dataset", help="Path to the dataset.")
    parser.add_argument("--classifier_path", type=str, default="models/classifier.pth",
                        help="Path to the pre-trained classifier model.")
    parser.add_argument("--attr", type=str, default="Gender", help="Attribute for training (default: 'Gender').")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128).")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs (default: 2).")
    parser.add_argument("--learning_rate", type=float, default=0.0002,
                        help="Learning rate for the optimizer (default: 0.0002).")
    parser.add_argument("--lambda_ae", type=float, default=1,
                        help="Weight for the autoencoder loss component (default: 1).")
    parser.add_argument("--lambda_dis", type=float, default=0.0001,
                        help="Weight for the discriminator loss component (default: 0.0001).")
    parser.add_argument("--train_slice", type=int, default=2,
                        help="Proportion of the dataset to use for training (default: 2).")
    parser.add_argument("--val_slice", type=int, default=2,
                        help="Proportion of the dataset to use for validation (default: 2).")

    args = parser.parse_args()
    return args


def main(args):

    writer = SummaryWriter(log_dir='models', comment='Gender')

    train_dataloader, val_dataloader, _ = get_dataloaders(
        args.data_path, name_attr=args.attr, batch_size=args.batch_size)
    check_attr(args)

    # Create instances of your models (Encoder, Decoder, Classifier, Discriminator)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device if using GPU
    autoencoder = AutoEncoder(n_attributes=1).to(args.device)
    discriminator = Discriminator(n_attributes=1).to(args.device)
    # classifier = torch.load(args.classifier_path).to(device).eval()

    evaluator = ModelEvaluator(autoencoder, discriminator, val_dataloader, args)

    # Define optimizers for each model
    autoencoder_optimizer, discriminator_optimizer = get_optimizer(
        autoencoder, discriminator, args.learning_rate)

    best_ae_loss = 1000

    # Training loop
    rec_loss_tot, adv_loss_tot, disc_loss_tot = [], [], []
    for epoch in range(args.num_epochs):
        rec_loss, adv_loss, disc_loss = [], [], []
        limited_train_dataloader = itertools.islice(train_dataloader, args.train_slice)
        for iter, (images, attributes) in tqdm(enumerate(limited_train_dataloader), total=args.train_slice):
            images, attributes = images.to(args.device), attributes.to(args.device)
            # When training encoder, the decoder is in val mode vice versa
            ae_loss = autoencoder_step(args, autoencoder, discriminator,
                                       images, attributes, autoencoder_optimizer)
            rec_loss.append(ae_loss[0])
            adv_loss.append(ae_loss[1])
            # The discriminator needs the autoencoder in val mode
            disc_loss.append(discriminator_step(
                args, discriminator, autoencoder, images, attributes, discriminator_optimizer))
            # print(f"Batch [{iter+1}/{train_slice}], reconstruction Loss: {rec_loss[-1]:.3}, adversarial Loss: {adv_loss[-1]:.3}, Discriminator Loss: {disc_loss[-1]:.3}")
        # Print or log the losses
        rec_loss_tot.append(sum(rec_loss)/len(rec_loss))
        adv_loss_tot.append(sum(adv_loss)/len(adv_loss))
        disc_loss_tot.append(sum(disc_loss)/len(disc_loss))
        print(f"\nTrain Epoch [{epoch}/{args.num_epochs}] :  reconstruction Loss: {rec_loss_tot[-1]}, adversarial Loss: {adv_loss_tot[-1]}, Discriminator Loss: {disc_loss_tot[-1]}")

        # Validation for autoencoder, Discriminator
        evaluator.update_models(autoencoder, discriminator)
        accuracies = evaluator.evaluate(epoch)
        print(
            f"Eval Epoch [{epoch}/{args.num_epochs}] :  AE accuracy: {accuracies['ae_loss']}, Disciminator accuracy: {accuracies['disc_accu']}")

        # Save model
        if accuracies['ae_loss'] < best_ae_loss:
            best_ae_loss = accuracies['ae_loss']
            save_models(autoencoder, discriminator)
        if epoch % 5 == 0 and epoch > 0:
            save_models(autoencoder, discriminator)

        writer.add_scalar('Reconstruction_loss', rec_loss_tot[-1], epoch)
        writer.add_scalar('Adversarial_loss', adv_loss_tot[-1], epoch)
        writer.add_scalar('Loss_total', (rec_loss_tot[-1]+adv_loss_tot[-1])/2, epoch)
        writer.add_scalar('AE_loss', (accuracies['ae_loss']), epoch)
        writer.add_scalar('Disciminator_accu', (accuracies['disc_accu']), epoch)


if __name__ == "__main__":
    args = configure_arg_parser()
    main(args)
