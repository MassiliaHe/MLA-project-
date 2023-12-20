import torch
import os
import torch.nn as nn

AVAILABLE_ATTR = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

def classifier_step(classifier, images, attributes, classifier_optimizer):
    """
    train the classifier 
    """

    # set the classifier to training mode
    classifier.train()
    # forward pass: compute the predicted outputs.
    preds = classifier(images)

    # create a loss function for binary classification
    loss_function = nn.BCEWithLogitsLoss()
    # compute the loss
    loss = loss_function(preds, attributes)

    # zero the gradients of the classifier parameters
    classifier_optimizer.zero_grad()
    # perform backpropagation to calculate gradients
    loss.backward()
    # update the classifier parameters
    classifier_optimizer.step()

    # return the loss as a Python float
    return loss.cpu().data.item()


def autoencoder_step(args, autoencoder, discriminator, images, attributes, autoencoder_optimizer):
    """
    Perform a training step for both the autoencoder and the discriminator.

    :param autoencoder: The AutoEncoder model.
    :param discriminator: The Discriminator model.
    :param images: The batch of images.
    :param attributes: The batch of attributes associated with the images.
    :param autoencoder_optimizer: The optimizer for the autoencoder.
    :param discriminator_optimizer: The optimizer for the discriminator.
    :param lambda_val: The weight for the adversarial loss.
    :param criterion: The loss function for reconstruction (e.g., nn.MSELoss()).
    :return: The reconstruction loss and adversarial loss.
    """
    autoencoder.train()
    discriminator.eval()

    # Forward pass through the autoencoder.
    encoded_imgs, decoded_imgs = autoencoder(images, attributes)

    # autoencoder loss from reconstruction
    MSELoss = nn.MSELoss()
    reconstruction_loss = MSELoss(images, decoded_imgs)

    # encoder loss from the discriminator
    attributes_pred = discriminator(encoded_imgs)
    adversarial_loss = cross_entropy(attributes_pred, 1-attributes) # Fake attributs
    
    # Total loss
    ae_loss = args.lambda_ae * reconstruction_loss + args.lambda_dis * adversarial_loss

    # Update the autoencoder.
    autoencoder_optimizer.zero_grad()
    ae_loss.backward()
    autoencoder_optimizer.step()

    return reconstruction_loss.cpu().data.item(), adversarial_loss.cpu().data.item()


def discriminator_step(args, discriminator, autoencoder, images, attributes, discriminator_optimizer):
    """
    Train the discriminator.
    """
    discriminator.train()
    autoencoder.eval()
    # Generate fake attributes using the autoencoder
    with torch.no_grad():
        encoded_imgs = autoencoder.encode(images)
    attributes_pred = discriminator(encoded_imgs)
    adversarial_loss = cross_entropy(attributes_pred, attributes)
    # Backpropagation and optimization
    discriminator_optimizer.zero_grad()
    adversarial_loss.backward()
    discriminator_optimizer.step()

    return adversarial_loss.cpu().data.item()


def step(autoencoder, classifier_optimizer, discriminator, images, attributes, autoencoder_optimizer, classifier, criterion, discriminator_optimizer):
    clf_loss = classifier_step(classifier, images, attributes, classifier_optimizer, criterion)
    # Train autoencoder
    ae_loss = autoencoder_step(autoencoder, discriminator, images, attributes, autoencoder_optimizer, criterion)
    # Train discriminator
    dis_loss = discriminator_step(discriminator, autoencoder, images, attributes, discriminator_optimizer, criterion)
    return clf_loss, ae_loss, dis_loss


def get_optimizer(autoencoder, discriminator, learning_rate):
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    return autoencoder_optimizer, discriminator_optimizer

def get_classifier_optimizer(classifier, learning_rate):
    return torch.optim.Adam(classifier.parameters(), lr=learning_rate)

def cross_entropy(output, attributes):
    """
    Compute attributes loss.
    """
    # categorical
    y = attributes.max(1)[1].unsqueeze(1).to(attributes.dtype)
    BCE_loss = nn.BCEWithLogitsLoss()
    return BCE_loss(output, y)

def check_attr(args):
    """
    Check attributes validy.
    """
    if args.attr == '*':
        args.attr = attr_flag(','.join(AVAILABLE_ATTR))
    elif len(args.attr.split(',')) == 1:
        args.attr = attr_flag(args.attr)
    else:
        assert all(name in AVAILABLE_ATTR and n_cat >= 2 for name, n_cat in args.attr)
    args.n_attr = sum([n_cat for _, n_cat in args.attr])

def attr_flag(s):
    """
    Parse attributes parameters.
    """
    if s == "*":
        return s
    attr = s.split(',')
    assert len(attr) == len(set(attr))
    attributes = []
    for x in attr:
        if '.' not in x:
            attributes.append((x, 2))
        else:
            split = x.split('.')
            assert len(split) == 2 and len(split[0]) > 0
            assert split[1].isdigit() and int(split[1]) >= 2
            attributes.append((split[0], int(split[1])))
    return sorted(attributes, key=lambda x: (x[1], x[0]))

def save_models(autoencoder, discriminator, name='best'):
    """
    Save the best models / periodically save the models.
    """
    torch.save(autoencoder, f'models/{name}_autoencoder.pt')
    torch.save(discriminator, f'models/{name}_discriminator.pt')


def save_classifier(classifier, directory="path to saving folder", filename='classifier_best.pth'):

    """
    Sauvegarde the best model for the classifier
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)

    torch.save(classifier.state_dict(), filepath)

    print(f"Classificateur sauvegardé avec succès dans {filepath}")
