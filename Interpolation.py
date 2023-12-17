import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.image

from dataset.dataloader import get_dataloaders


# Parse parameters for attribute swapping.
parser = argparse.ArgumentParser(description='Perform attribute swapping on images.')
parser.add_argument("--data_path", type=str, default="dataset", help="Path to the dataset")  # Path to the data.
parser.add_argument("--model_path", type=str, default="models/best_autoencoder.pt",
                    help="Path to the trained autoencoder model")  # Trained model path.
parser.add_argument("--n_images", type=int, default=2, help="Number of images to modify")  # Number of images to modify.
# First image index.
parser.add_argument("--offset", type=int, default=0, help="Index offset for the first image to be modified")
# Number of interpolations per image.
parser.add_argument("--n_interpolations", type=int, default=5, help="Number of attribute interpolations per image")
# Min interpolation value.
parser.add_argument("--alpha_min", type=float, default=0.0, help="Minimum alpha value for interpolation")
# Max interpolation value.
parser.add_argument("--alpha_max", type=float, default=1.0, help="Maximum alpha value for interpolation")
# Size of images in the grid.
parser.add_argument("--plot_size", type=int, default=5, help="Size of the plot grid for displaying images")
# Represent image interpolations horizontally.
parser.add_argument("--row_wise", type=bool, default=True,
                    help="Layout the interpolation row-wise if true, column-wise if false")
# Output path.
parser.add_argument("--output_path", type=str, default="output.png", help="File path for the output image")
params = parser.parse_args()


# check parameters
assert os.path.isfile(params.model_path)
assert params.n_images >= 1 and params.n_interpolations >= 2

# create logger / load trained model
params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae = torch.load(params.model_path).to(params.device)

# restore main parameters
params.debug = True
params.batch_size = 32
params.v_flip = False
params.h_flip = False
params.img_sz = 256
params.attr = ['Young']
params.n_attr = 2
if not (len(params.attr) == 1 and params.n_attr == 2):
    raise Exception("The model must use a single boolean attribute only.")

# load dataset

_, _, test_data = get_dataloaders(
    params.data_path, name_attr=params.attr[0], batch_size=params.batch_size)


def get_interpolations(ae, images, attributes, params):
    """
    Reconstruct images / create interpolations
    """
    assert len(images) == len(attributes)
    enc_outputs = ae.encode(images)

    # interpolation values
    alphas = np.linspace(1 - params.alpha_min, params.alpha_max, params.n_interpolations)
    alphas = [torch.FloatTensor([1 - alpha, alpha]).to(params.device) for alpha in alphas]

    # original image / reconstructed image / interpolations
    outputs = []
    outputs.append(images)
    outputs.append(ae.decode(enc_outputs, attributes))
    for alpha in alphas:
        alpha = Variable(alpha.unsqueeze(0).expand((len(images), 2)))
        outputs.append(ae.decode(enc_outputs, alpha))

    # return stacked images
    return torch.cat([x.unsqueeze(1) for x in outputs], 1).data.cpu()


interpolations = []

for k in range(0, params.n_images, 100):
    images, attributes = next(iter(test_data))
    images, attributes = images.to(params.device), attributes.to(params.device)
    interpolations.append(get_interpolations(ae, images[:params.n_images], attributes[:params.n_images], params))

interpolations = torch.cat(interpolations, 0)
assert interpolations.size() == (params.n_images, 2 + params.n_interpolations,
                                 3, params.img_sz, params.img_sz)


def get_grid(images, row_wise, plot_size=5):
    """
    Create a grid with all images.
    """
    n_images, n_columns, img_fm, img_sz, _ = images.size()
    if not row_wise:
        images = images.transpose(0, 1).contiguous()
    images = images.view(n_images * n_columns, img_fm, img_sz, img_sz)
    images.add_(1).div_(2.0)
    return make_grid(images, nrow=(n_columns if row_wise else n_images))


# generate the grid / save it to a PNG file
grid = get_grid(interpolations, params.row_wise, params.plot_size)
normalized_image = (grid.cpu().numpy().transpose((1, 2, 0)) - grid.cpu().numpy().min()) / \
    (grid.cpu().numpy().max() - grid.cpu().numpy().min())
matplotlib.image.imsave(params.output_path, normalized_image)
