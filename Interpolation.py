import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.image

from src.logging_util import init_logger
from src.image_loader import load_dataset, SampleData
from src.config_utils import parse_boolean

# Setup parameters
param_parser = argparse.ArgumentParser(description='Attribute modification')
param_parser.add_argument("--path_to_model", type=str, default="",
                          help="Path to the trained model")
param_parser.add_argument("--num_images", type=int, default=10,
                          help="Quantity of images to alter")
param_parser.add_argument("--start_index", type=int, default=0,
                          help="Index of the first image")
param_parser.add_argument("--num_interpolations", type=int, default=10,
                          help="Interpolations per image")
param_parser.add_argument("--min_alpha", type=float, default=1,
                          help="Minimum interpolation value")
param_parser.add_argument("--max_alpha", type=float, default=1,
                          help="Maximum interpolation value")
param_parser.add_argument("--grid_size", type=int, default=5,
                          help="Dimension of images in grid")
param_parser.add_argument("--horizontal_grid", type=parse_boolean, default=True,
                          help="Horizontal image interpolations")
param_parser.add_argument("--save_path", type=str, default="output.png",
                          help="Path for saving output")
parameters = param_parser.parse_args()

# Validate parameters
assert os.path.isfile(parameters.path_to_model)
assert parameters.num_images >= 1 and parameters.num_interpolations >= 2

# Initialize logger and load model
log = init_logger(None)
autoencoder_model = torch.load(parameters.path_to_model).eval()

# Set main parameters
parameters.debug_mode = True
parameters.batch_sz = 32
parameters.vertical_flip = False
parameters.horizontal_flip = False
parameters.image_size = autoencoder_model.img_sz
parameters.attributes = autoencoder_model.attr
parameters.num_attributes = autoencoder_model.n_attr
if not (len(parameters.attributes) == 1 and parameters.num_attributes == 2):
    raise Exception("Model must use a single boolean attribute.")

# Load data
images_data, attr_data = load_dataset(parameters)
test_images = SampleData(images_data[2], attr_data[2], parameters)


def generate_interpolations(autoencoder, img_batch, attr_batch, params):
    """
    Image reconstruction and interpolation generation.
    """
    assert len(img_batch) == len(attr_batch)
    encoded_outputs = autoencoder.encode(img_batch)

    # Interpolation values
    alpha_vals = np.linspace(1 - params.min_alpha, params.max_alpha, params.num_interpolations)
    alpha_vals = [torch.FloatTensor([1 - alpha, alpha]) for alpha in alpha_vals]

    # Generate outputs
    generated_outputs = []
    generated_outputs.append(img_batch)
    generated_outputs.append(autoencoder.decode(encoded_outputs, attr_batch)[-1])
    for alpha in alpha_vals:
        alpha_var = Variable(alpha.unsqueeze(0).expand((len(img_batch), 2)).cuda())
        generated_outputs.append(autoencoder.decode(encoded_outputs, alpha_var)[-1])

    # Stack images for output
    return torch.cat([x.unsqueeze(1) for x in generated_outputs], 1).data.cpu()


interpolated_images = []

for batch_start in range(0, parameters.num_images, 100):
    i = parameters.start_index + batch_start
    j = parameters.start_index + min(parameters.num_images, batch_start + 100)
    img_subset, attr_subset = test_images.eval_batch(i, j)
    interpolated_images.append(generate_interpolations(autoencoder_model, img_subset, attr_subset, parameters))

interpolated_images = torch.cat(interpolated_images, 0)
assert interpolated_images.size() == (parameters.num_images, 2 + parameters.num_interpolations,
                                      3, parameters.image_size, parameters.image_size)


def create_image_grid(image_set, grid_horizontal, grid_dimension=5):
    """
    Generate a grid of all images.
    """
 # TODO


# Create image grid and save as PNG
#TODO