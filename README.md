# FaderNetworks

PyTorch implementation of [Fader Networks](https://arxiv.org/pdf/1706.00409.pdf) (NIPS 2017).

Fader Networks can generate different realistic versions of images by modifying attributes such as gender or age group. They can swap multiple attributes at a time, and continuously interpolate between each attribute value. In this repository we provide the code to reproduce the results presented in the paper, as well as trained models.

### Single-attribute swap

Below are some examples of different attribute swaps:

## Model

The main branch of the model (Inference Model), is an autoencoder of images. Given an image `x` and an attribute `y` (e.g. male/female), the decoder is trained to reconstruct the image from the latent state `E(x)` and `y`. The other branch (Adversarial Component), is composed of a discriminator trained to predict the attribute from the latent state. The encoder of the Inference Model is trained not only to reconstruct the image, but also to fool the discriminator, by removing from `E(x)` the information related to the attribute. As a result, the decoder needs to consider `y` to properly reconstruct the image. During training, the model is trained using real attribute values, but at test time, `y` can be manipulated to generate variations of the original image.

## Dependencies
* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [PyTorch](http://pytorch.org/)
* OpenCV
* CUDA


## Installation

Simply clone the repository:

```bash
git clone https://github.com/MassiliaHe/MLA-project-.git
cd MLA-project-
```

## Dataset
Download the aligned and cropped CelebA dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Extract all images and move them to the `data/img_align_celeba/` folder. There should be 202599 images. The dataset also provides a file `list_attr_celeba.txt` containing the list of the 40 attributes associated with each image. Move it to `data/`.


It will resize images, and create 2 files: `images_256_256.pth` and `attributes.pth`. The first one contains a tensor of size `(202599, 3, 256, 256)` containing the concatenation of all resized images. Note that you can update the image size in `preprocess.py` to work with different resolutions. The second file is a pre-processed version of the attributes.


## Train your own models

### Train a classifier
To train your own model you first need to train a classifier to let the model evaluate the swap quality during the training. Training a good classifier is relatively simple for most attributes, and a good model can be trained in a few minutes. We provide a trained classifier for all attributes in `models/classifier256.pth`. Note that the classifier does not need to be state-of-the-art, it is not used during the training process, but is just here to monitor the swap quality. If you want to train your own classifier, you can run `classifier.py`, using the following parameters:


```bash
python classifier.py

```


### Train a Fader Network

You can train a Fader Network with `train.py`.

## Generate interpolations

Given a trained model, you can use it to swap attributes of images in the dataset. Below are examples using the pretrained models:

```bash

# Eyeglasses
python interpolate.py --model_path models/eyeglasses.pth --n_images 10 --n_interpolations 10 --alpha_min 2.0 --alpha_max 2.0 --output_path eyeglasses.png

```

These commands will generate images with 10 rows of 12 columns with the interpolated images. The first column corresponds to the original image, the second is the reconstructed image (without alteration of the attribute), and the remaining ones correspond to the interpolated images. `alpha_min` and `alpha_max` represent the range of the interpolation. Values superior to 1 represent generations over the True / False range of the boolean attribute in the model. Note that the variations of some attributes may only be noticeable for high values of alphas. For instance, for the "eyeglasses" or "gender" attributes, alpha_max=2 is usually enough, while for the "age" or "narrow eyes" attributes, it is better to go up to alpha_max=10.
