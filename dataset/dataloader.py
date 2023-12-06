import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Define a custom dataset class
class CelebADataset(Dataset):
    def __init__(self, root_dir, attribute_filename, image_size=(64, 64), normalize=True, transform=None):
        # Set important parameters and initialize
        self.root_dir = root_dir
        self.image_size = image_size
        self.normalize = normalize
        self.transform = transform
        self.attribute_filename = attribute_filename  # Path to attribute file

        # Load dataset (Complete the _load_dataset method)
        self.image_list, self.attribute_labels = self._load_dataset()

    def _load_dataset(self):
        # Load image list and attribute labels from the dataset (Complete this method)
        image_list = sorted(os.listdir(self.root_dir))
        attribute_labels = {}
        
        # TODO

        # Open the attribute file (Complete this part)
        # Iterate through lines in the attribute file (Complete this part)
        # Extract image names and attribute labels (Complete this part)

        return image_list, attribute_labels

    def _create_transform(self):
        # Create image transformations (Complete this method)
        transformations = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ]
        # TODO
        if self.normalize:
            # Normalize based on ImageNet statistics (Complete this part)
            transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]))

        return transforms.Compose(transformations)

    def __len__(self):
        # Return the total number of images in the dataset (Complete this line)
        return len(self.image_list)

    def __getitem__(self, index):
        # Retrieve image and attribute labels for a given index (Complete this part)
        image_name = self.image_list[index]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            # Apply transformations if specified (Complete this part)
            image = self.transform(image)

        # Get the attribute labels for the current image (Complete this part)
        # TODO
        attributes = torch.FloatTensor(self.attribute_labels[image_name])
        # TODO
        # You have to return the right tensor
        return {'image': image, 'attributes': attributes, 'filename': image_name}
    