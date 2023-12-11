import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np


class CelebADataset(Dataset):
    def __init__(self, img_dir, annotations_file,name_attr, img_ids, target_size=(256, 256)):
        self.img_dir = img_dir
        self.transform = transforms.Compose([ transforms.ToTensor()])
        self.target_size = target_size
        
        self.annotations = pd.read_csv(annotations_file, index_col=0)
        self.annotations.replace(-1, 0, inplace=True)

        # Filtrer les annotations pour les images spécifiées dans img_ids

        self.annotations = self.annotations.loc[img_ids]
        self.attributes = 1- pd.get_dummies(self.annotations[name_attr], prefix=name_attr)
        self.path_images = self.annotations.index

    def __len__(self):
        return len(self.annotations)

    def preprocess_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.crop((0, 0, 178, 178))
            img = img.resize(self.target_size)
            img = np.array(img, dtype=np.float32) / 255
            img = img*2 -1
        return img
      
    def __getitem__(self, idx):
        img_name = self.path_images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = self.preprocess_image(img_path)
        if self.transform:
            image = self.transform(image)

        # Transformation des attributs one-hot en tensor
        attributes = torch.tensor(self.attributes.iloc[idx].values, dtype=torch.float32)

        return image, attributes
