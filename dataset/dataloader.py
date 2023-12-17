import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader


class CelebADataset(Dataset):
    def __init__(self, img_dir, annotations_file, name_attr, img_ids, target_size=(256, 256)):
        self.img_dir = img_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.target_size = target_size

        self.annotations = pd.read_csv(annotations_file, index_col=0)
        self.annotations.replace(-1, 0, inplace=True)

        # Filtrer les annotations pour les images spécifiées dans img_ids

        self.annotations = self.annotations.loc[img_ids]
        self.attributes = 1 - pd.get_dummies(self.annotations[name_attr], prefix=name_attr)
        self.path_images = self.annotations.index

    def __len__(self):
        return len(self.annotations)

    def preprocess_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.crop((0, 0, 178, 178))
            img = img.resize(self.target_size)
            img = np.array(img, dtype=np.float32) / 255
            img = img*2 - 1
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


# Function to split the data
def split_data(annotations_file):
    id_status = []
    with open(annotations_file, 'r') as g:
        id_status = [line.strip().split(',') for line in g]

    datasets_ids = {"train": [], "validation": [], "test": []}
    for id in id_status:
        if id[1] == '0':
            datasets_ids["train"].append(id[0])
        if id[1] == '1':
            datasets_ids["validation"].append(id[0])
        if id[1] == '2':
            datasets_ids["test"].append(id[0])

    print("The train dataset has:", len(datasets_ids["train"]))
    print("The validation dataset has:", len(datasets_ids["validation"]))
    print("The test dataset has:", len(datasets_ids["test"]))

    return datasets_ids


def get_dataloaders(data_path, name_attr='Young', batch_size=32):
    datasets_ids = split_data(os.path.join(data_path, 'list_eval_partition.csv'))
    # Init datasets partitions
    images_path = os.path.join(data_path, 'img_align_celeba')
    list_attr_path = os.path.join(data_path, 'list_attr_celeba.csv')
    celeba_train_dataset = CelebADataset(img_dir=images_path, annotations_file=list_attr_path,
                                         name_attr=name_attr, img_ids=datasets_ids["train"], target_size=(256, 256))
    celeba_val_dataset = CelebADataset(img_dir=images_path, annotations_file=list_attr_path,
                                       name_attr=name_attr, img_ids=datasets_ids["validation"], target_size=(256, 256))
    celeba_test_dataset = CelebADataset(img_dir=images_path, annotations_file=list_attr_path,
                                        name_attr=name_attr, img_ids=datasets_ids["test"], target_size=(256, 256))

    # Create DataLoader
    train_dataloader = DataLoader(dataset=celeba_train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=celeba_val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=celeba_test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
