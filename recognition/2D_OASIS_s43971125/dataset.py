# Contains the data loader

import os
import torch
import torch.utils.data import Dataset, Dataloader
import nibabel as nib
import numpy as np


class OASISDataset(Dataset):
    def __init__(self, root_dir="/home/groups/comp3710/OASIS", split="train"):
        self.root_dir = root_dir
        self.split=split
        # filepaths go here
        self.image_paths=[]
        self.label_paths=[]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load images and labels
        image = nib.load(self.image_paths[idx]).get_fdata()
        label = nib.load(self.label_paths[idx]).get_fdata()

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_dataloaders(batch_size=4):
    # Returns the pytorch dataloaders for our 3 datasets
    train_dataset = OASISDataset(split="train")
    val_dataset = OASISDataset(split="val")
    test_dataset = OASISDataset(split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
