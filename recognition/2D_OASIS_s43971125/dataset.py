# Contains the data loader


import os
import torch
import torch.utils.data import Dataset, Dataloader
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob

class OASISDataset(Dataset):
    def __init__(self, root_dir="/home/groups/comp3710/OASIS", split="train", categorical=false):
        
        image_dir = os.path.join(root_dir, f"keras_png_slices_{split}")
        label_dir = os.path.join(root_dir, f"keras_png_slices_seg_{split}")
        
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.png")))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load images and labels
        image = Image.open(self.image_paths[idx]).convert("L")  
        label = Image.open(self.label_paths[idx]).convert("L")  

        # Convert to tensors
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(np.array(label), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Helper function that is called by train.py to get all 3 dataloaders
def get_dataloaders(root_dir="/home/groups/comp3710/OASIS", batch_size=4):
    # define the 3 datasets
    train_dataset = OASISDataset(root_dir=root_dir, split="train")
    val_dataset = OASISDataset(root_dir=root_dir, split="validate")
    test_dataset = OASISDataset(root_dir=root_dir, split="test")
    
    #Construct the 3 dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #Return the 3 dataloaders
    return train_loader, val_loader, test_loader

