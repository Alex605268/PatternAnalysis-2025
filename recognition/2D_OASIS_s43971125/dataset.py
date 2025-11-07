# Contains the data loader


import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import glob

'''
OASISDataset class.
Loads the 2D_OASIS_dataset, converts the images and labels to pytorch tensors,
Remaps the labels to one-hot encoding, so that they are compatible with the DiceLoss function
Optionally transforms the image if 'transform' is set, then returns the images and labels
'''
class OASISDataset(Dataset):
    '''Current setup assumes that 2D_OASIS dataset is located in the specified root_dir
    with the images and labels already split into train, test, and validation directory
    Also assumes that all labels and images are .png files
    /Oasis-
        -keras_png_slices_train
        -keras_png_slices_test
        -keras_png_slices_val
        -keras_png_slices_seg_train
        -keras_png_slices_seg_test
        -keras-png_slices_seg_val
    If there is a different location, need to adjust root_dir, image_dir and label_dir accordingly
    '''
    def __init__(self, root_dir="/home/groups/comp3710/OASIS", split="train", categorical=False, transform=None):
        
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

        # Convert to np arrays
        image_np = np.array(image)
        label_np = np.array(label, dtype=np.uint8)
        


        # Convert to tensors
        image = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(label_np, dtype=torch.long)

        #remap labels from [0, 85, 170, 255] to [0, 1, 2, 3], so that one-hot encoding of
        #targets (in the diceloss function) works correctly. We do this by just dividing every label
        #by 85, and rounding down.
        label = torch.div(label, 85, rounding_mode='floor')

        #optional transformation step - only used if transform is set in __init__
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

