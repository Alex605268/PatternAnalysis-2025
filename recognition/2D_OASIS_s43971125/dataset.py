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

    #Functions copied from Assignment - will be used if switch from easy to normal task
    #Unnecessary for OASIS dataset as it is already in PNG format
    def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
        channels = np.unique(arr)
        res = np.zeros(arr.shape + (len(channels),),dtype=dtype)
        for c in channels:
            c = int(c)
            res[..., c:c+1][arr == c] = 1

        return res

    def load_data_2D(imageNames, normImage=False, categorical=False, 
            dtype=np.float32, getAffines=False, early_stop=False):
        affines = []

        num = len(imageNames)
        first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
        if len(first_case.shape) == 3:
            first_case = first_case[:,:,0] #remove extra dimension
        if categorical:
            first_case = to_channels(first_case, dtype=dtype)
            rows, cols, channels = first_case.shape
            images = np.zeros((num, rows, cols, channels), dtype=dtype)
        else:
            rows, cols = first_case.shape
            images = np.zeros((num, rows, cols), dtype=dtype)

        for i, inName in enumerate(tqdm(imageNames)):
            niftiImage = nib.load(inName)
            inImage = niftiImage.get_fdata(caching='unchanged')
            affine = niftiImage.affine
            if len(inImage.shape) == 3:
            inImage = inImage[:,:,0]
            inImage = inImage.astype(dtype)
            if normImage:
                inImage = (inImage - inImage.mean()) / inImage.std()
            if categorical:
                inImage = utils.to_channels(inImage, dtype=dtype)
                images[i,:,:,:] = inImage
            else:
                images[i,:,:] = inIMage
            affines.append(affine)
            if i > 20 and early_stop:
                break
        if getAffines:
            return images, affines
        else:
            return images
