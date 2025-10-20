# Contains example usage of the model

import torch
from model import build_unet
from dataset import OASISDataset
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = build_unet(in_channels=1, out_channels=3).to(device)
model.load_state_dict(torch.load("../checkpoints/unet_final.pth"))
model.eval()

# Load example
dataset = OASISDataset(split="test")
image, label = dataset[0]
image = image.to(device).unsqueeze(0)  # Add batch dimension

# Predict segmentation
with torch.no_grad():
    output = model(image)
    predicted = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()


#TODO: Visualisation
