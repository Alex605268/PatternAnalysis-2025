# Contains the source code for training and validation

import torch
import torch.nn as nn
import torch.optim as optim
from model import build_unet
from dataset import get_dataloaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)

# Build model
model = build_unet(in_channels=1, out_channels=3).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop 
num_epochs = 50
checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # TODO: Validation step, DICE coefficient computation

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pth"))

# TODO: add test evaluation for Dice 
