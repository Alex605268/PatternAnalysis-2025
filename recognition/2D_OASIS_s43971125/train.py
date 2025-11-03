# Contains the source code for training and validation

import torch
import torch.nn as nn
import torch.optim as optim
import os

from modules import UNet
from dataset import get_dataloaders

#Create the DiceLoss functionality
class DiceLoss(nn.Module):
    #Subclass behaves like other Pytorch Loss functions
    #smooth is a small constant - used in the calculation step to avoid division by zero
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        #torch.softmax turns the UNet output scores into probabilities, which sum to 1
        preds = torch.softmax(preds, dim=1)
        #flatten the predictions and labels into 1D tensors
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
        #find the intersection between predicted probabilities and true values
        intersection = (preds_flat * targets_flat).sum()
        #Calculate the Dice Coefficient and subtract it from 1 to find the Dice Loss
        return 1 - ((2. * intersection + self.smooth) /
                    (preds_flat.sum() + targets_flat.sum() + self.smooth))

def dice_coefficient(preds, targets, smooth=1e-6):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the training paremeters
num_epochs = 1
batch_size = 4
learning_rate = 1e-4
save_path = "best.pth"

# Load in the datasets
train_load, val_load, test_load = get_dataloaders(batch_size=batch_size)

# Define the model, loss (DiceLoss) and optimiser (Adam)

model = UNet(in_channels=1, out_channels=3).to(device)
criterion = DiceLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)


best_val_dice = 50
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_load:
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_load)

    #Validation Step
    model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for images, labels in val_load:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_dice += dice_coefficient(outputs, labels)

    val_dice /= len(val_load)

    # Save the best model found so far
    if val_dice > best_val_dice:
        torch.save(model.state_dict, save_path)
        best_val_dice = val_dice


# Final test evaluation to find Dice Coefficient

model.load_state_dict(torch.load(save_path))
model.eval()
test_dice = 0.0
with torch.no_grad():
    for images, labels in test_load:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_dice += dice_coefficient(outputs, labels)
    test_dice /= len(test_load)

