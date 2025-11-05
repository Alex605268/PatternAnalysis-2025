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
        #preds = predictions, in form [BatchSize, Classes, Height, Width] = [B, C, H, W]
        #targets are in form [B, H, W]

        #torch.softmax turns the UNet output scores into probabilities, which sum to 1
        preds = torch.softmax(preds, dim=1)

        #Convert targets to one-hot encoding. This ensures that when we flatten it into
        # a 1D tensor, it's the same size as the predictions tensor, so we can calculate
        #the intersection.
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1])
        # This line swaps the order around, so it matches preds order (which is the just
        # the raw output of UNet, which gives [BatchSize, NumClasses, ImageHeight, ImageWidth]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  

        #Calculate the Dice score for each class
        intersection = (preds * targets_onehot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2,3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        #Subtract the mean dice score across all classes from 1 to find the average DiceLoss
        loss = 1 - dice_score.mean()
        return loss

def dice_coefficient(preds, targets, smooth=1e-6):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Set the training paremeters
num_epochs = 1
batch_size = 4
learning_rate = 1e-4
save_path = "best.pth"

# Load in the datasets
train_load, val_load, test_load = get_dataloaders(batch_size=batch_size)

# Define the model, loss (DiceLoss) and optimiser (Adam)

model = UNet(in_channels=1, out_channels=4).to(device)
criterion = DiceLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)


best_val_dice = 50
# Don't need below currently, only saving best model. May be used if I want all models.
#os.makedirs(checkpoint_dir, exist_ok=True)

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
        torch.save(model.state_dict(), save_path)
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

