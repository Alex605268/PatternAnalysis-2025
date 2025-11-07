# Contains the source code for training and validation

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

from modules import UNet
from dataset import get_dataloaders

#Create the DiceLoss functionality, as explained in the lectures
class DiceLoss(nn.Module):
    #Subclass behaves like other Pytorch Loss functions
    #smooth is a small constant - used in the calculation step to avoid potential division by zero
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

#Same as above, but this is used during the evaluation step, while the DiceLoss is used for training
#Calculates dice_coefficient per class
def dice_coefficient(preds, targets, num_classes=4, smooth=1e-6):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)

    dice_scores = []
    # for each class, compute the dice score, then add them to a tensor.
    for i in range(num_classes):
        pred_i = (preds == i).float()
        target_i = (targets == i).float()

        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    #return the tensor that contains the dice score for each class
    return torch.tensor(dice_scores, device=preds.device)

#Note: wrap the training loop so it's only run when train.py is called, not when it's imported
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    # Set the training paremeters
    num_epochs = 25 
    batch_size = 4
    learning_rate = 1e-4
    save_path = "best_improved.pth"

    # Load in the datasets
    train_load, val_load, test_load = get_dataloaders(batch_size=batch_size)

    # Define the model, loss (DiceLoss) and optimiser (Adam)

    model = ImprovedUNet(in_channels=1, out_channels=4).to(device)
    criterion = DiceLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)


    # set the initial best dice score to 0. Gets updated each time the model
    # finds a better score
    best_val_dice = 0.0

    # Training loop
    # Note all print statements have no impact on training, are optional to track progress
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
        print(f"Epoch {epoch+1} finished, Avg Loss: {avg_loss:.4f}")

        #Validation Step
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for images, labels in val_load:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                dice_batch = dice_coefficient(outputs, labels, num_classes=4)
                dice_scores.append(dice_batch)

        dice_scores = torch.stack(dice_scores)
        mean_dice = dice_scores.mean(dim=0)
    
        # Compute overall mean Dice score
        mean_val_dice = mean_dice.mean().item()

        # Print per-class results and average results
        print(f"Validation Dice per class: {mean_dice.cpu().numpy()}")
        print(f"Mean Validation Dice: {mean_val_dice:.4f}")

        # Mean probability for each class
        probs = torch.softmax(outputs, dim=1)
        mean_probs = probs.mean(dim=(0, 2, 3))
        print(f"Mean predicted probabilities per class: {mean_probs.cpu().numpy()}")


        # Save the best model so far, based on the average Dice score
        if mean_val_dice > best_val_dice:
            torch.save(model.state_dict(), save_path)
            best_val_dice = mean_val_dice
            best_per_class = mean_dice.cpu().numpy()
            print(f"✅ New best model saved (Avg Dice: {mean_val_dice:.4f}, per class: {best_per_class})")

        #blank line to separate each epoch printout in the log
        print()

    # Final test evaluation to find Dice Coefficient for each class
    model.load_state_dict(torch.load(save_path))
    model.eval()
    dice_totals = torch.zeros(4, device=device)
    with torch.no_grad():
        for images, labels in test_load:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            dice_scores = dice_coefficient(outputs, labels, num_classes=4)
            dice_totals += dice_scores.to(device)
    dice_averages = dice_totals / len(test_load)

    for i, dice in enumerate(dice_averages):
        print(f"Class {i} Dice: {dice:.4f}")
