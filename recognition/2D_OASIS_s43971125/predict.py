# Contains example usage of the model

import torch
from modules import UNet
from dataset import OASISDataset
import matplotlib.pyplot as plt
from train import dice_coefficient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = UNet(in_channels=1, out_channels=4).to(device)
#Note that in this instance, the trained model is named "best.pth" and is located in the same
#directory as predict.py - by default, this is what train.py will produce. 
model.load_state_dict(torch.load("best.pth"))
model.eval()

# Load a single test example
test_dataset = OASISDataset(split="test")
image, label = test_dataset[0]

#Add batch dimension and send image and label to the GPU
image = image.to(device).unsqueeze(0)  # Add batch dimension
label = label.to(device).unsqueeze(0)

# Forward pass
with torch.no_grad():
    output = model(image)
    probs = torch.softmax(output, dim=1)
    predicted = torch.argmax(output, dim=1)

#Compute Dice per class
dice_scores = dice_coefficient(output, label, num_classes=4)
print("Dice per Class:", dice_scores)

#Visualise results
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Input")
plt.imshow(image[0,0].cpu(), cmap='gray')
plt.subplot(1,3,2)
plt.title("Ground Truth")
plt.imshow(label.cpu(), cmap='jet')
plt.subplot(1,3,3)
plt.title("Prediction")
plt.imshow(predicted[0].cpu(), cmap='jet')
plt.show()

