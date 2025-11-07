# Contains example usage of the model

import torch
from modules import UNet
from modules import ImprovedUNet
from dataset import OASISDataset
import matplotlib.pyplot as plt
from train import dice_coefficient

def load_model(model_path="best_improved.pth", in_channels=1, out_channels=4, device=None):
    """
    Loads a trained UNet model from the specified checkpoint path.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImprovedUNet(in_channels=in_channels, out_channels=out_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def predict_and_visualise(model, dataset_split="test", sample_index=0, save_path="prediction_example.png", num_classes=4, device=None):
    """
    Runs inference on one sample from the test dataset and visualises input, ground truth, and output (prediction). Saves the
    generated image to save_path.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load one sample from dataset
    dataset = OASISDataset(split=dataset_split)
    image, label = dataset[sample_index]

    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1)

    # Compute Dice per class
    dice_scores = dice_coefficient(output, label, num_classes=num_classes)
    print(f"Dice per Class (sample {sample_index}):", dice_scores)

    # Visualise results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(image[0, 0].cpu(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(label.squeeze().cpu(), cmap='jet')

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(predicted[0].cpu(), cmap='jet')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction visualisation to {save_path}")

def predict_and_visualise_per_class(model, dataset_split="test", sample_index=0, save_path="prediction_per_class.png", num_classes=4, device=None):
    """
    Runs inference on one sample from the dataset and visualises input, ground truth,
    final predicted segmentation, and per-class probability maps.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load one sample from dataset
    dataset = OASISDataset(split=dataset_split)
    image, label = dataset[sample_index]

    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)  # [B, C, H, W]
        predicted = torch.argmax(output, dim=1)

    # Compute Dice per class
    dice_scores = dice_coefficient(output, label, num_classes=num_classes)
    print(f"Dice per Class (sample {sample_index}):", dice_scores)

    # Visualise results
    n_cols = 3 + num_classes  # input + GT + overall prediction + per-class
    plt.figure(figsize=(4 * n_cols, 4))

    # Each Class Probability Map
    for c in range(num_classes):
        plt.subplot(1, n_cols, 4 + c)
        plt.title(f"Class {c} Prob")
        plt.imshow(probs[0, c].cpu(), cmap='inferno')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualisation to {save_path}")


if __name__ == "__main__":
    model, device = load_model()
    predict_and_visualise(model, sample_index=5, save_path="prediction_example.png", device=device)
    predict_and_visualise_per_class(model, sample_index=5, save_path="prediction_per_class.png", device=device)




