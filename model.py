import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class MNISTNet(nn.Module):
    """Convolutional Neural Network for MNIST digit classification."""

    def __init__(self):
        super(MNISTNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # First conv block: conv -> relu -> pool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # Second conv block: conv -> relu -> pool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for MNIST classification.

    Args:
        image: PIL Image (can be any size, will be resized)

    Returns:
        Tensor of shape (1, 1, 28, 28) ready for model input
    """
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    # Invert if background is white (MNIST has black background)
    if img_array.mean() > 127:
        img_array = 255 - img_array
    # Normalize to [0, 1] then apply MNIST normalization
    img_array = img_array / 255.0
    img_array = (img_array - 0.1307) / 0.3081
    # Convert to tensor and add batch and channel dimensions
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return tensor
