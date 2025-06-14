import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """
    Neural Network for MNIST digit recognition.
    TODO: Implement the network architecture here.
    
    Suggested structure:
    - Input layer: 784 neurons (28x28 pixels)
    - Hidden layer(s): Your choice
    - Output layer: 10 neurons (digits 0-9)
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        # TODO: Define your layers here
        pass

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10)
        """
        # TODO: Implement the forward pass
        pass
