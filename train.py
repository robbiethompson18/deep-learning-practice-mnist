import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.pytorch_nn import MNISTNet

def load_data(batch_size=64):
    """
    Load MNIST dataset and create data loaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download training data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Download test data
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to train on (cpu/cuda)
    Returns:
        float: Average training loss for this epoch
    """
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f'Train Batch [{batch_idx + 1}/{total_batches}] Loss: {loss.item():.4f}')

    # Calculate average loss
    avg_loss = running_loss / total_batches
    print(f'Training Average Loss: {avg_loss:.4f}')
    return avg_loss

def test(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    Returns:
        float: Average loss on test set
        float: Accuracy on test set
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to track gradients for testing
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            test_loss += criterion(output, target).item()
            
            # Get predictions
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # Average loss and accuracy
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return test_loss, accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    epochs = 10

    # Load data
    train_loader, test_loader = load_data(batch_size)

    # Initialize model, loss, and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train(model, train_loader, optimizer, criterion, device)
        test(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()
