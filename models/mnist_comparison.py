import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import ssl

from pytorch_nn import MNISTNet
from model_config import ModelConfig, TrainingState, get_save_path

# Bypass SSL certificate verification for MNIST download
ssl._create_default_https_context = ssl._create_unverified_context

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in tqdm(train_loader, desc='Training', leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(test_loader)
    return avg_loss, accuracy

def train_model(config: ModelConfig) -> TrainingState:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.norm_mean, config.norm_std)
    ])
    
    # Load datasets
    print('Loading MNIST dataset...')
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model based on normalization type
    # if config.norm_type == 'batch':
    model_class = MNISTNet
    # elif config.norm_type == 'layer':
        # model_class = MNISTNetLayerNorm
    # else:
        # model_class = MNISTNetWithoutBN
    
    model = model_class(
        hidden_sizes=config.hidden_sizes,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    if config.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f'Unsupported optimizer: {config.optimizer}')
    
    # Initialize training state
    state = TrainingState(config)
    start_time = time.time()
    
    # Training loop
    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        state.history['train_loss'].append(train_loss)
        state.history['train_acc'].append(train_acc)
        
        # Evaluate
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        state.history['test_loss'].append(test_loss)
        state.history['test_acc'].append(test_acc)
        
        # Update best accuracy
        if test_acc > state.best_test_acc:
            state.best_test_acc = test_acc
        
        print(f'Epoch {epoch+1}/{config.epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'  Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    
    # Save final model state and training time
    state.model_state = model.state_dict()
    state.training_time = time.time() - start_time
    
    return state

def load_or_train(config: ModelConfig) -> TrainingState:
    save_path = get_save_path(config)
    
    if os.path.exists(save_path):
        print(f'Loading existing model {config.get_unique_id()}')
        return TrainingState.load(save_path)
    
    print(f'Training new model {config.get_unique_id()}')
    state = train_model(config)
    state.save(save_path)
    return state

def compare_models(model_config_list):
    """Compare models with and without batch normalization.
    
    Args:
        model_config_list: List of model configurations.
        return_results: If True, return the model state for statistical analysis.
    """
    
    # Train or load both models
    trained_models = [load_or_train(model_config) for model_config in model_config_list]
    max_epochs = max(model_config.epochs for model_config in model_config_list)
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    epochs_range = range(1, max_epochs + 1)
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for model, config in zip(trained_models, model_config_list):
        plt.plot(epochs_range, model.history['train_loss'],  label=config.name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(1, 2, 2)
    for model, config in zip(trained_models, model_config_list):
        plt.plot(epochs_range, model.history['train_acc'], label=config.name)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/training_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.title('Last Epoch Training Loss vs Test Accuracy')
    plt.xlabel('Last Epoch Training Loss')
    plt.ylabel('Test Accuracy (%)')
    
    # Get data points
    x = [model.history['train_loss'][-1] for model in trained_models]
    y = [model.best_test_acc for model in trained_models]
    labels = [config.name for config in model_config_list]
    
    # Create scatter plot with labeled points
    scatter = plt.scatter(x, y, s=100)
    
    # Add labels with adjusted positions
    for i, label in enumerate(labels):
        plt.annotate(label,
                    (x[i], y[i]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5',
                             fc='yellow',
                             alpha=0.5),
                    arrowprops=dict(arrowstyle='->',
                                  connectionstyle='arc3,rad=0'))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save first, then show
    plt.savefig('plots/last_epoch_training_loss_vs_test_accuracy.png',
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()

    # Print final results
    print('\nFinal Results:')
    for model, config in zip(trained_models, model_config_list):
        print(f'  {config.name}:')
        print(f'  Best Test Accuracy: {model.best_test_acc:.2f}%')
        print(f'  Training Time: {model.training_time:.1f}s')

if __name__ == '__main__':
    compare_models()
