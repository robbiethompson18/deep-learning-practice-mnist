import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self, hidden_sizes=[128, 64], dropout_rate=0.2, norm_type='batch'):
        super(MNISTNet, self).__init__()
        self.norm_type = norm_type
        
        # Input layer
        self.fc1 = nn.Linear(784, hidden_sizes[0])
        
        # Hidden layers
        layers = []
        for i in range(len(hidden_sizes)-1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(hidden_sizes[i+1]))
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_sizes[-1], 10)
        
        # Input normalization
        if norm_type == 'batch':
            self.input_norm = nn.BatchNorm1d(784)
        elif norm_type == 'layer':
            self.input_norm = nn.LayerNorm(784)
        else:
            self.input_norm = None
    
    def forward(self, x):
        x = x.view(-1, 784)
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = F.relu(self.fc1(x))
        x = self.hidden_layers(x)
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)

# class MNISTNet(MNISTBaseNet):
#     def __init__(self, hidden_sizes=[128, 64], dropout_rate=0.2):
#         super(MNISTNet, self).__init__(hidden_sizes=hidden_sizes, dropout_rate=dropout_rate, norm_type='batch')

# class MNISTNetWithoutBN(MNISTBaseNet):
#     def __init__(self, hidden_sizes=[128, 64], dropout_rate=0.2):
#         super(MNISTNetWithoutBN, self).__init__(hidden_sizes=hidden_sizes, dropout_rate=dropout_rate, norm_type='none')

# class MNISTNetLayerNorm(MNISTBaseNet):
#     def __init__(self, hidden_sizes=[128, 64], dropout_rate=0.2):
#         super(MNISTNetLayerNorm, self).__init__(hidden_sizes=hidden_sizes, dropout_rate=dropout_rate, norm_type='layer')

# class MNISTNetNoDroput(MNISTBaseNet):
#     def __init__(self, hidden_sizes=[128, 64], dropout_rate=0.2):
#         super(MNISTNetLayerNorm, self).__init__(hidden_sizes=hidden_sizes, dropout_rate=0, norm_type='layer')