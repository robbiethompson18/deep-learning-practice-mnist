import os
import json
import hashlib
import torch

class ModelConfig:
    def __init__(self, name="MODEL NOT NAMED", hidden_sizes=[128, 64], dropout_rate=0.2, norm_type='batch',
                 learning_rate=0.01, batch_size=64, epochs=5, optimizer='SGD',
                 norm_mean=(0.1307,), norm_std=(0.3081,)):
        self.name = name
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.norm_mean = norm_mean
        self.norm_std = norm_std
    
    def get_unique_id(self):
        """Generate a unique ID for this configuration."""
        config_str = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:10]
    
    def to_dict(self):
        """Convert config to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    # @classmethod
    # def default_with_bn(cls):
    #     """Create default configuration with batch normalization."""
    #     return cls(norm_type='batch')
    
    # @classmethod
    # def default_without_bn(cls):
    #     """Create default configuration without batch normalization."""
    #     return cls(norm_type='none')
    
    # @classmethod
    # def default_with_ln(cls):
    #     """Create default configuration with layer normalization."""
    #     return cls(norm_type='layer')
    
    # @classmethod
    # def no_dropout(cls):
    #     """Create configuration with no dropout."""
    #     return cls(dropout_rate=0.0)
    
    # @classmethod
    # def with_adam(cls):
    #     """Create configuration with Adam optimizer."""
    #     return cls(optimizer='Adam')
    
    # @classmethod
    # def extra_hidden_layer(cls):
    #     """Create configuration with an extra hidden layer."""
    #     return cls(hidden_sizes=[128, 64, 64])
    
    # @classmethod
    # def double_width(cls):
    #     """Create configuration with double width."""
    #     return cls(hidden_sizes=[256, 128])
    
    # @classmethod
    # def default_with_bn(cls):
    #     """Create default configuration with batch normalization."""
    #     return cls(
    #         hidden_sizes=[128, 64],
    #         dropout_rate=0.2,
    #         norm_type='batch',
    #         learning_rate=0.01,
    #         batch_size=64,
    #         epochs=5,
    #         optimizer='SGD',
    #         norm_mean=(0.1307,),
    #         norm_std=(0.3081,)
    #     )
    
    # @classmethod
    # def default_without_bn(cls):
    #     """Create default configuration without batch normalization."""
    #     return cls(
    #         hidden_sizes=[128, 64],
    #         dropout_rate=0.2,
    #         norm_type='none',
    #         use_batch_norm=False,
    #         learning_rate=0.01,
    #         batch_size=64,
    #         epochs=5,
    #         optimizer='SGD',
    #         norm_mean=(0.1307,),
    #         norm_std=(0.3081,)
    #     )
    
    # @classmethod
    # def default_with_ln(cls):
    #     """Create default configuration with layer normalization."""
    #     return cls(
    #         hidden_sizes=[128, 64],
    #         dropout_rate=0.2,
    #         norm_type='layer',
    #         learning_rate=0.01,
    #         batch_size=64,
    #         epochs=5,
    #         optimizer='SGD',
    #         norm_mean=(0.1307,),
    #         norm_std=(0.3081,)
    #     )
    
    # @classmethod
    # def no_dropout(cls):
    #     """Create configuration with no dropout."""
    #     return cls(
    #         hidden_sizes=[128, 64],
    #         dropout_rate=0.0,
    #         norm_type='batch',
    #         learning_rate=0.01,
    #         batch_size=64,
    #         epochs=5,
    #         optimizer='SGD',
    #         norm_mean=(0.1307,),
    #         norm_std=(0.3081,)
    #     )
    
    # @classmethod
    # def with_adam(cls):
    #     """Create configuration with Adam optimizer."""
    #     return cls(
    #         hidden_sizes=[128, 64],
    #         dropout_rate=0.2,
    #         norm_type='batch',
    #         learning_rate=0.01,
    #         batch_size=64,
    #         epochs=5,
    #         optimizer='Adam',
    #         norm_mean=(0.1307,),
    #         norm_std=(0.3081,)
    #     )
    
    # @classmethod
    # def extra_hidden_layer(cls):
    #     """Create configuration with an extra hidden layer."""
    #     return cls(
    #         hidden_sizes=[128, 64, 64],
    #         dropout_rate=0.2,
    #         norm_type='batch',
    #         learning_rate=0.01,
    #         batch_size=64,
    #         epochs=5,
    #         optimizer='SGD',
    #         norm_mean=(0.1307,),
    #         norm_std=(0.3081,)
    #     )
    
    # @classmethod
    # def double_width(cls):
    #     """Create configuration with double width."""
    #     return cls(
    #         hidden_sizes=[256, 128],
    #         dropout_rate=0.2,
    #         norm_type='batch',
    #         learning_rate=0.01,
    #         batch_size=64,
    #         epochs=5,
    #         optimizer='SGD',
    #         norm_mean=(0.1307,),
    #         norm_std=(0.3081,)
    #     )

class TrainingState:
    """Stores the complete state of model training."""
    def __init__(self, config):
        self.config = config
        self.model_state = None
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        self.best_test_acc = 0.0
        self.training_time = 0.0
    
    def save(self, path):
        """Save training state to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'config': self.config.to_dict(),
            'model_state': self.model_state,
            'history': self.history,
            'best_test_acc': self.best_test_acc,
            'training_time': self.training_time
        }
        torch.save(state_dict, path)
    
    @classmethod
    def load(cls, path):
        """Load training state from file."""
        state_dict = torch.load(path)
        config = ModelConfig.from_dict(state_dict['config'])
        state = cls(config)
        state.model_state = state_dict['model_state']
        state.history = state_dict['history']
        state.best_test_acc = state_dict['best_test_acc']
        state.training_time = state_dict['training_time']
        return state

def get_save_path(config: ModelConfig) -> str:
    """Get the save path for a model configuration."""
    return os.path.join('saved_models', f"{config.get_unique_id()}.pt")
