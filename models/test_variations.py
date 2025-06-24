import numpy as np
from scipy import stats
from mnist_comparison import compare_models
from model_config import ModelConfig
import torch

def calculate_confidence_interval(accuracies, confidence=0.95):
    """Calculate confidence interval for mean accuracy."""
    n = len(accuracies)
    mean = np.mean(accuracies)
    se = stats.sem(accuracies)
    ci = stats.t.interval(confidence, n-1, mean, se)
    return mean, ci

def test_model_variations():
    BATCH_SIZE = 32
    EPOCHS = 12
    N_TRIALS = 5  # Number of trials with different random seeds
    
    configs = [
        ModelConfig(
            name="Baseline",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS
        ),
        ModelConfig(
            name="No Dropout",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            dropout_rate=0.0
        ),
        ModelConfig(
            name="Adam Optimizer",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            optimizer='Adam'
        ),
        ModelConfig(
            name="Layer Norm",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            norm_type='layer'
        ),
        ModelConfig(
            name="Extra Hidden Layer",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            hidden_sizes=[128, 64, 64]
        ),
        ModelConfig(
            name="Double Width",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            hidden_sizes=[256, 128]
        ),
        ModelConfig(
            name="Half Width",
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            hidden_sizes=[128, 64]
        )
    ]
    
    # Store results for each model across trials
    results = {config.name: [] for config in configs}
    
    # Run multiple trials
    for trial in range(N_TRIALS):
        print(f"\n=== Trial {trial + 1}/{N_TRIALS} ===")
        # Set random seeds
        seed = trial * 100
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
        # Train and test each configuration
        for config in configs:
            print(f"\nTesting {config.name}")
            state = compare_models(
                [configs[0], config],  # Compare with baseline
                return_results=True
            )
            results[config.name].append(state.best_test_acc)
    
    # Calculate statistics
    print("\n=== Final Results (95% Confidence Intervals) ===")
    for name, accuracies in results.items():
        accuracies = np.array(accuracies)
        mean, (ci_low, ci_high) = calculate_confidence_interval(accuracies)
        print(f"\n{name}:")
        print(f"  Mean Accuracy: {mean:.2f}%")
        print(f"  95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]")

    # Compare all configurations
    compare_models(configs)

if __name__ == '__main__':
    test_model_variations()
