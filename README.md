# MNIST Digit Recognition Project

This project implements a neural network for recognizing handwritten digits using the MNIST dataset.

## Project Structure
```
mnist_project/
├── requirements.txt
├── README.md
├── data/           # MNIST dataset (downloaded automatically)
├── models/         # Neural network implementations
│   ├── pytorch_nn.py
│   └── manual_nn.py (future)
└── train.py       # Training script
```

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the training script:
```bash
python train.py
```

## Implementation Details
- Phase 1: PyTorch implementation
- Phase 2: Manual implementation with NumPy (coming soon)
