# sieci_g

Neural network exercises.

## Week 1: Highway Networks on CIFAR-10

The `tydzien 1` folder contains a compact convolutional Highway Network based on the included paper. The notebook includes image samples and examples of correct and incorrect predictions.

- `highway_cifar10.ipynb`: interactive notebook.
- `highway_cifar10.py`: standalone training script.
- `requirements.txt`: Python dependencies.
- `1505.00387v2.pdf`: reference paper.

## Run

Use Python 3.11 or newer. From the repository root:

```bash
cd "tydzien 1"
python3.11 -m venv .venv-mps
source .venv-mps/bin/activate
python -m pip install -r requirements.txt
python highway_cifar10.py
```

For the notebook, select the `.venv-mps` Python environment in VS Code. Training selects MPS on a supported Mac, CUDA when available, or CPU.

CIFAR-10 downloads automatically into `dane`. Virtual environments, datasets, and generated model weights are excluded from Git.
