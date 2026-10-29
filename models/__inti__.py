"""
Factory function for model creation.
Both server.py and client.py use get_model(config)
to create the correct model for the current experiment.
"""

from models.simple_cnn   import SimpleCNN
from models.resnet_cifar import ResNetCIFAR


def get_model(config: dict):
    """
    Create and return the correct model for this experiment.

    Model is determined by config['model'] which is set
    automatically by the dataset config functions:
        mnist    → SimpleCNN(num_classes=10)
        femnist  → SimpleCNN(num_classes=62)
        cifar10  → ResNetCIFAR(num_classes=10)
        cifar100 → ResNetCIFAR(num_classes=100)

    Args:
        config: Experiment config dict from config/config.py

    Returns:
        Initialized PyTorch model (weights are random)

    Raises:
        ValueError: If config['model'] is not recognized
    """
    model_name  = config['model']
    num_classes = config['num_classes']

    if model_name == 'SimpleCNN':
        return SimpleCNN(num_classes=num_classes)

    elif model_name == 'ResNetCIFAR':
        return ResNetCIFAR(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: ['SimpleCNN', 'ResNetCIFAR']"
        )


def count_parameters(model) -> int:
    """
    Count total trainable parameters in a model.
    Useful for reporting model size in the paper.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
