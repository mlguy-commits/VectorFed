"""
Comprehensive evaluation metrics for federated learning.
Includes accuracy, fairness, convergence, communication cost, and more.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple
import time


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted labels
        labels: True labels
    
    Returns:
        Accuracy (0.0 to 1.0)
    """
    return np.mean(predictions == labels)


def compute_f1_score(predictions: np.ndarray, labels: np.ndarray, average='macro') -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        average: 'macro', 'micro', or 'weighted'
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    return f1_score(labels, predictions, average=average, zero_division=0)


def compute_auroc(
    predictions: np.ndarray, 
    probabilities: np.ndarray, 
    labels: np.ndarray,
    num_classes: int = 10
) -> float:
    """
    Compute Area Under ROC Curve (multi-class).
    
    Args:
        predictions: Predicted labels
        probabilities: Class probabilities (n_samples, n_classes)
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        AUROC score (0.0 to 1.0)
    """
    try:
        # One-vs-rest AUROC
        return roc_auc_score(
            labels, 
            probabilities, 
            multi_class='ovr',
            average='macro'
        )
    except:
        # If AUROC fails (e.g., only one class present), return NaN
        return np.nan


def compute_jains_fairness(contributions: List[float]) -> float:
    """
    Compute Jain's Fairness Index.
    
    Measures how fairly clients contribute to the global model.
    JFI = (sum(x_i))^2 / (n * sum(x_i^2))
    
    Args:
        contributions: List of client contributions (weights or update counts)
    
    Returns:
        Fairness index (0.0 to 1.0, where 1.0 is perfectly fair)
    """
    if len(contributions) == 0:
        return 0.0
    
    contributions = np.array(contributions)
    n = len(contributions)
    
    numerator = np.sum(contributions) ** 2
    denominator = n * np.sum(contributions ** 2)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_communication_cost(
    num_clients: int,
    model_size_bytes: int,
    num_rounds: int
) -> Dict[str, float]:
    """
    Compute communication cost.
    
    Args:
        num_clients: Number of clients
        model_size_bytes: Size of model in bytes
        num_rounds: Number of communication rounds
    
    Returns:
        Dict with communication metrics
    """
    # Each round:
    # - Server sends model to each client (downlink)
    # - Each client sends update to server (uplink)
    
    per_round_cost = model_size_bytes * num_clients * 2  # Upload + download
    total_cost = per_round_cost * num_rounds
    
    return {
        'per_round_mb': per_round_cost / (1024 ** 2),
        'total_mb': total_cost / (1024 ** 2),
        'total_gb': total_cost / (1024 ** 3)
    }


def compute_age_of_information(
    generation_timestamps: List[float],
    aggregation_timestamp: float
) -> Dict[str, float]:
    """
    Compute Age of Information (AoI) metrics.
    
    AoI = time from when update was generated to when it's aggregated.
    
    Args:
        generation_timestamps: When each update was generated
        aggregation_timestamp: When aggregation happened
    
    Returns:
        Dict with AoI metrics
    """
    if not generation_timestamps:
        return {'avg_aoi': 0.0, 'max_aoi': 0.0, 'min_aoi': 0.0}
    
    aois = [aggregation_timestamp - gen_time for gen_time in generation_timestamps]
    
    return {
        'avg_aoi': np.mean(aois),
        'max_aoi': np.max(aois),
        'min_aoi': np.min(aois),
        'std_aoi': np.std(aois)
    }


def detect_convergence(
    accuracy_history: List[float],
    window_size: int = 5,
    threshold: float = 0.01
) -> Tuple[bool, int]:
    """
    Detect if training has converged.
    
    Convergence is detected when accuracy stabilizes (std < threshold over window).
    
    Args:
        accuracy_history: List of accuracy values over rounds
        window_size: Number of recent rounds to check
        threshold: Maximum std deviation to consider converged
    
    Returns:
        Tuple of (has_converged, convergence_round)
    """
    if len(accuracy_history) < window_size:
        return False, -1
    
    # Check last window_size rounds
    recent_accuracies = accuracy_history[-window_size:]
    std = np.std(recent_accuracies)
    
    if std < threshold:
        # Find when convergence first occurred
        for i in range(len(accuracy_history) - window_size + 1):
            window = accuracy_history[i:i+window_size]
            if np.std(window) < threshold:
                return True, i + window_size
        return True, len(accuracy_history)
    
    return False, -1


def compute_convergence_time(
    accuracy_history: List[float],
    time_history: List[float],
    target_accuracy: float = 0.95
) -> Dict[str, float]:
    """
    Compute time to reach target accuracy.
    
    Args:
        accuracy_history: Accuracy values over rounds
        time_history: Cumulative time at each round
        target_accuracy: Target accuracy threshold
    
    Returns:
        Dict with convergence metrics
    """
    # Find first round where target accuracy is reached
    for i, acc in enumerate(accuracy_history):
        if acc >= target_accuracy:
            return {
                'converged': True,
                'convergence_round': i + 1,
                'convergence_time': time_history[i] if i < len(time_history) else 0,
                'final_accuracy': accuracy_history[-1]
            }
    
    return {
        'converged': False,
        'convergence_round': -1,
        'convergence_time': -1,
        'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0
    }


def evaluate_model_comprehensive(
    model,
    test_loader,
    device='cpu',
    num_classes=10
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Computes accuracy, F1, AUROC, and per-class metrics.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run on
        num_classes: Number of classes
    
    Returns:
        Dict with all metrics
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get probabilities (softmax)
            probs = torch.softmax(output, dim=1)
            
            # Get predictions
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    metrics = {
        'accuracy': compute_accuracy(all_predictions, all_labels),
        'f1_macro': compute_f1_score(all_predictions, all_labels, average='macro'),
        'f1_weighted': compute_f1_score(all_predictions, all_labels, average='weighted'),
        'auroc': compute_auroc(all_predictions, all_probabilities, all_labels, num_classes)
    }
    
    # Per-class accuracy
    for class_id in range(num_classes):
        mask = all_labels == class_id
        if mask.sum() > 0:
            class_acc = compute_accuracy(
                all_predictions[mask], 
                all_labels[mask]
            )
            metrics[f'class_{class_id}_accuracy'] = class_acc
    
    return metrics