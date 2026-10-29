"""
Single source of truth for all experiment configuration.
Both server.py and client.py import from this file.

Usage:
    from config.config import get_cifar10_config, get_mnist_config

    config = get_cifar10_config(algorithm='VectorFed', trial=1)
    config['server_ip']    → '<VM1 Server IP>'
    config['num_rounds']   → 50
    config['gamma_causal'] → 0.1
"""

import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

# NETWORK CONFIGURATION
# VM1: Server
SERVER_IP   = os.getenv("VM1")
SERVER_PORT = os.getenv("SERVER_PORT")

# Client distribution across VMs
# Key   = client_id (1-indexed)
# Value = VM IP address
CLIENT_REGISTRY = {
    1:  os.getenv("VM2"),   # VM2
    2:  os.getenv("VM2"),   # VM2
    3:  os.getenv("VM2"),   # VM2
    4:  os.getenv("VM3"),   # VM3
    5:  os.getenv("VM3"),   # VM3
    6:  os.getenv("VM3"),   # VM3
    7:  os.getenv("VM4"),   # VM4
    8:  os.getenv("VM4"),   # VM4
    9:  os.getenv("VM5"),   # VM5
    10: os.getenv("VM5"),   # VM5
}

# Total number of processes = num_clients + 1 server
# Used to initialize vector clock size
NUM_CLIENTS   = 10
NUM_PROCESSES = NUM_CLIENTS + 1   # indices 0-9 = clients, index 10 = server
SERVER_ID     = NUM_PROCESSES - 1 # index 10


# Algorithmic parameters (decay rates, buffer sizes, etc.)
# VectorFed: causal decay rate
# lambda_k = exp(-GAMMA_CAUSAL * causal_distance)
GAMMA_CAUSAL = 0.1

# SyncFed: network delay decay rate
# lambda_k = exp(-GAMMA_SYNC * network_delay)
GAMMA_SYNC = 0.1

# OracleFed: dual timestamp decay rates
# lambda_gen = exp(-GAMMA_GEN * (T_server - T_generation))
# lambda_arr = exp(-GAMMA_ARR * (T_server - T_arrival))
# lambda_k   = ALPHA_ORACLE * lambda_gen + (1 - ALPHA_ORACLE) * lambda_arr
GAMMA_GEN    = 0.1
GAMMA_ARR    = 0.1
ALPHA_ORACLE = 0.5

# FedBuff: aggregate when this many updates are buffered
BUFFER_SIZE = 5


# System configuration

# How long server waits for a client response before timing out (seconds)
# Set high enough for slow training rounds
TIMEOUT = 10800

# Base directory for results (on VM1)
RESULTS_DIR = './results'

# Base directory for datasets (on all VMs)
DATA_DIR = './data'


# Supported algorithms and datasets
ALGORITHMS = [
    'FedAvg',
    'SyncFed',
    'OracleFed',
    'VectorFed',
    'FedBuff',
]

DATASETS = [
    'mnist',
    'cifar10',
    'cifar100',
    'femnist',
]


# Config builder function
def _base_config(algorithm: str, trial: int) -> dict:
    """
    Build the base config dict shared across all datasets.

    Args:
        algorithm: One of ALGORITHMS
        trial:     Trial number (1-indexed, for Monte Carlo)

    Returns:
        Dict with network, algorithm, and system settings
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: {ALGORITHMS}"
        )

    return {
        # --- Identity ---
        'algorithm':      algorithm,
        'trial':          trial,

        # --- Network ---
        'server_ip':      SERVER_IP,
        'server_port':    SERVER_PORT,
        'client_registry': CLIENT_REGISTRY,
        'timeout':        TIMEOUT,

        # --- FL system ---
        'num_clients':    NUM_CLIENTS,
        'num_processes':  NUM_PROCESSES,
        'server_id':      SERVER_ID,

        # --- Algorithm parameters ---
        'gamma_causal':   GAMMA_CAUSAL,
        'gamma_sync':     GAMMA_SYNC,
        'gamma_gen':      GAMMA_GEN,
        'gamma_arr':      GAMMA_ARR,
        'alpha_oracle':   ALPHA_ORACLE,
        'buffer_size':    BUFFER_SIZE,

        # --- Paths ---
        'results_dir':    RESULTS_DIR,
        'data_dir':       DATA_DIR,
    }


def get_mnist_config(algorithm: str, trial: int = 1) -> dict:
    """
    Config for MNIST experiments.

    Dataset:  60,000 training / 10,000 test
    Model:    SimpleCNN (2 conv layers + 2 fc layers)
    Rounds:   20 (MNIST converges quickly)

    Args:
        algorithm: One of ALGORITHMS
        trial:     Trial number for Monte Carlo (default 1)

    Returns:
        Complete flat config dict
    """
    config = _base_config(algorithm, trial)
    config.update({
        # --- Dataset ---
        'dataset':         'mnist',
        'num_classes':     10,
        'input_channels':  1,
        'input_size':      28,

        # --- Training ---
        'num_rounds':      10,
        'local_epochs':    10,
        'batch_size':      32,
        'learning_rate':   0.01,
        'momentum':        0.9,
        'weight_decay':    1e-4,
        'timeout': 1800,

        # --- Data partitioning ---
        'dirichlet_alpha': 0.5,

        # --- Model ---
        'model':           'SimpleCNN',
    })
    return config


def get_cifar10_config(algorithm: str, trial: int = 1) -> dict:
    """
    Config for CIFAR-10 experiments.

    Dataset:  50,000 training / 10,000 test
    Model:    ResNet-8 (lightweight ResNet for CIFAR)
    Rounds:   30 (CIFAR-10 needs more rounds)

    Args:
        algorithm: One of ALGORITHMS
        trial:     Trial number for Monte Carlo (default 1)

    Returns:
        Complete flat config dict
    """
    config = _base_config(algorithm, trial)
    config.update({
        # --- Dataset ---
        'dataset':         'cifar10',
        'num_classes':     10,
        'input_channels':  3,
        'input_size':      32,

        # --- Training ---
        'num_rounds':      10,
        'local_epochs':    10,
        'batch_size':      64,
        'learning_rate':   0.01,
        'momentum':        0.9,
        'weight_decay':    1e-4,
        'timeout': 2400,

        # --- Data partitioning ---
        'dirichlet_alpha': 0.5,

        # --- Model ---
        'model':           'ResNetCIFAR',
    })
    return config


def get_cifar100_config(algorithm: str, trial: int = 1) -> dict:
    """
    Config for CIFAR-100 experiments.

    Dataset:  50,000 training / 10,000 test, 100 classes
    Model:    ResNet-8 (same as CIFAR-10, output layer = 100)
    Rounds:   30

    Args:
        algorithm: One of ALGORITHMS
        trial:     Trial number for Monte Carlo (default 1)

    Returns:
        Complete flat config dict
    """
    config = _base_config(algorithm, trial)
    config.update({
        # --- Dataset ---
        'dataset':         'cifar100',
        'num_classes':     100,
        'input_channels':  3,
        'input_size':      32,

        # --- Training ---
        'num_rounds':      10,
        'local_epochs':    20,
        'batch_size':      64,
        'learning_rate':   0.01,
        'momentum':        0.9,
        'weight_decay':    1e-4,
        'timeout': 4000,

        # --- Data partitioning ---
        'dirichlet_alpha': 0.5,

        # --- Model ---
        'model':           'ResNetCIFAR',
    })
    return config


def get_femnist_config(algorithm: str, trial: int = 1) -> dict:
    """
    Config for FEMNIST experiments.

    Dataset:  Federated EMNIST, naturally non-IID by writer
              ~226,000 training samples, 62 classes
    Model:    SimpleCNN (same as MNIST, output layer = 62)
    Rounds:   30

    Note: FEMNIST is already naturally partitioned by writer.
    Dirichlet partitioning is NOT used for this dataset.
    Each client gets data from a subset of writers.

    Args:
        algorithm: One of ALGORITHMS
        trial:     Trial number for Monte Carlo (default 1)

    Returns:
        Complete flat config dict
    """
    config = _base_config(algorithm, trial)
    config.update({
        # --- Dataset ---
        'dataset':         'femnist',
        'num_classes':     62,
        'input_channels':  1,
        'input_size':      28,

        # --- Training ---
        'num_rounds':      10,
        'local_epochs':    15,
        'batch_size':      32,
        'learning_rate':   0.01,
        'momentum':        0.9,
        'weight_decay':    1e-4,

        # --- Data partitioning ---
        # FEMNIST is naturally non-IID: no Dirichlet needed
        'dirichlet_alpha': None,

        # --- Model ---
        'model':           'SimpleCNN',
    })
    return config


def get_config(algorithm: str, dataset: str, trial: int = 1) -> dict:
    """
    Generic config getter. Dispatches to the correct dataset function.

    Useful when dataset is a runtime variable (e.g. in loops):
        for dataset in ['mnist', 'cifar10', 'cifar100']:
            config = get_config('VectorFed', dataset, trial=1)

    Args:
        algorithm: One of ALGORITHMS
        dataset:   One of DATASETS
        trial:     Trial number (default 1)

    Returns:
        Complete flat config dict

    Raises:
        ValueError: If dataset is not recognized
    """
    dispatch = {
        'mnist':    get_mnist_config,
        'cifar10':  get_cifar10_config,
        'cifar100': get_cifar100_config,
        'femnist':  get_femnist_config,
    }

    if dataset not in dispatch:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {DATASETS}"
        )

    return dispatch[dataset](algorithm, trial)


# QUICK SANITY CHECK
# Run directly to print all configs:
#   python3 config/config.py

if __name__ == '__main__':
    print("=" * 60)
    print("CausalGuard Configuration Sanity Check")
    print("=" * 60)

    for algo in ALGORITHMS:
        for dataset in DATASETS:
            config = get_config(algo, dataset, trial=1)
            print(
                f"\n{algo:12s} | {dataset:10s} | "
                f"rounds={config['num_rounds']:>3} | "
                f"batch={config['batch_size']:>3} | "
                f"lr={config['learning_rate']} | "
                f"model={config['model']}"
            )

    print("\n" + "=" * 60)
    print("Network:")
    print(f"  Server:  {SERVER_IP}:{SERVER_PORT}")
    print(f"  Clients: {NUM_CLIENTS} total")
    for cid, ip in CLIENT_REGISTRY.items():
        print(f"    Client {cid:>2} → {ip}")
    print("=" * 60)
