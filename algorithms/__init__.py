"""
Factory function for algorithm selection.
Both server.py and server_async.py call get_algorithm(config).

Usage:
    from algorithms import get_algorithm
    aggregator = get_algorithm(config, server_clock)
    new_weights = aggregator.aggregate(updates, server_clock)
"""

from algorithms.fedavg import FedAvg
from algorithms.syncfed import SyncFed
from algorithms.oraclefed import OracleFed
from algorithms.vectorfed import VectorFed
from algorithms.fedbuff import FedBuff
from utils.vector_clock import VectorClock


def get_algorithm(
    config:       dict,
    server_clock: VectorClock,
) -> 'BaseAggregator':
    """
    Instantiate and return the correct aggregation algorithm.

    Args:
        config:       Experiment config from config/config.py
        server_clock: Server's VectorClock instance
                      (passed by reference, shared with server)

    Returns:
        Instantiated aggregator object

    Raises:
        ValueError: If algorithm name is not recognized
    """
    algorithm = config['algorithm']

    registry = {
        'FedAvg':    FedAvg,
        'SyncFed':   SyncFed,
        'OracleFed': OracleFed,
        'VectorFed': VectorFed,
        'FedBuff':   FedBuff,
    }

    if algorithm not in registry:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: {list(registry.keys())}"
        )

    return registry[algorithm](config, server_clock)
