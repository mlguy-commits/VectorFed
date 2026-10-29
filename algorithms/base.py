"""
Base aggregator class shared by all algorithms.
Provides weighted_average() so no algorithm duplicates it.

All algorithms inherit from BaseAggregator and implement:
    aggregate(updates, server_clock) -> state_dict
"""

import torch
from utils.vector_clock import VectorClock


class BaseAggregator:
    """
    Base class for all federated aggregation algorithms.

    Subclasses must implement aggregate().
    Subclasses may use weighted_average() for the final step.
    """

    def __init__(self, config: dict, server_clock: VectorClock):
        """
        Args:
            config:       Experiment config from config/config.py
            server_clock: Server's VectorClock instance
                          (shared reference, not a copy)
        """
        self.config = config
        self.server_clock = server_clock

    def aggregate(
        self,
        updates:      list,
        server_clock: VectorClock,
    ) -> dict:
        """
        Aggregate client updates into new global model weights.

        Args:
            updates:      List of metadata dicts from collect_updates()
                          Each dict has 'state_dict' injected by server.
            server_clock: Server's current VectorClock

        Returns:
            New global model state_dict (weighted average)

        Raises:
            NotImplementedError: Subclass must implement this
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement aggregate()"
        )

    def weighted_average(
        self,
        updates: list,
        weights: list,
    ) -> dict:
        """
        Compute weighted average of model state dicts.

        Shared by all algorithms. Called after weights are computed.

        Args:
            updates: List of update metadata dicts (with 'state_dict')
            weights: List of floats, must sum to 1.0, same length as updates

        Returns:
            New state_dict: weighted average of all client models
        """
        assert len(updates) == len(weights), (
            f"updates ({len(updates)}) and weights ({len(weights)}) "
            f"must have same length"
        )
        assert abs(sum(weights) - 1.0) < 1e-6, (
            f"weights must sum to 1.0, got {sum(weights):.6f}"
        )

        new_state_dict = {}
        reference_sd = updates[0]['state_dict']

        for key in reference_sd.keys():
            # Stack all client tensors for this key
            tensors = torch.stack([
                u['state_dict'][key].float()
                for u in updates
            ])

            # Build weight tensor with correct shape for broadcasting
            # Shape: (num_updates, 1, 1, ...) matching tensor dims
            w = torch.tensor(weights, dtype=torch.float32)
            for _ in range(tensors.dim() - 1):
                w = w.unsqueeze(-1)

            # Weighted sum along client dimension
            new_state_dict[key] = (tensors * w).sum(dim=0)

        return new_state_dict

    def normalize_weights(self, raw_weights: list) -> list:
        """
        Normalize a list of raw weights to sum to 1.0.

        Guards against all-zero weights (falls back to uniform).

        Args:
            raw_weights: List of non-negative floats

        Returns:
            List of floats summing to 1.0
        """
        total = sum(raw_weights)

        if total < 1e-10:
            # All weights are zero or near-zero
            # Fall back to uniform weighting
            n = len(raw_weights)
            return [1.0 / n] * n

        return [w / total for w in raw_weights]