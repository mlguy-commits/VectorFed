"""
VectorFed: Causal federated learning via vector clocks.
This is the novel algorithm proposed in this dissertation.

Aggregation weight:
    d_k      = causal_distance(V_k, V_server)
    lambda_k = exp(-gamma_causal * d_k)
    w_k      = (lambda_k * n_k) / sum(lambda_j * n_j)

Key properties:
    1. No physical clock synchronization required
    2. Causal ordering guaranteed via topological sort
    3. Stale updates automatically downweighted by causal distance
    4. Infrastructure-free: works on IoT, air-gapped, embedded systems

Replaces NTP-dependent physical timestamps with
Lamport-style vector clocks for temporal awareness.
"""

import math
from algorithms.base import BaseAggregator
from utils.vector_clock import VectorClock, compare_vector_clocks


class VectorFed(BaseAggregator):
    """
    Vector clock-based federated learning aggregation.

    Three-step algorithm:
        1. Sort updates by causal order (topological sort)
        2. Compute weights from causal distance
        3. Weighted average + server clock update
    """

    def aggregate(
        self,
        updates:      list,
        server_clock: VectorClock,
    ) -> dict:
        """
        Aggregate updates using causal distance weighting.

        Args:
            updates:      List of client update metadata dicts
                          Must contain:
                              'vector_clock'    (list) client's clock
                              'causal_distance' (int)  pre-computed by server
                              'num_samples'     (int)
            server_clock: Server's VectorClock (merged after aggregation)

        Returns:
            New global model state_dict
        """
        gamma_causal = self.config['gamma_causal']

        # Causal sort
        sorted_updates = self._causal_sort(updates)

        # Compute causal weights
        raw_weights = []

        for u in sorted_updates:
            d_k      = float(u.get('causal_distance', 0))
            lambda_k = math.exp(-gamma_causal * d_k)
            raw_weights.append(lambda_k * float(u.get('num_samples', 1)))

        weights = self.normalize_weights(raw_weights)

        # Weighted average
        new_state_dict = self.weighted_average(sorted_updates, weights)

        # Update server vector clock
        # Merge all client clocks into server clock
        # Preserves causal history across rounds
        server_clock.increment()   # aggregation event
        for u in sorted_updates:
            client_clock = u.get('vector_clock', [])
            if client_clock:
                server_clock.merge(client_clock)

        return new_state_dict

    def _causal_sort(self, updates: list) -> list:
        """
        Topological sort of updates by happens-before relation.

        Algorithm:
            Repeatedly find an update with no causal predecessors
            in the remaining set. Add it to sorted list. Remove it.
            Repeat until all updates are sorted.

        Complexity: O(n^2 * K) where n = num_updates, K = clock size.
        For n=10 clients and K=11: ~1100 comparisons. Negligible.

        Concurrent updates (most common case) are ordered by
        arrival_time as tiebreaker for determinism.

        Args:
            updates: List of update metadata dicts

        Returns:
            Same updates in causal order
        """
        # Sort by arrival_time first as tiebreaker for concurrent updates
        # This makes topological sort deterministic
        remaining = sorted(
            updates,
            key=lambda u: float(u.get('arrival_time', 0))
        )

        sorted_list = []

        while remaining:
            for i, u_i in enumerate(remaining):
                v_i = u_i.get('vector_clock', [])

                # Check if any other update happened-before u_i
                has_predecessor = any(
                    compare_vector_clocks(
                        u_j.get('vector_clock', []),
                        v_i
                    ) == -1
                    for u_j in remaining
                    if u_j is not u_i
                )

                if not has_predecessor:
                    sorted_list.append(u_i)
                    remaining.pop(i)
                    break

        return sorted_list