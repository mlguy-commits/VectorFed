"""
Vector Clock implementation for causal ordering in distributed
federated learning. Provides causal ordering without requiring
physical clock synchronization.
"""

from typing import List
import copy

class VectorClock:
    """
    Vector clock for tracking causal relationships in distributed systems.

    Each process maintains a vector of logical timestamps.
    Vector size = num_clients + 1 (server is last process).

    Example for 10 clients + 1 server:
        process_id 0-9  → clients
        process_id 10   → server
    """

    def __init__(self, num_processes: int, process_id: int):
        """
        Args:
            num_processes: Total number of processes (clients + server)
            process_id:    This process's identifier (0-indexed)
        """
        self.num_processes = num_processes
        self.process_id    = process_id
        self.clock         = [0] * num_processes

    def increment(self):
        """
        Increment own logical clock.
        Called on every local event (training complete, aggregation, etc.)
        """
        self.clock[self.process_id] += 1

    def update(self, other_clock: List[int]):
        """
        Update clock upon receiving a message (receive event).

        Takes element-wise maximum of own clock and received clock,
        then increments own position.

        Args:
            other_clock: Vector clock received in message (as list)
        """
        for i in range(self.num_processes):
            self.clock[i] = max(self.clock[i], other_clock[i])
        self.increment()

    # Alias: 'merge' makes receive-event semantics explicit
    # in distributed code. Both names do the same thing.
    merge = update

    def get_clock(self) -> List[int]:
        """
        Return a copy of the current clock state.

        Returns a copy, not a reference, so the caller
        cannot accidentally mutate the internal state.
        """
        return copy.copy(self.clock)

    # Alias: used when packing metadata for network transmission
    # metadata['vector_clock'] = clock.to_list()
    to_list = get_clock

    def set_clock(self, clock: List[int]):
        """
        Overwrite the clock with a given list of values.

        Args:
            clock: New clock state as list of ints
        """
        self.clock = copy.copy(clock)

    # Alias: used when unpacking metadata after network reception
    # clock.from_list(metadata['vector_clock'])
    from_list = set_clock

    def happens_before(self, other: 'VectorClock') -> bool:
        """
        Check if this clock happened-before another clock object.

        A -> B iff (A <= B) and (A != B)
        meaning: every element of A is <= corresponding element of B,
        and at least one element is strictly less.

        Args:
            other: Another VectorClock object

        Returns:
            True if this event happened-before other
        """
        return self.happens_before_list(other.clock)

    def happens_before_list(self, other_clock: List[int]) -> bool:
        """
        Check if this clock happened-before another clock given as list.

        Useful when comparing against a received clock value
        without constructing a full VectorClock object.

        Args:
            other_clock: Other vector clock as plain list

        Returns:
            True if this happens-before other_clock
        """
        # Every element must be <= corresponding element in other
        less_or_equal = all(
            self.clock[i] <= other_clock[i]
            for i in range(self.num_processes)
        )

        # At least one element must be strictly less (not equal)
        strictly_less = any(
            self.clock[i] < other_clock[i]
            for i in range(self.num_processes)
        )

        return less_or_equal and strictly_less

    def concurrent(self, other: 'VectorClock') -> bool:
        """
        Check if two events are concurrent (no causal relationship).

        Two events are concurrent if neither happened-before the other.
        In FL this is the common case: clients training simultaneously
        from the same global model produce concurrent updates.

        Args:
            other: Another VectorClock object

        Returns:
            True if this event and other are concurrent
        """
        return (
            not self.happens_before(other) and
            not other.happens_before(self)
        )

    def causal_distance(self, other_clock: List[int]) -> int:
        """
        Compute causal distance between this clock and another.

        Distance = L1 norm (sum of absolute differences).
        Used by VectorFed to compute aggregation weights:
            lambda_k = exp(-gamma * causal_distance(V_k, V_server))

        Larger distance = more stale = lower aggregation weight.

        Args:
            other_clock: Other vector clock as plain list

        Returns:
            Causal distance as non-negative integer
        """
        return sum(
            abs(self.clock[i] - other_clock[i])
            for i in range(self.num_processes)
        )

    def __str__(self):
        return f"VC(pid={self.process_id}, clock={self.clock})"

    def __repr__(self):
        return self.__str__()


def compare_vector_clocks(clock_a: List[int], clock_b: List[int]) -> int:
    """
    Compare two vector clocks given as plain lists.

    Useful for sorting a list of updates without constructing
    VectorClock objects for each one.

    Args:
        clock_a: First vector clock as list
        clock_b: Second vector clock as list

    Returns:
        -1  if clock_a happened-before clock_b
         0  if concurrent (no causal relationship)
        +1  if clock_b happened-before clock_a
    """
    a_less_equal   = all(clock_a[i] <= clock_b[i] for i in range(len(clock_a)))
    a_strictly_less = any(clock_a[i] <  clock_b[i] for i in range(len(clock_a)))

    if a_less_equal and a_strictly_less:
        return -1   # clock_a happened-before clock_b

    b_less_equal    = all(clock_b[i] <= clock_a[i] for i in range(len(clock_a)))
    b_strictly_less = any(clock_b[i] <  clock_a[i] for i in range(len(clock_a)))

    if b_less_equal and b_strictly_less:
        return +1   # clock_b happened-before clock_a

    return 0        # concurrent
