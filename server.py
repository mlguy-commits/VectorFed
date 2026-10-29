"""
Synchronous federated learning server.
Handles: FedAvg, SyncFed, OracleFed, VectorFed

One round = broadcast → collect all → aggregate → evaluate → log
Server drives the pace. Clients respond each round.

Usage:
    python3 server.py --algorithm VectorFed --dataset cifar10 --trial 1

NOT for FedBuff. Use server_async.py for FedBuff.
"""

import os
import sys
import socket
import argparse
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import get_config
from models import get_model
from data import get_test_loader
from utils.communication import send_message, recv_message
from utils.serialization import pack, unpack, make_metadata
from utils.vector_clock import VectorClock
from utils.logger import Logger


# Minimum fraction of clients required for aggregation
MIN_CLIENT_FRACTION = 0.5


# Evaluation
def evaluate(
    model:       nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device:      torch.device,
) -> tuple:
    """
    Evaluate global model accuracy and loss on full test set.

    Args:
        model:       Global model (PyTorch)
        test_loader: DataLoader over full test set
        device:      Compute device

    Returns:
        (accuracy: float 0-100, loss: float)
    """
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    model.train()
    return (
        100.0 * total_correct / total_samples,
        total_loss / total_samples,
    )


# Registration
def register_clients(
    server_sock: socket.socket,
    num_clients: int,
    timeout:     int,
    logger:      Logger,
) -> dict:
    """
    Accept connections from all clients before round 1.

    Blocks until all num_clients have registered,
    or raises RuntimeError if timeout expires first.

    Args:
        server_sock: Bound, listening server socket
        num_clients: Expected number of clients
        timeout:     Max seconds to wait for all clients
        logger:      Logger instance

    Returns:
        Dict: client_id (int) → socket object

    Raises:
        RuntimeError: If not all clients connect in time
    """
    connections = {}
    deadline = time.time() + timeout

    print(f"\nWaiting for {num_clients} clients...")
    logger.log_event(
        f"Registration started, expecting {num_clients} clients"
    )

    server_sock.settimeout(5.0)

    while len(connections) < num_clients:

        if time.time() > deadline:
            raise RuntimeError(
                f"Registration timeout: {len(connections)}/"
                f"{num_clients} clients connected"
            )

        try:
            conn, addr = server_sock.accept()
        except socket.timeout:
            remaining = int(deadline - time.time())
            print(
                f"  {len(connections)}/{num_clients} connected "
                f"({remaining}s remaining)..."
            )
            continue

        try:
            conn.settimeout(30.0)
            data = recv_message(conn)
            state_dict, meta = unpack(data)

            if meta.get('msg_type') != 'register':
                conn.close()
                continue

            client_id = meta['client_id']

            if client_id in connections:
                print(f"  Duplicate registration from client {client_id}, rejected")
                conn.close()
                continue

            conn.settimeout(None)
            connections[client_id] = conn

            print(
                f"  Client {client_id:>2} registered "
                f"from {addr[0]}  "
                f"({len(connections)}/{num_clients})"
            )
            logger.log_event(
                f"Client {client_id} registered from {addr[0]}"
            )

        except Exception as e:
            print(f"  Registration error from {addr}: {e}")
            conn.close()

    server_sock.settimeout(None)
    print(f"\nAll {num_clients} clients registered.\n")
    logger.log_event(f"All {num_clients} clients registered")

    return connections


# Re-registration
def attempt_reregistration(
    server_sock: socket.socket,
    connections: dict,
    round_num:   int,
    logger:      Logger,
):
    """
    Short non-blocking window to re-accept dropped clients.

    Waits 3 seconds for any previously disconnected clients
    to reconnect. Does not block experiment if none reconnect.

    Args:
        server_sock: Bound, listening server socket
        connections: Dict client_id → socket (None = disconnected)
        round_num:   Current round number
        logger:      Logger instance
    """
    dropped = [cid for cid, s in connections.items() if s is None]

    if not dropped:
        return

    print(f"  Re-registration window open for clients: {dropped}")
    server_sock.settimeout(30.0)
    deadline = time.time() + 30.0

    while time.time() < deadline:
        try:
            conn, addr = server_sock.accept()
            conn.settimeout(10.0)

            data = recv_message(conn)
            state_dict, meta = unpack(data)

            if meta.get('msg_type') != 'register':
                conn.close()
                continue

            client_id = meta['client_id']

            if client_id in dropped:
                conn.settimeout(None)
                connections[client_id] = conn
                dropped.remove(client_id)
                print(f"  Client {client_id} re-registered from {addr[0]}")
                logger.log_event(
                    f"Round {round_num}: client {client_id} "
                    f"re-registered from {addr[0]}"
                )
            else:
                conn.close()

        except socket.timeout:
            break
        except Exception:
            break

    server_sock.settimeout(None)


# Broadcast
def broadcast_model(
    connections:  dict,
    global_model: nn.Module,
    server_clock: VectorClock,
    round_num:    int,
    logger:       Logger,
) -> dict:
    """
    Send current global model to all connected clients.

    Args:
        connections:  Dict client_id → socket (None = disconnected)
        global_model: Current global model
        server_clock: Server vector clock (incremented here)
        round_num:    Current round number
        logger:       Logger instance

    Returns:
        Dict client_id → send_timestamp for clients reached
    """
    server_clock.increment()

    metadata = make_metadata(
        msg_type     = 'broadcast',
        client_id    = -1,
        round_num    = round_num,
        vector_clock = server_clock.to_list(),
    )
    payload = pack(global_model.state_dict(), metadata)

    sent_to = {}

    for client_id in sorted(connections.keys()):
        sock = connections[client_id]

        if sock is None:
            continue

        try:
            send_message(sock, payload)
            sent_to[client_id] = time.time()

        except (OSError, ConnectionError) as e:
            print(f"  Broadcast failed → client {client_id}: {e}")
            logger.log_event(
                f"Round {round_num}: broadcast failed "
                f"to client {client_id}: {e}"
            )
            connections[client_id] = None

    return sent_to


# Collect Updates
def collect_updates(
    connections:  dict,
    sent_to:      dict,
    server_clock: VectorClock,
    round_num:    int,
    timeout:      int,
    logger:       Logger,
) -> list:
    """
    Collect one update from each client that received the broadcast.

    Records actual arrival time, network delay, and causal distance
    for each received update. Handles timeouts and disconnections
    gracefully by skipping the client and logging the failure.

    Args:
        connections:  Dict client_id → socket
        sent_to:      Dict client_id → send_timestamp
        server_clock: Server vector clock (for causal distance)
        round_num:    Current round number
        timeout:      Seconds to wait per client
        logger:       Logger instance

    Returns:
        List of metadata dicts. Each has 'state_dict' injected.
    """
    updates = []
    skipped = []

    for client_id in sorted(sent_to.keys()):

        sock = connections.get(client_id)
        if sock is None:
            skipped.append((client_id, 'disconnected_before_collect'))
            continue

        sock.settimeout(timeout)

        try:
            data             = recv_message(sock)
            arrival_time     = time.time()
            state_dict, meta = unpack(data)
            sock.settimeout(None)

            # Inject server-side measurements
            meta['arrival_time']     = arrival_time
            meta['send_time']        = sent_to[client_id]
            meta['network_delay']    = arrival_time - sent_to[client_id]
            meta['causal_distance']  = server_clock.causal_distance(
                meta.get('vector_clock', [])
            )
            meta['state_dict']       = state_dict

            updates.append(meta)

            print(
                f"  Client {client_id:>2} | "
                f"loss={meta.get('loss', 0):.4f} | "
                f"acc={meta.get('accuracy', 0):.2f}% | "
                f"train={meta.get('training_time', 0):.1f}s | "
                f"delay={meta['network_delay']:.2f}s | "
                f"causal_dist={meta['causal_distance']}"
            )

        except socket.timeout:
            sock.settimeout(None)
            reason = f"timeout after {timeout}s"
            connections[client_id] = None
            skipped.append((client_id, reason))
            print(f"  Client {client_id:>2} SKIPPED: {reason}")
            logger.log_event(
                f"Round {round_num}: client {client_id} "
                f"skipped ({reason})"
            )

        except (ConnectionError, OSError) as e:
            sock.settimeout(None)
            reason = f"connection lost: {e}"
            connections[client_id] = None
            skipped.append((client_id, reason))
            print(f"  Client {client_id:>2} SKIPPED: {reason}")
            logger.log_event(
                f"Round {round_num}: client {client_id} "
                f"skipped ({reason})"
            )

        except Exception as e:
            sock.settimeout(None)
            reason = f"unexpected error: {e}"
            skipped.append((client_id, reason))
            print(f"  Client {client_id:>2} SKIPPED: {reason}")
            logger.log_event(
                f"Round {round_num}: client {client_id} "
                f"skipped ({reason})"
            )

    if skipped:
        summary = ", ".join(
            f"client {cid} ({reason})" for cid, reason in skipped
        )
        logger.log_event(
            f"Round {round_num}: skipped clients → {summary}"
        )

    return updates


# Main Server Loop
def run_server(config: dict):
    """
    Run the complete synchronous federated experiment.

    Args:
        config: Experiment config from config/config.py
    """
    algorithm   = config['algorithm']
    dataset     = config['dataset']
    trial       = config['trial']
    num_rounds  = config['num_rounds']
    num_clients = config['num_clients']
    timeout     = config['timeout']
    device      = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"  CausalGuard FL Server  (Synchronous)")
    print(f"{'='*60}")
    print(f"  Algorithm : {algorithm}")
    print(f"  Dataset   : {dataset}")
    print(f"  Trial     : {trial}")
    print(f"  Rounds    : {num_rounds}")
    print(f"  Clients   : {num_clients}")
    print(f"  Timeout   : {timeout}s per client per round")
    print(f"{'='*60}\n")

    # Logger
    logger = Logger(config)

    # Global model
    global_model = get_model(config)
    global_model.to(device)
    logger.log_event(f"Model: {config['model']}")

    # Vector clock
    server_clock = VectorClock(
        num_processes = config['num_processes'],
        process_id    = config['server_id'],
    )

    # Algorithm
    from algorithms import get_algorithm
    aggregator = get_algorithm(config, server_clock)
    logger.log_event(f"Algorithm: {algorithm}")

    # Test set
    print("Loading test set...")
    test_loader = get_test_loader(config)

    # Server socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', config['server_port']))
    server_sock.listen(num_clients + 5)
    print(f"Listening on 0.0.0.0:{config['server_port']}\n")

    # Registration
    try:
        connections = register_clients(
            server_sock = server_sock,
            num_clients = num_clients,
            timeout     = timeout,
            logger      = logger,
        )
    except RuntimeError as e:
        print(f"\nFATAL: {e}")
        server_sock.close()
        return

    # Baseline evaluation
    init_acc, init_loss = evaluate(global_model, test_loader, device)
    print(f"Baseline | acc={init_acc:.2f}%  loss={init_loss:.4f}\n")
    logger.log_event(
        f"Baseline | accuracy={init_acc:.4f}% | loss={init_loss:.6f}"
    )

    min_clients = max(1, int(num_clients * MIN_CLIENT_FRACTION))

    # Training rounds
    for round_num in range(1, num_rounds + 1):

        print(f"\n{'─'*60}")
        print(f"  Round {round_num}/{num_rounds}")
        print(f"{'─'*60}")
        logger.mark_round_start()
        logger.log_event(f"Round {round_num} started")

        # Re-registration window (skip round 1)
        if round_num > 1:
            attempt_reregistration(
                server_sock = server_sock,
                connections = connections,
                round_num   = round_num,
                logger      = logger,
            )

        # Active client count
        active = [cid for cid, s in connections.items() if s is not None]
        print(f"  Active clients: {active}")

        if len(active) < min_clients:
            print(
                f"  Only {len(active)} active clients "
                f"(need {min_clients}). Skipping round."
            )
            logger.log_event(
                f"Round {round_num} skipped: "
                f"{len(active)} active (need {min_clients})"
            )
            continue

        # Broadcast
        print("  Broadcasting...")
        sent_to = broadcast_model(
            connections  = connections,
            global_model = global_model,
            server_clock = server_clock,
            round_num    = round_num,
            logger       = logger,
        )

        # Collect
        print(f"  Collecting (timeout={timeout}s)...")
        updates = collect_updates(
            connections  = connections,
            sent_to      = sent_to,
            server_clock = server_clock,
            round_num    = round_num,
            timeout      = timeout,
            logger       = logger,
        )

        # Minimum participation check
        if len(updates) < min_clients:
            print(
                f"  Only {len(updates)} updates received "
                f"(need {min_clients}). Keeping current model."
            )
            logger.log_event(
                f"Round {round_num}: aggregation skipped, "
                f"{len(updates)}/{num_clients} updates"
            )
            accuracy, loss = evaluate(global_model, test_loader, device)
            logger.log_round(round_num, accuracy, loss, updates)
            continue

        # Aggregate
        print(f"  Aggregating {len(updates)} updates via {algorithm}...")
        new_weights = aggregator.aggregate(updates, server_clock)
        global_model.load_state_dict(new_weights)

        # Evaluate
        print("  Evaluating...")
        accuracy, loss = evaluate(global_model, test_loader, device)

        # Log
        logger.log_round(round_num, accuracy, loss, updates)
        logger.log_event(
            f"Round {round_num} done | "
            f"acc={accuracy:.4f}% | "
            f"loss={loss:.6f} | "
            f"clients={len(updates)}/{num_clients}"
        )

    # Finalize
    final_acc, final_loss = evaluate(global_model, test_loader, device)
    logger.finalize(final_acc, final_loss)

    # Close connections
    for sock in connections.values():
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    server_sock.close()
    logger.log_event("Server shut down cleanly")


# Entry Point
def parse_args():
    parser = argparse.ArgumentParser(
        description='CausalGuard Synchronous FL Server'
    )
    parser.add_argument(
        '--algorithm',
        type     = str,
        required = True,
        choices  = ['FedAvg', 'SyncFed', 'OracleFed', 'VectorFed', 'FedBuff'],
        help     = 'Aggregation algorithm',
    )
    parser.add_argument(
        '--dataset',
        type     = str,
        required = True,
        choices  = ['mnist', 'cifar10', 'cifar100'],
        help     = 'Dataset',
    )
    parser.add_argument(
        '--trial',
        type    = int,
        default = 1,
        help    = 'Trial number (default: 1)',
    )
    parser.add_argument(
        '--gamma',
        type    = float,
        default = None,
        help    = 'Override gamma_causal for sensitivity analysis '
                '(default: use config value)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    config = get_config(args.algorithm, args.dataset, args.trial)

    # Override gamma_causal if provided (for sensitivity analysis)
    if args.gamma is not None:
        config['gamma_causal'] = args.gamma
        print(f"  [Override] gamma_causal = {args.gamma}")
        
    run_server(config)