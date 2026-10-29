"""
Federated learning client. Runs on VM2-VM5.

Responsibilities:
    1. Connect to server and register
    2. Receive global model each round
    3. Train locally on private data partition
    4. Send update (weights + metrics + vector clock) to server
    5. Exit cleanly on terminate signal

Works for ALL algorithms:
    Synchronous (FedAvg, SyncFed, OracleFed, VectorFed):
        receive → train → send → wait for next broadcast
    Asynchronous (FedBuff):
        receive → train → send → immediately receive again
    The loop structure is identical.
    server_async.py handles the async buffering.
    This client just keeps looping until terminate.

Usage:
    python3 client.py --client_id 3 --algorithm VectorFed
                      --dataset cifar10 --trial 1
"""

import os
import sys
import socket
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import get_config
from models import get_model
from data import get_train_loader
from utils.communication import send_message, recv_message
from utils.serialization import pack, unpack, make_metadata
from utils.vector_clock import VectorClock

# Retry settings for connection resilience
MAX_RETRIES = 3
RETRY_DELAY = 5   # seconds between retries

# Local evaluation
def evaluate_local(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """
    Evaluate model on client's local training data.

    Called after local training each round to measure
    loss and accuracy before sending update to server.

    Args:
        model:  Locally trained model
        loader: Client's training DataLoader
        device: Compute device

    Returns:
        (loss: float, accuracy: float 0-100)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    model.train()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy

# Local training
def train_local(
    model:        nn.Module,
    loader:       torch.utils.data.DataLoader,
    config:       dict,
    device:       torch.device,
    round_num:    int,
    client_id:    int,
) -> tuple:
    """
    Train model locally for one FL round.

    Args:
        model:     Global model received from server (modified in place)
        loader:    Client's training DataLoader
        config:    Experiment config
        device:    Compute device
        round_num: Current round number (for logging)
        client_id: This client's ID (for logging)

    Returns:
        (training_time: float, final_loss: float, final_accuracy: float)
    """
    model.train()
    model.to(device)

    lr = config['learning_rate']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    local_epochs = config['local_epochs']

    optimizer = optim.SGD(
        model.parameters(),
        lr           = lr,
        momentum     = momentum,
        weight_decay = weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print(
        f"  [Client {client_id}] Round {round_num} | "
        f"Training {local_epochs} epochs on "
        f"{len(loader.dataset)} samples..."
    )

    start_time   = time.time()
    epoch_losses = []

    for epoch in range(local_epochs):
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_samples += labels.size(0)

        epoch_loss = running_loss / running_samples if running_samples > 0 else 0.0
        epoch_losses.append(epoch_loss)

    training_time = time.time() - start_time

    # Evaluate on local training set after all epochs
    final_loss, final_accuracy = evaluate_local(model, loader, device)

    print(
        f"  [Client {client_id}] Round {round_num} | "
        f"Done | "
        f"Loss: {final_loss:.4f} | "
        f"Acc: {final_accuracy:.2f}% | "
        f"Time: {training_time:.1f}s"
    )

    return training_time, final_loss, final_accuracy

# Connection
def connect_to_server(
    server_ip:   str,
    server_port: int,
    client_id:   int,
    config:      dict,
    model:       nn.Module,
    timeout:     int = 30,
) -> socket.socket:
    """
    Connect to server and send registration message.

    Retries up to MAX_RETRIES times with RETRY_DELAY seconds
    between attempts if connection fails.

    Args:
        server_ip:   Server IP address
        server_port: Server port number
        client_id:   This client's ID
        config:      Experiment config
        model:       Initialized model (state dict sent with registration)
        timeout:     Socket timeout for registration response

    Returns:
        Connected and registered socket

    Raises:
        RuntimeError: If all retry attempts fail
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"  [Client {client_id}] Connecting to "
                f"{server_ip}:{server_port} "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((server_ip, server_port))

            # Send registration message
            metadata = make_metadata(
                msg_type = 'register',
                client_id = client_id,
                round_num = 0,
            )
            payload = pack(model.state_dict(), metadata)
            send_message(sock, payload)

            # Remove timeout after successful registration
            sock.settimeout(None)

            print(f"  [Client {client_id}] Registered successfully.")
            return sock

        except (ConnectionRefusedError, OSError, socket.timeout) as e:
            print(
                f"  [Client {client_id}] Connection failed: {e}"
            )
            try:
                sock.close()
            except Exception:
                pass

            if attempt < MAX_RETRIES:
                print(
                    f"  [Client {client_id}] Retrying in "
                    f"{RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Client {client_id}: failed to connect after "
        f"{MAX_RETRIES} attempts"
    )

# Main client loop
def run_client(config: dict, client_id: int):
    """
    Run the complete federated learning client.

    Args:
        config:    Experiment config from config/config.py
        client_id: This client's unique identifier (1-indexed)
    """
    algorithm = config['algorithm']
    dataset = config['dataset']
    trial = config['trial']
    device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"  CausalGuard FL Client {client_id}")
    print(f"{'='*60}")
    print(f"  Algorithm : {algorithm}")
    print(f"  Dataset   : {dataset}")
    print(f"  Trial     : {trial}")
    print(f"  Server    : {config['server_ip']}:{config['server_port']}")
    print(f"{'='*60}\n")

    # Initilize model
    model = get_model(config)
    model.to(device)
    print(f"  [Client {client_id}] Model: {config['model']}")

    # Initialize vector clock
    # Client process_id is client_id - 1 (0-indexed internally)
    client_clock = VectorClock(
        num_processes = config['num_processes'],
        process_id = client_id - 1,
    )
    print(f"  [Client {client_id}] Vector clock initialized.")

    # Load data partition
    print(f"  [Client {client_id}] Loading data partition...")
    train_loader = get_train_loader(config, client_id)
    num_samples = len(train_loader.dataset)
    print(
        f"  [Client {client_id}] Loaded {num_samples} samples "
        f"from {dataset} partition."
    )

    # Connect to the server
    try:
        sock = connect_to_server(
            server_ip   = config['server_ip'],
            server_port = config['server_port'],
            client_id   = client_id,
            config      = config,
            model       = model,
        )
    except RuntimeError as e:
        print(f"\n  FATAL: {e}")
        return
    
    # Main training loop
    round_num = 0
    total_rounds_done = 0

    print(f"\n  [Client {client_id}] Waiting for server broadcast...\n")

    while True:
        # Receive from server
        try:
            data = recv_message(sock)
            state_dict, meta = unpack(data)

        except (ConnectionError, OSError) as e:
            print(f"\n  [Client {client_id}] Connection lost: {e}")
            print(f"  [Client {client_id}] Attempting to reconnect...")

            try:
                sock.close()
            except Exception:
                pass
            
            # Wait before reconnecting to the server
            time.sleep(5)
            
            try:
                sock = connect_to_server(
                    server_ip = config['server_ip'],
                    server_port = config['server_port'],
                    client_id = client_id,
                    config = config,
                    model = model,
                )
                print(
                    f"  [Client {client_id}] Reconnected. "
                    f"Waiting for next broadcast..."
                )
                continue

            except RuntimeError:
                print(
                    f"  [Client {client_id}] Reconnection failed. "
                    f"Exiting."
                )
                break

        # Handle message type
        msg_type = meta.get('msg_type', '')

        # Terminate
        if msg_type == 'terminate':
            print(
                f"\n  [Client {client_id}] Received terminate signal. "
                f"Experiment complete."
            )
            break

        # Broadcast message response
        elif msg_type == 'broadcast':
            round_num = meta.get('round', round_num + 1)

            print(
                f"\n  [Client {client_id}] "
                f"Received broadcast for round {round_num}."
            )

            # Load global model weights
            model.load_state_dict(state_dict)

            # Update vector clock: receive event
            # merge server clock then increment own position
            server_clock_received = meta.get('vector_clock', [])
            if server_clock_received:
                client_clock.merge(server_clock_received)
            else:
                client_clock.increment()

            # Train locally
            training_time, final_loss, final_accuracy = train_local(
                model = model,
                loader = train_loader,
                config = config,
                device = device,
                round_num = round_num,
                client_id = client_id,
            )

            # Update vector clock: training complete (local event)
            client_clock.increment()

            # Build and send update
            update_meta = make_metadata(
                msg_type      = 'update',
                client_id     = client_id,
                round_num     = round_num,
                num_samples   = num_samples,
                vector_clock  = client_clock.to_list(),
                loss          = final_loss,
                accuracy      = final_accuracy,
                training_time = training_time,
            )

            try:
                payload = pack(model.state_dict(), update_meta)
                send_message(sock, payload)

                total_rounds_done += 1
                print(
                    f"  [Client {client_id}] Round {round_num} | "
                    f"Update sent | "
                    f"Clock: {client_clock.to_list()}"
                )

            except (ConnectionError, OSError) as e:
                print(
                    f"  [Client {client_id}] Failed to send update: {e}"
                )
                break
        
        # Unecpected message type
        else:
            print(
                f"  [Client {client_id}] WARNING: "
                f"Unexpected msg_type '{msg_type}'. Ignoring."
            )
            continue

    try:
        sock.close()
    except:
        pass

    print(f"\n{'='*60}")
    print(f"  Client {client_id} finished.")
    print(f"  Algorithm  : {algorithm}")
    print(f"  Dataset    : {dataset}")
    print(f"  Rounds done: {total_rounds_done}")
    print(f"  Final clock: {client_clock.to_list()}")
    print(f"{'='*60}\n")

# Entry point
def parse_args():
    parser = argparse.ArgumentParser(
        description='CausalGuard Federated Learning Client'
    )
    parser.add_argument(
        '--client_id',
        type     = int,
        required = True,
        help     = 'Client ID (1-10)',
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
        help    = 'Trial number for Monte Carlo (default: 1)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.algorithm, args.dataset, args.trial)
    run_client(config, args.client_id)