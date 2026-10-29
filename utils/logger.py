"""
CausalGuard/utils/logger.py

Centralized logging for federated learning experiments.
Runs exclusively on the server. Clients send their metrics
inside update metadata. Server writes everything to disk.

Directory structure:
    results/
        <algorithm>/
            <dataset>/
                rounds_trial_<N>.csv     ← one row per round
                summary_trial_<N>.json   → experiment summary
                events_trial_<N>.log     → debug event log

Three public functions:
    logger = Logger(config)
    logger.log_round(round_num, metrics, client_updates)
    logger.log_event(message)
    logger.finalize(final_accuracy, final_loss)
"""

import os
import csv
import json
import time
from datetime import datetime


# CSV column names for round-level log
ROUND_CSV_COLUMNS = [
    'experiment_id',
    'algorithm',
    'dataset',
    'trial',
    'round',
    'timestamp',
    'global_accuracy',
    'global_loss',
    'round_duration',
    'num_clients',
    'client_data',
]


class Logger:
    """
    Handles all experiment logging for one trial of one algorithm.

    One Logger instance is created per experiment run.
    It writes three files:
        rounds_trial_<N>.csv   → round-by-round metrics
        summary_trial_<N>.json → final experiment summary
        events_trial_<N>.log   → timestamped debug events
    """

    def __init__(self, config: dict):
        """
        Initialize logger and create output directory + files.

        Args:
            config: Experiment configuration dict containing:
                algorithm    (str)  e.g. 'VectorFed'
                dataset      (str)  e.g. 'cifar10'
                trial        (int)  trial number (1-10)
                num_rounds   (int)  total rounds planned
                num_clients  (int)  total number of clients
                results_dir  (str)  base results directory
                              default: 'results'
        """
        self.algorithm   = config['algorithm']
        self.dataset     = config['dataset']
        self.trial       = config['trial']
        self.num_rounds  = config['num_rounds']
        self.num_clients = config['num_clients']

        # Build experiment ID string
        self.experiment_id = (
            f"{self.algorithm.lower()}_"
            f"{self.dataset.lower()}_"
            f"trial{self.trial}"
        )

        # Build output directory: results/<algorithm>/<dataset>/
        base_dir = config.get('results_dir', 'results')
        self.output_dir = os.path.join(
            base_dir,
            self.algorithm.lower(),
            self.dataset.lower(),
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Build file paths
        suffix = f"trial_{self.trial}"
        self.csv_path     = os.path.join(self.output_dir, f"rounds_{suffix}.csv")
        self.summary_path = os.path.join(self.output_dir, f"summary_{suffix}.json")
        self.log_path     = os.path.join(self.output_dir, f"events_{suffix}.log")

        # Track experiment timing
        self.experiment_start = time.time()
        self.round_start      = None

        # Store round history for summary
        self.round_history = []

        # Initialize CSV with header row
        self._init_csv()

        # Write first log event
        self.log_event(
            f"Experiment started | "
            f"algorithm={self.algorithm} | "
            f"dataset={self.dataset} | "
            f"trial={self.trial} | "
            f"num_rounds={self.num_rounds} | "
            f"num_clients={self.num_clients}"
        )

    def _init_csv(self):
        """
        Create the CSV file and write the header row.
        If file already exists (e.g. experiment resumed), append.
        """
        file_exists = os.path.exists(self.csv_path)

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ROUND_CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()

    def mark_round_start(self):
        """
        Record wall-clock time when a round begins.
        Call this at the start of each round on the server,
        just before broadcasting to clients.
        """
        self.round_start = time.time()

    def log_round(
        self,
        round_num:      int,
        global_accuracy: float,
        global_loss:     float,
        client_updates:  list,
    ):
        """
        Log metrics for one completed round.

        Args:
            round_num:       Current round number (1-indexed)
            global_accuracy: Test accuracy of global model (0-100)
            global_loss:     Test loss of global model
            client_updates:  List of metadata dicts received from clients
                             Each dict has: client_id, loss, accuracy,
                             training_time, num_samples, vector_clock,
                             causal_distance (added by server before calling)
        """
        # Compute round duration
        round_duration = (
            time.time() - self.round_start
            if self.round_start is not None
            else 0.0
        )

        # Build per-client details dict
        client_data = {}
        for update in client_updates:
            cid = str(update.get('client_id', -1))
            client_data[cid] = {
                'loss':           round(update.get('loss',          0.0), 6),
                'accuracy':       round(update.get('accuracy',      0.0), 4),
                'training_time':  round(update.get('training_time', 0.0), 4),
                'num_samples':    update.get('num_samples',   0),
                'causal_distance':update.get('causal_distance', 0),
                'vector_clock':   update.get('vector_clock',  []),
                'timestamp':      round(update.get('timestamp', 0.0), 6),
            }

        # Build the row dict
        row = {
            'experiment_id':   self.experiment_id,
            'algorithm':       self.algorithm,
            'dataset':         self.dataset,
            'trial':           self.trial,
            'round':           round_num,
            'timestamp':       round(time.time(), 6),
            'global_accuracy': round(global_accuracy, 6),
            'global_loss':     round(global_loss, 6),
            'round_duration':  round(round_duration, 4),
            'num_clients':     len(client_updates),
            'client_data':     json.dumps(client_data),
        }

        # Append row to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ROUND_CSV_COLUMNS)
            writer.writerow(row)

        # Keep in memory for summary
        self.round_history.append({
            'round':           round_num,
            'global_accuracy': global_accuracy,
            'global_loss':     global_loss,
            'round_duration':  round_duration,
            'num_clients':     len(client_updates),
        })

        # Print progress to terminal
        print(
            f"  Round {round_num:>3}/{self.num_rounds} | "
            f"acc={global_accuracy:.2f}% | "
            f"loss={global_loss:.4f} | "
            f"duration={round_duration:.1f}s | "
            f"clients={len(client_updates)}"
        )

        # Reset round timer
        self.round_start = None

    def log_event(self, message: str):
        """
        Write a timestamped debug message to the .log file.

        Use for: client connections, disconnections, round starts,
        errors, algorithm-specific events (e.g. causal sort order).

        Args:
            message: Plain text message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        line = f"[{timestamp}] {message}\n"

        with open(self.log_path, 'a') as f:
            f.write(line)

    def finalize(self, final_accuracy: float, final_loss: float):
        """
        Write experiment summary JSON and final log event.

        Call this once, after the last round completes.

        Args:
            final_accuracy: Final global model test accuracy (0-100)
            final_loss:     Final global model test loss
        """
        total_duration = time.time() - self.experiment_start

        # Compute summary statistics across all rounds
        accuracies     = [r['global_accuracy'] for r in self.round_history]
        losses         = [r['global_loss']     for r in self.round_history]
        durations      = [r['round_duration']  for r in self.round_history]

        summary = {
            'experiment_id':    self.experiment_id,
            'algorithm':        self.algorithm,
            'dataset':          self.dataset,
            'trial':            self.trial,
            'num_rounds':       self.num_rounds,
            'num_clients':      self.num_clients,
            'final_accuracy':   round(final_accuracy, 6),
            'final_loss':       round(final_loss, 6),
            'best_accuracy':    round(max(accuracies), 6) if accuracies else 0.0,
            'best_round':       (
                accuracies.index(max(accuracies)) + 1
                if accuracies else 0
            ),
            'total_duration_s': round(total_duration, 2),
            'avg_round_s':      round(
                sum(durations) / len(durations), 4
            ) if durations else 0.0,
            'min_round_s':      round(min(durations), 4) if durations else 0.0,
            'max_round_s':      round(max(durations), 4) if durations else 0.0,
            'accuracy_history': [round(a, 4) for a in accuracies],
            'loss_history':     [round(l, 6) for l in losses],
            'completed_at':     datetime.now().isoformat(),
        }

        # Write summary JSON
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Write final log event
        self.log_event(
            f"Experiment finished | "
            f"final_accuracy={final_accuracy:.4f}% | "
            f"final_loss={final_loss:.6f} | "
            f"total_duration={total_duration:.1f}s"
        )

        # Print to terminal
        print(
            f"\n{'='*60}\n"
            f"Experiment complete: {self.experiment_id}\n"
            f"  Final accuracy : {final_accuracy:.2f}%\n"
            f"  Final loss     : {final_loss:.6f}\n"
            f"  Best accuracy  : {summary['best_accuracy']:.2f}% "
            f"(round {summary['best_round']})\n"
            f"  Total duration : {total_duration:.1f}s\n"
            f"  Results saved  : {self.output_dir}/\n"
            f"{'='*60}\n"
        )

        return summary