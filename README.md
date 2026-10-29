# Running CausalGuard-FL Experiments

## Prerequisites

All VMs must have completed the initial setup:

```bash
cd ~/CausalGuard
source .venv/bin/activate
```

Run this on every VM before starting any experiment.

---

## Step 1: Deploy Latest Code (Local Machine)

```bash
./scripts/deploy.sh
```

---

## Step 2: Start the Server (VM1 only)

For VectorFed server:

```bash
python3 server.py --algorithm VectorFed --dataset cifar10 --trial 1
```

Wait until you see:

```
Listening on 0.0.0.0:9000
Waiting for 10 clients...
```

**Do not start clients until this message appears.**

---

## Step 3: Start Clients

Open a separate terminal on each VM and run the commands below.
All client commands must be started within **30 seconds** of each other.

### VM2 — Clients 1, 2, 3

```bash
python3 client.py --client_id 1 --algorithm VectorFed --dataset cifar10 --trial 1 &
sleep 3
python3 client.py --client_id 2 --algorithm VectorFed --dataset cifar10 --trial 1 &
sleep 3
python3 client.py --client_id 3 --algorithm VectorFed --dataset cifar10 --trial 1 &
wait
```

### VM3 — Clients 4, 5, 6

```bash
python3 client.py --client_id 4 --algorithm VectorFed --dataset cifar10 --trial 1 &
sleep 3
python3 client.py --client_id 5 --algorithm VectorFed --dataset cifar10 --trial 1 &
sleep 3
python3 client.py --client_id 6 --algorithm VectorFed --dataset cifar10 --trial 1 &
wait
```

### VM4 — Clients 7, 8

```bash
python3 client.py --client_id 7 --algorithm VectorFed --dataset cifar10 --trial 1 &
sleep 3
python3 client.py --client_id 8 --algorithm VectorFed --dataset cifar10 --trial 1 &
wait
```

### VM5 — Clients 9, 10

```bash
python3 client.py --client_id 9  --algorithm VectorFed --dataset cifar10 --trial 1 &
sleep 3
python3 client.py --client_id 10 --algorithm VectorFed --dataset cifar10 --trial 1 &
wait
```

---

## Step 4: Monitor Progress

On VM1, watch the server output live:

```bash
tail -f logs/server_VectorFed_cifar10_trial1.log
```

On any client VM, watch a specific client:

```bash
tail -f logs/client_1_VectorFed_cifar10_trial1.log
```

---

## Step 5: Running Multiple Trials

Repeat Steps 2 and 3 for each trial, incrementing `--trial` each time.
Before starting a new trial, clear the previous results to avoid CSV append issues:

```bash
# On VM1, before each new trial:
rm -f results/vectorfed/cifar10/rounds_trial_1.csv
rm -f results/vectorfed/cifar10/summary_trial_1.json
rm -f results/vectorfed/cifar10/events_trial_1.log
```

---

## Step 6: Kill Leftover Processes (if anything goes wrong)

On any client VM:

```bash
pkill -f "client.py"
```

On VM1:

```bash
pkill -f "server.py"
```

---

## Step 7: Collect Results (Local Machine)

After all trials are complete, pull results from VM1:

```bash
rsync -avz -e "ssh -i ~/.ssh/<your_key.pem>" \
    ubuntu@<VM_IP_address>:~/CausalGuard/results/ \
    ./results/
```

---

## Dataset Reference

| Dataset  | Flag              | Rounds |
|----------|-------------------|--------|
| MNIST    | `--dataset mnist`   | 20   |
| CIFAR-10 | `--dataset cifar10` | 30   |

You can also use `python3 server.py --help` to see the argument parser references.

---

## Sensitivity Analysis (VectorFed only)

To run with a specific gamma value, add `--gamma` to the server command:

```bash
python3 server.py --algorithm VectorFed --dataset cifar10 --trial 1 --gamma 0.5
```

Client commands remain unchanged for sensitivity runs.

---

## Using tmux (Recommended for Long Runs)

SSH sessions will die if your connection drops. Use tmux on every VM:

```bash
# Start a new session
tmux new -s causalguard

# Detach (session keeps running after you disconnect)
Ctrl+b then d

# Reattach after reconnecting
tmux attach -t causalguard
```
