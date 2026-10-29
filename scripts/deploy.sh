#!/usr/bin/env bash

set -e

# Configuration
VM_USER="ubuntu"
VMS=(
    "worker1"
    "worker2"
    "worker3"
    "worker4"
    "worker5"
)

LOCAL_DIR="$(dirname "$(realpath "$0")")/../"
REMOTE_DIR="~/CausalGuard/"

# rsync options
RSYNC_OPTS=(
    -avz
    --exclude ".git/"
    --exclude ".causalguard/"
    --exclude "__pycache__/"
    --exclude "*.pyc"
    --exclude "data/MNIST/"
    --exclude "data/femnist_partitions/" 
    --exclude "data/cifar-10-batches-py/"
    --exclude "data/cifar-100-python/"
    --exclude "data/cifar-10-python.tar.gz"
    --exclude "data/cifar-100-python.tar.gz"
    --exclude "results/"
)

# Deploy functions
deploy_to_vm () {
    VM=$1

    echo "Deploying to $VM..."

    rsync "${RSYNC_OPTS[@]}" \
        "$LOCAL_DIR" \
        "$VM_USER@$VM:$REMOTE_DIR"

    echo "Done: $VM"
    echo ""
}

# Main loop (parallel)
for VM in "${VMS[@]}"; do
    deploy_to_vm "$VM" &
done

wait

echo "Deployment complete!"
echo ""
