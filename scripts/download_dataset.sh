#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Downloading datasets..."
echo ""

python3 << 'EOF'
import sys

from data.mnist_loader import get_mnist_test_loader
from data.cifar10_loader import get_cifar10_test_loader
from data.cifar100_loader import get_cifar100_test_loader

print("Downloading MNIST...")
get_mnist_test_loader('./data')
print("MNIST done.")

print("Downloading CIFAR-10...")
get_cifar10_test_loader('./data')
print("CIFAR-10 done.")

print("Downloading CIFAR-100...")
get_cifar100_test_loader('./data')
print("CIFAR-100 done.")

print("\nAll datasets downloaded successfully.")
EOF