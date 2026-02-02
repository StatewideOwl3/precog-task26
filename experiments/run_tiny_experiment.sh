#!/bin/bash
set -e

# 1. Generate Dataset with weak bias (0.95)
echo "Generating Dataset (Bias 0.95)..."
venv/bin/python experiments/generate_dataset.py --data_root experiments/data/colored_mnist --mnist_root experiments/data --bias 0.95 --force

# 2. Check Bias
echo "Verifying Dataset Bias..."
venv/bin/python experiments/check_bias.py --data_root experiments/data/colored_mnist

# 3. Train Tiny Model
echo "Training TinyCNN..."
venv/bin/python experiments/train.py --data_root experiments/data/colored_mnist --epochs 5 --lr 0.01 --batch_size 64 --model tiny

echo "Experiment Complete."
