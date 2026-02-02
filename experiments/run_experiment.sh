#!/bin/bash
set -e

# 1. Generate Dataset
echo "Generating Dataset..."
# Using the default 0.95 bias
../venv/bin/python generate_dataset.py --data_root experiments/data/colored_mnist --mnist_root experiments/data --bias 0.999 --force

# 2. Check Bias
echo "Verifying Dataset Bias..."
../venv/bin/python check_bias.py --data_root experiments/data/colored_mnist

# 3. Train Model
echo "Training Model..."
../venv/bin/python train.py --data_root experiments/data/colored_mnist --epochs 5 --lr 0.01 --batch_size 64

echo "Experiment Complete."
