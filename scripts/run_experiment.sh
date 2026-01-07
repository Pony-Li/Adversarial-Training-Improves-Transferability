#!/bin/bash

# Usage: ./scripts/run_experiment.sh train [args...]
#        ./scripts/run_experiment.sh finetune [args...]

if [ $# -lt 1 ]; then
    echo "Usage: $0 {train|finetune} [additional args...]"
    exit 1
fi

TASK=$1
shift  # Remove the first argument (task)

# Set default number of GPUs, can be overridden by environment variable
NUM_GPUS=${NUM_GPUS:-8}

if [ "$TASK" = "train" ]; then
    echo "Running adversarial training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS -m src.train "$@"
elif [ "$TASK" = "finetune" ]; then
    echo "Running transfer evaluation with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS -m src.finetune_eval "$@"
else
    echo "Invalid task: $TASK. Use 'train' or 'finetune'."
    exit 1
fi