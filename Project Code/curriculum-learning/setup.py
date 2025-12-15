"""
Setup and Configuration for Curriculum Learning Project

Quick start guide and configuration templates.
"""

import os
from pathlib import Path


# Project structure
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"


def setup_directories():
    """Create all necessary directories."""
    for directory in [
        SRC_DIR,
        CONFIG_DIR,
        SCRIPTS_DIR,
        EXPERIMENTS_DIR,
        CHECKPOINTS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def get_config():
    """Get default configuration."""
    return {
        # Training parameters
        "training": {
            "total_steps": 1000000,
            "num_processes": 4,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
        # Environment parameters
        "environment": {
            "max_steps": 100,
            "target_speed_range": [0.0, 0.89],
            "floor_size_range": [2.0, 24.0],
            "reach_threshold": 0.5,
        },
        # Progression function
        "progression": {
            "type": "friction",  # linear, exponential, friction, none
            "friction": {
                "min_performance": 0.1,
                "friction_coeff": 0.5,
            },
            "exponential": {
                "slopes": [0.1, 0.73, 1.37, 2.0],
            },
        },
        # Paths
        "paths": {
            "checkpoint_dir": str(CHECKPOINTS_DIR),
            "log_dir": str(LOGS_DIR),
            "results_dir": str(RESULTS_DIR),
        },
        # Logging
        "logging": {
            "verbose": 1,
            "save_interval": 5000,
            "log_interval": 1000,
        },
    }


# Quick start commands
QUICK_START_COMMANDS = """
# Quick Start Guide

## Installation

1. Create virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

2. Install dependencies:
   pip install -r requirements.txt

3. (Optional) Set up MuJoCo:
   Download license from https://www.deepmind.com/documents/314/mujoco_license.txt
   Save to ~/.mujoco/mjkey.txt

## Training

### Friction-Based (Adaptive) Progression - RECOMMENDED
python scripts/train.py --progression friction --total-steps 1000000

### Exponential Progression (Fixed)
python scripts/train.py --progression exponential --total-steps 1000000

### Linear Progression (Fixed)
python scripts/train.py --progression linear --total-steps 1000000

### Baseline (No Curriculum)
python scripts/train.py --progression none --total-steps 1000000

## Advanced Options

# Control parallel environments
python scripts/train.py --progression friction --num-processes 8

# Adjust learning rate
python scripts/train.py --progression friction --learning-rate 1e-4

# Change batch size
python scripts/train.py --progression friction --batch-size 128

# Run only 100k steps
python scripts/train.py --progression friction --total-steps 100000

# Custom directories
python scripts/train.py --progression friction \\
    --checkpoint-dir ./my_checkpoints \\
    --log-dir ./my_logs

## Monitoring Training

# View TensorBoard logs
tensorboard --logdir experiments/logs

# Training logs appear in:
experiments/logs/
experiments/checkpoints/

## Evaluation

# Evaluate trained model
python scripts/evaluate.py --model experiments/checkpoints/final_model.zip --episodes 100

# Plot results
python scripts/plot_results.py --log-dir experiments/logs

## Expected Results

With default settings (1M steps, 4 processes):
- Friction-Based:  ~80% success rate (4-5 hours on GPU)
- Exponential:     ~70% success rate
- No Curriculum:   ~18% success rate
"""


if __name__ == "__main__":
    print(__doc__)
    print(QUICK_START_COMMANDS)
    setup_directories()
    print("\nDirectories created successfully!")
