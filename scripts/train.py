"""
CLI script to train the World Model and Baseline.
Usage:
    python scripts/train.py [--config config.yaml] [--data data/sample/sample_traffic.csv]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_world_model import train_world_model
from training.train_baseline import train_baseline


def main():
    parser = argparse.ArgumentParser(description="Train Cyber World Model and Baseline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", type=str, default=None, help="Path to training CSV traffic records")
    args = parser.parse_args()

    print(">>> 1/2 Training Neural World Model...")
    wm_res = train_world_model(config_path=args.config, dataset_csv_path=args.data)

    print("\n>>> 2/2 Training Logistic Regression Baseline...")
    bl_res = train_baseline(config_path=args.config)

    print("\n[SUCCESS] Training completed successfully!")
    print(f"World model checkpoint: {wm_res['checkpoint_path']}")
    print(f"Baseline model: {bl_res['baseline_path']}")


if __name__ == "__main__":
    main()
