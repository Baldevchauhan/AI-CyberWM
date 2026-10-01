"""
CLI script to evaluate the World Model and Baseline on chronological test partition.
Usage:
    python scripts/evaluate.py [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate import evaluate_models


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cyber World Model against Baseline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    print(">>> Running comprehensive chronological evaluation...")
    evaluate_models(config_path=args.config)
    print("\n[SUCCESS] Evaluation complete! Check reports/ for CSVs and plots.")


if __name__ == "__main__":
    main()
