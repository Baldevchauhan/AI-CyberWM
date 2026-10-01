"""
Temporal Train/Val/Test Splitter.
Guarantees chronological ordering to avoid temporal data leakage.
"""

from typing import Tuple, List, Any
import numpy as np
import pandas as pd


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    timestamp_col: str = "timestamp"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame strictly along the chronological timeline.
    Ensures Train occurs strictly before Validation, which occurs strictly before Test.
    """
    sorted_df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def temporal_sequence_split(
    sequences: np.ndarray,
    targets: np.ndarray,
    stages: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Splits temporal sequence arrays chronologically.
    """
    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = (sequences[:train_end], targets[:train_end], stages[:train_end])
    val = (sequences[train_end:val_end], targets[train_end:val_end], stages[train_end:val_end])
    test = (sequences[val_end:], targets[val_end:], stages[val_end:])

    return train, val, test
