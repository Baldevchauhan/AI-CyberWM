"""
Flow feature extractors and temporal metrics calculator.
Computes rates, ratios, inter-arrival time statistics, and port distribution metrics.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


def compute_inter_arrival_times(timestamps: np.ndarray) -> np.ndarray:
    """Computes inter-arrival times (IAT) between sorted packet or flow events."""
    if len(timestamps) <= 1:
        return np.array([0.0])
    sorted_ts = np.sort(timestamps)
    iats = np.diff(sorted_ts)
    return iats


def compute_port_scan_heuristic(
    unique_dst_ports: int,
    unique_dst_ips: int,
    flow_count: int,
    syn_count: int
) -> float:
    """
    Computes a normalized port scan risk score [0.0, 1.0].
    High port-to-IP ratio with high SYN rate indicates horizontal/vertical port sweeps.
    """
    if flow_count == 0:
        return 0.0
    
    port_fanout = unique_dst_ports / max(unique_dst_ips, 1)
    syn_density = syn_count / flow_count
    
    # Scale heuristic
    score = (min(port_fanout / 15.0, 1.0) * 0.6) + (syn_density * 0.4)
    return float(np.clip(score, 0.0, 1.0))
