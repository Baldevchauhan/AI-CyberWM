"""
Base Dataset Adapter for Network Traffic Ingestion.
Standardizes heterogeneous datasets into a canonical network flow schema.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


CANONICAL_COLUMNS = [
    "timestamp",        # float or datetime epoch seconds
    "src_ip",           # string
    "dst_ip",           # string
    "src_port",         # int
    "dst_port",         # int
    "protocol",         # string (TCP, UDP, ICMP, OTHER)
    "packets",          # int (total packets in flow)
    "bytes",            # int (total bytes in flow)
    "duration",         # float (duration in seconds)
    "syn_flag",         # int (1 if SYN present, 0 otherwise)
    "ack_flag",         # int (1 if ACK present, 0 otherwise)
    "rst_flag",         # int (1 if RST present, 0 otherwise)
    "fin_flag",         # int (1 if FIN present, 0 otherwise)
    "psh_flag",         # int (1 if PSH present, 0 otherwise)
    "urg_flag",         # int (1 if URG present, 0 otherwise)
    "label",            # int (0 for benign, 1 for malicious)
    "attack_stage"      # int (0: Benign, 1: Recon, 2: Initial Access, 3: Lateral Movement, 4: C2, 5: Exfiltration)
]


class BaseDatasetAdapter(ABC):
    """Abstract Base Class for network traffic adapters."""

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        self.column_mapping = column_mapping or {}

    @abstractmethod
    def load(self, source: str | pd.DataFrame) -> pd.DataFrame:
        """
        Loads raw data and maps it to the CANONICAL_COLUMNS schema.
        Must return a pandas DataFrame with canonical columns and sorted by timestamp.
        """
        pass

    def validate_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates and ensures canonical types and fills missing values."""
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                if col in ["syn_flag", "ack_flag", "rst_flag", "fin_flag", "psh_flag", "urg_flag", "label", "attack_stage"]:
                    df[col] = 0
                elif col in ["packets", "bytes"]:
                    df[col] = 1
                elif col in ["duration"]:
                    df[col] = 0.001
                elif col == "protocol":
                    df[col] = "TCP"
                elif col in ["src_port", "dst_port"]:
                    df[col] = 80
                elif col in ["src_ip", "dst_ip"]:
                    df[col] = "0.0.0.0"
                elif col == "timestamp":
                    df[col] = 0.0

        # Enforce types
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)
        df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").fillna(0).astype(int)
        df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype(int)
        df["packets"] = pd.to_numeric(df["packets"], errors="coerce").fillna(1).astype(int)
        df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce").fillna(0).astype(int)
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).astype(float)
        
        flag_cols = ["syn_flag", "ack_flag", "rst_flag", "fin_flag", "psh_flag", "urg_flag", "label", "attack_stage"]
        for f in flag_cols:
            df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0).astype(int)

        # Sort chronologically
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        return df[CANONICAL_COLUMNS]
