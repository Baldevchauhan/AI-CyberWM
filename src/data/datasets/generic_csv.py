"""
Generic CSV Dataset Adapter.
Accepts user-defined column mappings or auto-detects standard network headers.
"""

from typing import Dict, Optional
import pandas as pd
from .base import BaseDatasetAdapter, CANONICAL_COLUMNS


DEFAULT_CSV_MAPPING = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "ts": "timestamp",
    "src_ip": "src_ip",
    "source_ip": "src_ip",
    "srcip": "src_ip",
    "dst_ip": "dst_ip",
    "destination_ip": "dst_ip",
    "dstip": "dst_ip",
    "src_port": "src_port",
    "sport": "src_port",
    "dst_port": "dst_port",
    "dport": "dst_port",
    "protocol": "protocol",
    "proto": "protocol",
    "packets": "packets",
    "tot_pkts": "packets",
    "bytes": "bytes",
    "tot_bytes": "bytes",
    "duration": "duration",
    "dur": "duration",
    "label": "label",
    "attack_stage": "attack_stage"
}


class GenericCSVAdapter(BaseDatasetAdapter):
    """Adapter for arbitrary CSV files with customizable column mapping."""

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        super().__init__(column_mapping)

    def load(self, source: str | pd.DataFrame) -> pd.DataFrame:
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = source.copy()

        # Normalize column names in incoming dataframe: strip whitespace & lowercase
        col_rename = {}
        incoming_cols = {c.strip().lower(): c for c in df.columns}

        # Apply user mapping first
        for canonical_col, source_col in self.column_mapping.items():
            if source_col in df.columns:
                col_rename[source_col] = canonical_col

        # Auto-match remaining canonical columns from standard names
        for lower_alias, canonical_target in DEFAULT_CSV_MAPPING.items():
            if canonical_target not in col_rename.values() and lower_alias in incoming_cols:
                original_col = incoming_cols[lower_alias]
                col_rename[original_col] = canonical_target

        df = df.rename(columns=col_rename)

        # Handle timestamp parsing if in string/datetime format
        if "timestamp" in df.columns:
            if df["timestamp"].dtype == object:
                parsed_time = pd.to_datetime(df["timestamp"], errors="coerce")
                if parsed_time.notna().any():
                    # convert to epoch seconds
                    df["timestamp"] = parsed_time.astype("int64") / 1e9
                else:
                    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)

        # Map binary label if string (e.g. 'Benign', 'Attack', 'Malicious')
        if "label" in df.columns and df["label"].dtype == object:
            df["label"] = df["label"].apply(
                lambda x: 0 if str(x).strip().lower() in ["benign", "normal", "0", "false"] else 1
            )

        return self.validate_canonical(df)
