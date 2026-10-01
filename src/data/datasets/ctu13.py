"""
CTU-13 Dataset Adapter.
Normalizes CTU-13 Netflow/binetflow records into the canonical schema.
"""

from typing import Dict, Optional
import pandas as pd
from .base import BaseDatasetAdapter


class CTU13Adapter(BaseDatasetAdapter):
    """Adapter for CTU-13 botnet traffic captures."""

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        super().__init__(column_mapping)

    def load(self, source: str | pd.DataFrame) -> pd.DataFrame:
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = source.copy()

        # Lowercase column names
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})

        rename_map = {
            "starttime": "timestamp",
            "dur": "duration",
            "proto": "protocol",
            "srcaddr": "src_ip",
            "dstaddr": "dst_ip",
            "sport": "src_port",
            "dport": "dst_port",
            "totpkts": "packets",
            "totbytes": "bytes"
        }
        df = df.rename(columns=rename_map)

        # Parse timestamp
        if "timestamp" in df.columns:
            parsed = pd.to_datetime(df["timestamp"], errors="coerce")
            if parsed.notna().any():
                df["timestamp"] = parsed.astype("int64") / 1e9
            else:
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)

        # Labels in CTU-13: e.g. "flow=From-Botnet-V42-UDP-DNS", "flow=Background", "flow=Normal"
        if "label" in df.columns:
            labels_str = df["label"].astype(str).str.lower()
            df["label"] = labels_str.apply(lambda x: 1 if "botnet" in x or "malicious" in x or "attack" in x else 0)
            
            # Map stages: Botnet C2 is Stage 4; Portscan is Stage 1; Normal is Stage 0
            def map_stage(lbl: str) -> int:
                if "botnet" in lbl or "c&c" in lbl or "cc" in lbl:
                    return 4  # Command and Control
                if "scan" in lbl:
                    return 1  # Reconnaissance
                if "ddos" in lbl or "dos" in lbl:
                    return 2  # Initial Access / Disruption
                if "normal" in lbl or "background" in lbl:
                    return 0
                return 1

            df["attack_stage"] = labels_str.apply(map_stage)

        # Parse TCP flags if present in state field
        if "state" in df.columns:
            states = df["state"].astype(str).str.upper()
            df["syn_flag"] = states.apply(lambda s: 1 if "S" in s else 0)
            df["ack_flag"] = states.apply(lambda s: 1 if "A" in s else 0)
            df["rst_flag"] = states.apply(lambda s: 1 if "R" in s else 0)
            df["fin_flag"] = states.apply(lambda s: 1 if "F" in s else 0)

        return self.validate_canonical(df)
