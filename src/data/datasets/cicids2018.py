"""
CIC-IDS-2018 Dataset Adapter.
Normalizes CSE-CIC-IDS2018 CSV flow records into canonical schema.
"""

from typing import Dict, Optional
import pandas as pd
from .base import BaseDatasetAdapter


STAGE_MAPPING_CICIDS = {
    "benign": 0,
    "portscan": 1,
    "ftp-bruteforce": 2,
    "ssh-bruteforce": 2,
    "dos attacks-goldeneye": 2,
    "dos attacks-slowloris": 2,
    "dos attacks-slowhttptest": 2,
    "dos attacks-hulk": 2,
    "ddos attacks-loic-http": 2,
    "ddos attack-loic-udp": 2,
    "ddos attack-hoic": 2,
    "brute force -web": 2,
    "brute force -xss": 2,
    "sql injection": 2,
    "infiltration": 3,
    "bot": 4,
    "botnet": 4,
    "exfiltration": 5
}


class CICIDS2018Adapter(BaseDatasetAdapter):
    """Adapter specifically tailored for CSE-CIC-IDS2018 datasets."""

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        super().__init__(column_mapping)

    def load(self, source: str | pd.DataFrame) -> pd.DataFrame:
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = source.copy()

        # Clean column names (strip spaces, lowercase)
        clean_cols = {c: c.strip().lower() for c in df.columns}
        df = df.rename(columns=clean_cols)

        # Standard CIC-IDS2018 columns
        # Dst Port, Protocol, Timestamp, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts,
        # TotLen Fwd Pkts, TotLen Bwd Pkts, SYN Flag Cnt, RST Flag Cnt, ACK Flag Cnt, Label
        rename_dict = {}
        for c in df.columns:
            if "timestamp" in c:
                rename_dict[c] = "timestamp"
            elif "dst port" in c or "dport" in c:
                rename_dict[c] = "dst_port"
            elif "src ip" in c:
                rename_dict[c] = "src_ip"
            elif "dst ip" in c:
                rename_dict[c] = "dst_ip"
            elif "protocol" in c:
                rename_dict[c] = "protocol"
            elif "flow duration" in c:
                rename_dict[c] = "duration"
            elif "tot fwd pkts" in c:
                rename_dict[c] = "fwd_pkts"
            elif "tot bwd pkts" in c:
                rename_dict[c] = "bwd_pkts"
            elif "totlen fwd pkts" in c:
                rename_dict[c] = "fwd_bytes"
            elif "totlen bwd pkts" in c:
                rename_dict[c] = "bwd_bytes"
            elif "syn flag cnt" in c:
                rename_dict[c] = "syn_flag"
            elif "ack flag cnt" in c:
                rename_dict[c] = "ack_flag"
            elif "rst flag cnt" in c:
                rename_dict[c] = "rst_flag"
            elif "fin flag cnt" in c:
                rename_dict[c] = "fin_flag"
            elif "label" in c:
                rename_dict[c] = "label"

        df = df.rename(columns=rename_dict)

        # Compute total packets and bytes if split
        if "packets" not in df.columns:
            f_pkts = df.get("fwd_pkts", 1)
            b_pkts = df.get("bwd_pkts", 0)
            df["packets"] = pd.to_numeric(f_pkts, errors="coerce").fillna(1) + pd.to_numeric(b_pkts, errors="coerce").fillna(0)

        if "bytes" not in df.columns:
            f_bytes = df.get("fwd_bytes", 0)
            b_bytes = df.get("bwd_bytes", 0)
            df["bytes"] = pd.to_numeric(f_bytes, errors="coerce").fillna(0) + pd.to_numeric(b_bytes, errors="coerce").fillna(0)

        # Durations in CIC-IDS are usually microseconds -> convert to seconds
        if "duration" in df.columns:
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0) / 1e6

        # Convert timestamps
        if "timestamp" in df.columns:
            parsed = pd.to_datetime(df["timestamp"], errors="coerce")
            if parsed.notna().any():
                df["timestamp"] = parsed.astype("int64") / 1e9
            else:
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)

        # Map labels and attack stages
        if "label" in df.columns:
            raw_labels = df["label"].astype(str).str.strip().str.lower()
            df["attack_stage"] = raw_labels.map(lambda x: STAGE_MAPPING_CICIDS.get(x, 0 if "benign" in x else 2))
            df["label"] = raw_labels.map(lambda x: 0 if "benign" in x else 1)

        return self.validate_canonical(df)
