"""
Network State Builder.
Aggregates continuous packet and flow streams into fixed-duration time-windowed states S(t).
Computes the canonical 20-dimensional state vector and builds sliding sequence tensors.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .flow_features import compute_inter_arrival_times, compute_port_scan_heuristic


STATE_FEATURE_NAMES = [
    "flow_count",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_dst_ports",
    "tcp_ratio",
    "udp_ratio",
    "syn_rate",
    "ack_rate",
    "rst_rate",
    "fin_rate",
    "avg_packet_size",
    "avg_flow_duration",
    "avg_iat",
    "iat_variance",
    "retransmission_rate",
    "ttl_mean",
    "ttl_variance",
    "port_scan_score",
    "inbound_bytes",
    "outbound_bytes"
]


class NetworkStateBuilder:
    """Builds temporal network state matrices S(t) from canonical flow data."""

    def __init__(self, window_size_seconds: float = 10.0):
        self.window_size = float(window_size_seconds)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def build_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates flows into consecutive time windows of duration `self.window_size`.
        Returns a DataFrame of chronological states S(t).
        """
        if df.empty:
            return pd.DataFrame(columns=["window_start", "window_end", "attack_risk", "attack_stage"] + STATE_FEATURE_NAMES)

        df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
        min_ts = df_sorted["timestamp"].min()
        max_ts = df_sorted["timestamp"].max()

        # Generate window bins
        num_windows = max(int(np.ceil((max_ts - min_ts) / self.window_size)), 1)
        # Ensure at least 1 window
        if num_windows == 1 and max_ts == min_ts:
            max_ts += self.window_size

        window_edges = np.linspace(min_ts, min_ts + num_windows * self.window_size, num_windows + 1)
        df_sorted["window_idx"] = np.digitize(df_sorted["timestamp"], window_edges) - 1
        df_sorted["window_idx"] = df_sorted["window_idx"].clip(0, num_windows - 1)

        state_records = []

        for w_idx in range(num_windows):
            w_start = min_ts + w_idx * self.window_size
            w_end = w_start + self.window_size
            w_flows = df_sorted[df_sorted["window_idx"] == w_idx]

            if w_flows.empty:
                # Quiescent / idle state
                record = {
                    "window_idx": w_idx,
                    "window_start": w_start,
                    "window_end": w_end,
                    "flow_count": 0,
                    "unique_src_ips": 0,
                    "unique_dst_ips": 0,
                    "unique_dst_ports": 0,
                    "tcp_ratio": 0.0,
                    "udp_ratio": 0.0,
                    "syn_rate": 0.0,
                    "ack_rate": 0.0,
                    "rst_rate": 0.0,
                    "fin_rate": 0.0,
                    "avg_packet_size": 0.0,
                    "avg_flow_duration": 0.0,
                    "avg_iat": self.window_size,
                    "iat_variance": 0.0,
                    "retransmission_rate": 0.0,
                    "ttl_mean": 64.0,
                    "ttl_variance": 0.0,
                    "port_scan_score": 0.0,
                    "inbound_bytes": 0,
                    "outbound_bytes": 0,
                    "attack_risk": 0.0,
                    "attack_stage": 0
                }
            else:
                flow_count = len(w_flows)
                u_src = w_flows["src_ip"].nunique()
                u_dst = w_flows["dst_ip"].nunique()
                u_ports = w_flows["dst_port"].nunique()

                protos = w_flows["protocol"].str.upper()
                tcp_cnt = (protos == "TCP").sum()
                udp_cnt = (protos == "UDP").sum()
                tcp_ratio = float(tcp_cnt / flow_count)
                udp_ratio = float(udp_cnt / flow_count)

                syn_rate = float(w_flows["syn_flag"].mean())
                ack_rate = float(w_flows["ack_flag"].mean())
                rst_rate = float(w_flows["rst_flag"].mean())
                fin_rate = float(w_flows["fin_flag"].mean())

                tot_bytes = w_flows["bytes"].sum()
                tot_pkts = max(w_flows["packets"].sum(), 1)
                avg_packet_size = float(tot_bytes / tot_pkts)
                avg_flow_dur = float(w_flows["duration"].mean())

                ts_vals = w_flows["timestamp"].values
                iats = compute_inter_arrival_times(ts_vals)
                avg_iat = float(np.mean(iats))
                iat_var = float(np.var(iats)) if len(iats) > 1 else 0.0

                # Heuristic retransmission rate (short TCP SYN/RST bursts)
                retrans_est = float(((w_flows["syn_flag"] == 1) & (w_flows["duration"] < 0.1)).mean())

                # TTL baseline heuristics
                ttl_mean = 64.0
                ttl_var = 0.0

                # Port scan heuristic
                port_scan = compute_port_scan_heuristic(
                    unique_dst_ports=u_ports,
                    unique_dst_ips=u_dst,
                    flow_count=flow_count,
                    syn_count=int(w_flows["syn_flag"].sum())
                )

                # Bytes distribution (internal vs external heuristics)
                # If src is private subnet (10.x, 192.168.x, 172.16.x), count as outbound
                src_ips = w_flows["src_ip"].astype(str)
                is_internal = src_ips.str.startswith(("10.", "192.168.", "172.16."))
                out_bytes = float(w_flows.loc[is_internal, "bytes"].sum())
                in_bytes = float(w_flows.loc[~is_internal, "bytes"].sum())

                # Labels for ground truth
                malicious_flows = (w_flows["label"] == 1).sum()
                attack_risk = float(malicious_flows / flow_count)

                # Ground truth attack stage: mode of non-zero stages or 0
                active_stages = w_flows.loc[w_flows["attack_stage"] > 0, "attack_stage"]
                if not active_stages.empty:
                    stage = int(active_stages.mode().iloc[0])
                else:
                    stage = 0

                record = {
                    "window_idx": w_idx,
                    "window_start": w_start,
                    "window_end": w_end,
                    "flow_count": flow_count,
                    "unique_src_ips": u_src,
                    "unique_dst_ips": u_dst,
                    "unique_dst_ports": u_ports,
                    "tcp_ratio": tcp_ratio,
                    "udp_ratio": udp_ratio,
                    "syn_rate": syn_rate,
                    "ack_rate": ack_rate,
                    "rst_rate": rst_rate,
                    "fin_rate": fin_rate,
                    "avg_packet_size": avg_packet_size,
                    "avg_flow_duration": avg_flow_dur,
                    "avg_iat": avg_iat,
                    "iat_variance": iat_var,
                    "retransmission_rate": retrans_est,
                    "ttl_mean": ttl_mean,
                    "ttl_variance": ttl_var,
                    "port_scan_score": port_scan,
                    "inbound_bytes": in_bytes,
                    "outbound_bytes": out_bytes,
                    "attack_risk": attack_risk,
                    "attack_stage": stage
                }

            state_records.append(record)

        states_df = pd.DataFrame(state_records)
        return states_df

    def fit_scaler(self, states_df: pd.DataFrame) -> None:
        """Fits standard scaler on training network states."""
        feat_matrix = states_df[STATE_FEATURE_NAMES].values
        self.scaler.fit(feat_matrix)
        self.is_fitted = True

    def transform_states(self, states_df: pd.DataFrame) -> np.ndarray:
        """Scales state features using fitted scaler or identity if not fitted."""
        feat_matrix = states_df[STATE_FEATURE_NAMES].values
        if self.is_fitted:
            return self.scaler.transform(feat_matrix)
        return feat_matrix

    def inverse_transform_state(self, scaled_state: np.ndarray) -> np.ndarray:
        """Reverts scaled state back to raw physical units."""
        if self.is_fitted:
            return self.scaler.inverse_transform(scaled_state)
        return scaled_state

    def create_sliding_sequences(
        self,
        states_df: pd.DataFrame,
        seq_length: int = 6,
        horizon: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms chronological states into training sequence windows:
        X: (N, seq_length, num_features) - Historical trajectory S(t-seq_len+1) ... S(t)
        Y_next: (N, num_features) - Immediate next state S(t+1)
        Y_risk: (N,) - Ground-truth attack risk at S(t+1)
        Y_stage: (N,) - Ground-truth attack stage at S(t+1)
        """
        scaled_features = self.transform_states(states_df)
        risks = states_df["attack_risk"].values
        stages = states_df["attack_stage"].values

        total_steps = len(states_df)
        if total_steps < seq_length + 1:
            # Pad or duplicate if sequence is too short
            pad_len = (seq_length + 1) - total_steps
            scaled_features = np.pad(scaled_features, ((0, pad_len), (0, 0)), mode="edge")
            risks = np.pad(risks, (0, pad_len), mode="edge")
            stages = np.pad(stages, (0, pad_len), mode="edge")
            total_steps = len(scaled_features)

        X_list, Y_next_list, Y_risk_list, Y_stage_list = [], [], [], []

        for i in range(total_steps - seq_length):
            x_seq = scaled_features[i : i + seq_length]
            y_next = scaled_features[i + seq_length]
            y_risk = risks[i + seq_length]
            y_stage = stages[i + seq_length]

            X_list.append(x_seq)
            Y_next_list.append(y_next)
            Y_risk_list.append(y_risk)
            Y_stage_list.append(y_stage)

        return (
            np.array(X_list, dtype=np.float32),
            np.array(Y_next_list, dtype=np.float32),
            np.array(Y_risk_list, dtype=np.float32),
            np.array(Y_stage_list, dtype=np.int64)
        )
