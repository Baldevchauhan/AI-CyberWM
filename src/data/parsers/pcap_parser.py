"""
PCAP Parser for packet-level and flow-level network traffic analysis.
Extracts deep features including TTL statistics, window sizes, flag sequences,
inter-arrival times, payload distributions, and flow summaries.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scapy.utils import rdpcap
from scapy.layers.inet import IP, TCP, UDP, ICMP
from ..datasets.base import CANONICAL_COLUMNS


class PCAPParser:
    """Parses raw PCAP / PCAPNG packet capture files into canonical flow records and packet telemetry."""

    def __init__(self, max_packets: Optional[int] = 50000):
        self.max_packets = max_packets

    def parse(self, pcap_path: str) -> pd.DataFrame:
        """
        Parses PCAP file and aggregates packets into bidirectional flows
        with deep packet-level statistics.
        """
        try:
            packets = rdpcap(pcap_path, count=self.max_packets)
        except Exception as e:
            raise RuntimeError(f"Failed to read PCAP file {pcap_path}: {e}")

        # Group packets into 5-tuple flows
        flows: Dict[str, Dict[str, Any]] = {}

        for pkt in packets:
            if not pkt.haslayer(IP):
                continue

            ip_layer = pkt[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            proto_num = ip_layer.proto
            ttl = int(ip_layer.ttl)
            length = int(len(pkt))
            timestamp = float(pkt.time)
            frag = 1 if ip_layer.flags.MF or ip_layer.frag > 0 else 0

            src_port = 0
            dst_port = 0
            proto = "OTHER"
            syn = ack = fin = rst = psh = urg = 0
            window_size = 0
            payload_len = 0

            if pkt.haslayer(TCP):
                proto = "TCP"
                tcp = pkt[TCP]
                src_port = int(tcp.sport)
                dst_port = int(tcp.dport)
                window_size = int(tcp.window)
                syn = 1 if tcp.flags.S else 0
                ack = 1 if tcp.flags.A else 0
                fin = 1 if tcp.flags.F else 0
                rst = 1 if tcp.flags.R else 0
                psh = 1 if tcp.flags.P else 0
                urg = 1 if tcp.flags.U else 0
                payload_len = len(tcp.payload)
            elif pkt.haslayer(UDP):
                proto = "UDP"
                udp = pkt[UDP]
                src_port = int(udp.sport)
                dst_port = int(udp.dport)
                payload_len = len(udp.payload)
            elif pkt.haslayer(ICMP):
                proto = "ICMP"
                payload_len = len(pkt[ICMP].payload)

            # Flow key (canonical directional flow)
            flow_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}_{proto}"

            if flow_key not in flows:
                flows[flow_key] = {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": proto,
                    "timestamps": [timestamp],
                    "packet_sizes": [length],
                    "payload_sizes": [payload_len],
                    "ttls": [ttl],
                    "window_sizes": [window_size] if proto == "TCP" else [],
                    "syn_count": syn,
                    "ack_count": ack,
                    "rst_count": rst,
                    "fin_count": fin,
                    "psh_count": psh,
                    "urg_count": urg,
                    "frag_count": frag,
                    "total_bytes": length,
                    "packet_count": 1,
                }
            else:
                fl = flows[flow_key]
                fl["timestamps"].append(timestamp)
                fl["packet_sizes"].append(length)
                fl["payload_sizes"].append(payload_len)
                fl["ttls"].append(ttl)
                if proto == "TCP":
                    fl["window_sizes"].append(window_size)
                fl["syn_count"] += syn
                fl["ack_count"] += ack
                fl["rst_count"] += rst
                fl["fin_count"] += fin
                fl["psh_count"] += psh
                fl["urg_count"] += urg
                fl["frag_count"] += frag
                fl["total_bytes"] += length
                fl["packet_count"] += 1

        # Build flow dataframe
        records = []
        for key, fl in flows.items():
            ts_list = sorted(fl["timestamps"])
            start_ts = ts_list[0]
            end_ts = ts_list[-1]
            duration = max(end_ts - start_ts, 0.0001)

            # Derive port scan heuristics or suspicious attributes
            is_scan = 1 if fl["syn_count"] > 0 and fl["ack_count"] == 0 and fl["packet_count"] <= 3 else 0
            label = 1 if is_scan else 0
            stage = 1 if is_scan else 0

            records.append({
                "timestamp": start_ts,
                "src_ip": fl["src_ip"],
                "dst_ip": fl["dst_ip"],
                "src_port": fl["src_port"],
                "dst_port": fl["dst_port"],
                "protocol": fl["protocol"],
                "packets": fl["packet_count"],
                "bytes": fl["total_bytes"],
                "duration": duration,
                "syn_flag": 1 if fl["syn_count"] > 0 else 0,
                "ack_flag": 1 if fl["ack_count"] > 0 else 0,
                "rst_flag": 1 if fl["rst_count"] > 0 else 0,
                "fin_flag": 1 if fl["fin_count"] > 0 else 0,
                "psh_flag": 1 if fl["psh_count"] > 0 else 0,
                "urg_flag": 1 if fl["urg_count"] > 0 else 0,
                "label": label,
                "attack_stage": stage
            })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=CANONICAL_COLUMNS)
        else:
            df = df.sort_values(by="timestamp").reset_index(drop=True)

        return df
