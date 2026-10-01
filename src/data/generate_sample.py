"""
Synthetic Sample Network Traffic Generator.
Generates realistic multi-stage cyber kill chain traffic:
Stage 0 (Normal) -> Stage 1 (Recon) -> Stage 2 (Initial Access) ->
Stage 3 (Lateral Movement) -> Stage 4 (C2 Beaconing) -> Stage 5 (Exfiltration).
Maintains realistic background enterprise traffic throughout the timeline.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import wrpcap


def generate_sample_dataset(
    output_csv_path: str = "data/sample/sample_traffic.csv",
    output_pcap_path: str = "data/sample/sample_traffic.pcap",
    base_timestamp: float = 1700000000.0,
    seed: int = 42
):
    """Generates synthetic multi-stage cyber attack traffic dataset with continuous background flows."""
    random.seed(seed)
    np.random.seed(seed)

    records = []
    current_time = base_timestamp

    internal_hosts = ["192.168.1.10", "192.168.1.15", "192.168.1.20", "192.168.1.25"]
    dmz_server = "192.168.1.100"  # Web server
    internal_dc = "192.168.1.5"   # Domain controller / SMB
    external_clients = ["198.51.100.12", "198.51.100.45", "203.0.113.88"]
    attacker_ext = "203.0.113.195"
    c2_server = "198.51.100.220"

    def add_benign_flows(start_t: float, end_t: float, count: int):
        nonlocal records
        for _ in range(count):
            t = random.uniform(start_t, end_t)
            src = random.choice(internal_hosts + external_clients)
            dst = dmz_server if src in external_clients else random.choice(external_clients)
            proto = "TCP" if random.random() > 0.2 else "UDP"
            sport = random.randint(30000, 60000)
            dport = random.choice([80, 443, 53, 123])
            pkts = random.randint(4, 25)
            bytes_val = pkts * random.randint(64, 1200)
            dur = random.uniform(0.1, 2.5)

            records.append({
                "timestamp": t,
                "src_ip": src,
                "dst_ip": dst,
                "src_port": sport,
                "dst_port": dport,
                "protocol": proto,
                "packets": pkts,
                "bytes": bytes_val,
                "duration": dur,
                "syn_flag": 1 if proto == "TCP" else 0,
                "ack_flag": 1 if proto == "TCP" else 0,
                "rst_flag": 0,
                "fin_flag": 1 if proto == "TCP" and random.random() > 0.5 else 0,
                "psh_flag": 1 if random.random() > 0.6 else 0,
                "urg_flag": 0,
                "label": 0,
                "attack_stage": 0
            })

    total_duration = 450.0  # 45 windows of 10s each
    # Continuous background baseline flows across all 450 seconds
    add_benign_flows(base_timestamp, base_timestamp + total_duration, 450)

    # 1. Reconnaissance attack burst: [80s to 130s]
    scan_ports = [21, 22, 23, 25, 80, 110, 135, 139, 443, 445, 1433, 3306, 3389, 8080, 8443]
    t_recon = base_timestamp + 80.0
    for _ in range(160):
        t_recon += random.uniform(0.05, 0.28)
        records.append({
            "timestamp": t_recon,
            "src_ip": attacker_ext,
            "dst_ip": dmz_server,
            "src_port": random.randint(40000, 65000),
            "dst_port": random.choice(scan_ports),
            "protocol": "TCP",
            "packets": 2,
            "bytes": 128,
            "duration": random.uniform(0.01, 0.05),
            "syn_flag": 1,
            "ack_flag": 0,
            "rst_flag": 1 if random.random() > 0.4 else 0,
            "fin_flag": 0,
            "psh_flag": 0,
            "urg_flag": 0,
            "label": 1,
            "attack_stage": 1
        })

    # 2. Initial Access burst: [180s to 230s]
    t_access = base_timestamp + 180.0
    for _ in range(120):
        t_access += random.uniform(0.1, 0.4)
        pkts = random.randint(15, 60)
        records.append({
            "timestamp": t_access,
            "src_ip": attacker_ext,
            "dst_ip": dmz_server,
            "src_port": random.randint(50000, 65000),
            "dst_port": 443,
            "protocol": "TCP",
            "packets": pkts,
            "bytes": pkts * random.randint(500, 1400),
            "duration": random.uniform(0.5, 3.0),
            "syn_flag": 1,
            "ack_flag": 1,
            "rst_flag": 1 if random.random() > 0.7 else 0,
            "fin_flag": 0,
            "psh_flag": 1,
            "urg_flag": 0,
            "label": 1,
            "attack_stage": 2
        })

    # 3. Lateral Movement burst: [260s to 310s]
    t_lat = base_timestamp + 260.0
    for _ in range(120):
        t_lat += random.uniform(0.1, 0.4)
        pkts = random.randint(10, 45)
        records.append({
            "timestamp": t_lat,
            "src_ip": dmz_server,
            "dst_ip": random.choice([internal_dc, "192.168.1.15", "192.168.1.20"]),
            "src_port": random.randint(45000, 60000),
            "dst_port": random.choice([445, 135, 3389, 22]),
            "protocol": "TCP",
            "packets": pkts,
            "bytes": pkts * random.randint(200, 800),
            "duration": random.uniform(0.2, 1.8),
            "syn_flag": 1,
            "ack_flag": 1,
            "rst_flag": 0,
            "fin_flag": 1 if random.random() > 0.5 else 0,
            "psh_flag": 1,
            "urg_flag": 0,
            "label": 1,
            "attack_stage": 3
        })

    # 4. Command and Control burst: [330s to 380s]
    t_c2 = base_timestamp + 330.0
    for _ in range(25):
        t_c2 += 2.0 + random.uniform(-0.08, 0.08)  # Regular beaconing (low IAT variance)
        records.append({
            "timestamp": t_c2,
            "src_ip": dmz_server,
            "dst_ip": c2_server,
            "src_port": random.randint(52000, 53000),
            "dst_port": 8443,
            "protocol": "TCP",
            "packets": 8,
            "bytes": 640,
            "duration": 0.35,
            "syn_flag": 1,
            "ack_flag": 1,
            "rst_flag": 0,
            "fin_flag": 1,
            "psh_flag": 1,
            "urg_flag": 0,
            "label": 1,
            "attack_stage": 4
        })

    # 5. Exfiltration burst: [390s to 440s]
    t_exfil = base_timestamp + 390.0
    for _ in range(90):
        t_exfil += random.uniform(0.1, 0.5)
        pkts = random.randint(150, 600)
        records.append({
            "timestamp": t_exfil,
            "src_ip": dmz_server,
            "dst_ip": c2_server,
            "src_port": random.randint(54000, 56000),
            "dst_port": 443,
            "protocol": "TCP",
            "packets": pkts,
            "bytes": pkts * 1460,
            "duration": random.uniform(1.0, 5.0),
            "syn_flag": 1,
            "ack_flag": 1,
            "rst_flag": 0,
            "fin_flag": 1 if random.random() > 0.8 else 0,
            "psh_flag": 1,
            "urg_flag": 0,
            "label": 1,
            "attack_stage": 5
        })

    df = pd.DataFrame(records).sort_values(by="timestamp").reset_index(drop=True)

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_pcap_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv_path, index=False)
    print(f"Generated {len(df)} synthetic canonical flow records at: {output_csv_path}")

    # Generate synthetic PCAP packets
    scapy_pkts = []
    for _, row in df.head(200).iterrows():
        ts = float(row["timestamp"])
        ip_layer = IP(src=str(row["src_ip"]), dst=str(row["dst_ip"]), ttl=64)
        if row["protocol"] == "TCP":
            flags = "S" if row["syn_flag"] else ""
            if row["ack_flag"]:
                flags += "A"
            if row["rst_flag"]:
                flags += "R"
            if not flags:
                flags = "PA"
            transport = TCP(sport=int(row["src_port"]), dport=int(row["dst_port"]), flags=flags, window=65535)
        else:
            transport = UDP(sport=int(row["src_port"]), dport=int(row["dst_port"]))

        p = ip_layer / transport
        p.time = ts
        scapy_pkts.append(p)

    try:
        wrpcap(output_pcap_path, scapy_pkts)
        print(f"Generated sample PCAP with {len(scapy_pkts)} packets at: {output_pcap_path}")
    except Exception as e:
        print(f"PCAP write notice: {e}")

    return df


if __name__ == "__main__":
    generate_sample_dataset()
