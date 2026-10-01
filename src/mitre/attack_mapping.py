"""
MITRE ATT&CK Knowledge Base and Stage Mapping Engine.
Maps model state predictions and temporal signatures to Enterprise MITRE ATT&CK tactics,
techniques, sub-techniques, and actionable defensive mitigations.
"""

from typing import Dict, Any, List


MITRE_TACTICS_KB = {
    0: {
        "stage_name": "Benign / Normal",
        "tactic_id": "TA0000",
        "tactic_name": "Normal Operations",
        "techniques": [],
        "description": "Network telemetry matches standard baseline operational parameters.",
        "defensive_action": "Routine passive monitoring; maintain standard baseline telemetry.",
        "indicator_signatures": ["Balanced TCP/UDP ratios", "Standard IAT distribution", "Low SYN/RST ratio"]
    },
    1: {
        "stage_name": "Reconnaissance",
        "tactic_id": "TA0043",
        "tactic_name": "Reconnaissance",
        "techniques": [
            {
                "id": "T1046",
                "name": "Network Service Discovery",
                "description": "Adversary is scanning the target network for active services and open ports.",
                "url": "https://attack.mitre.org/techniques/T1046/"
            },
            {
                "id": "T1595.001",
                "name": "Active Scanning: Scanning IP Blocks",
                "description": "Adversary sweeps multiple host IP ranges to identify responsive nodes.",
                "url": "https://attack.mitre.org/techniques/T1595/001/"
            }
        ],
        "description": "The adversary is actively probing network perimeter or internal addresses to identify targets and open ports.",
        "defensive_action": "Enable port-scan rate limiting, isolate unauthenticated sweeps, review edge firewall logs.",
        "indicator_signatures": ["Elevated SYN flag rate", "High unique destination port count", "Rapid connection attempts", "Short flow durations"]
    },
    2: {
        "stage_name": "Initial Access",
        "tactic_id": "TA0001",
        "tactic_name": "Initial Access",
        "techniques": [
            {
                "id": "T1190",
                "name": "Exploit Public-Facing Application",
                "description": "Adversary attempts to leverage a vulnerability or repeated brute-force against exposed ports (HTTP, SSH, RDP).",
                "url": "https://attack.mitre.org/techniques/T1190/"
            },
            {
                "id": "T1133",
                "name": "External Remote Services",
                "description": "Adversary leverages external-facing remote access endpoints (VPNs, Citrix, SSH).",
                "url": "https://attack.mitre.org/techniques/T1133/"
            }
        ],
        "description": "Adversary is attempting to gain initial foothold via targeted exploit delivery, brute force, or credential spraying.",
        "defensive_action": "Verify multi-factor authentication, rate-limit ingress authentication services, block persistent offenders.",
        "indicator_signatures": ["Repetitive connections to single service port", "Surge in inbound payload bytes", "Multiple RST drops"]
    },
    3: {
        "stage_name": "Lateral Movement",
        "tactic_id": "TA0008",
        "tactic_name": "Lateral Movement",
        "techniques": [
            {
                "id": "T1021.002",
                "name": "Remote Services: SMB/Windows Admin Shares",
                "description": "Adversary utilizes SMB/RPC to navigate between internal subnet endpoints.",
                "url": "https://attack.mitre.org/techniques/T1021/002/"
            },
            {
                "id": "T1021.001",
                "name": "Remote Services: Remote Desktop Protocol",
                "description": "Adversary pivots across internal hosts using interactive graphical login sessions.",
                "url": "https://attack.mitre.org/techniques/T1021/001/"
            }
        ],
        "description": "Adversary has compromised an internal host and is actively pivoting east-west across internal subnets.",
        "defensive_action": "Implement internal micro-segmentation, restrict host-to-host SMB/RDP, quarantine source workstation.",
        "indicator_signatures": ["Internal host-to-host flow expansion", "High internal destination IP diversity", "Port 445/3389/22 traversal"]
    },
    4: {
        "stage_name": "Command and Control",
        "tactic_id": "TA0011",
        "tactic_name": "Command and Control",
        "techniques": [
            {
                "id": "T1071.001",
                "name": "Application Layer Protocol: Web Protocols",
                "description": "Adversary communicates with remote C2 infrastructure using HTTP/HTTPS beacons.",
                "url": "https://attack.mitre.org/techniques/T1071/001/"
            },
            {
                "id": "T1095",
                "name": "Non-Application Layer Protocol",
                "description": "Adversary uses custom TCP/UDP channels or ICMP tunnels for command routing.",
                "url": "https://attack.mitre.org/techniques/T1095/"
            }
        ],
        "description": "Compromised systems are establishing periodic or jittered outbound sessions to external attacker infrastructure.",
        "defensive_action": "Sinkhole external destination IP/FQDN, inspect outbound TLS sessions, terminate active sockets.",
        "indicator_signatures": ["Periodic beaconing (low IAT variance)", "Repeated small-payload outbound packets", "Persistent outbound connections"]
    },
    5: {
        "stage_name": "Exfiltration",
        "tactic_id": "TA0010",
        "tactic_name": "Exfiltration",
        "techniques": [
            {
                "id": "T1048.003",
                "name": "Exfiltration Over Alternative Protocol: Unencrypted/Symmetric Protocol",
                "description": "Adversary steals sensitive data by transmitting bulk files over FTP, HTTP, or raw sockets.",
                "url": "https://attack.mitre.org/techniques/T1048/003/"
            },
            {
                "id": "T1567",
                "name": "Exfiltration Over Web Service",
                "description": "Adversary moves target data to external cloud storage or unauthorized hosting sites.",
                "url": "https://attack.mitre.org/techniques/T1567/"
            }
        ],
        "description": "Adversary is exfiltrating intellectual property or credentials out of the internal perimeter.",
        "defensive_action": "Sever external egress connection, engage incident response, trigger data loss prevention forensic review.",
        "indicator_signatures": ["Drastic surge in outbound bytes", "High packet size distribution", "Unbalanced bidirectional byte ratio"]
    }
}


class MITREAttackMapper:
    """Provides MITRE ATT&CK mappings and contextual defensive intelligence."""

    @staticmethod
    def get_stage_info(stage_id: int) -> Dict[str, Any]:
        """Returns MITRE mapping dictionary for a given stage ID."""
        return MITRE_TACTICS_KB.get(int(stage_id), MITRE_TACTICS_KB[0])

    @staticmethod
    def map_prediction(
        predicted_stage: int,
        confidence: float,
        supporting_features: List[str]
    ) -> Dict[str, Any]:
        """
        Creates a structured MITRE report for defender dashboards and alert exports.
        """
        stage_meta = MITREAttackMapper.get_stage_info(predicted_stage)
        
        return {
            "predicted_stage_id": int(predicted_stage),
            "predicted_stage_name": stage_meta["stage_name"],
            "mitre_tactic_id": stage_meta["tactic_id"],
            "mitre_tactic_name": stage_meta["tactic_name"],
            "confidence": round(float(confidence), 4),
            "techniques": stage_meta["techniques"],
            "tactical_description": stage_meta["description"],
            "recommended_defensive_mitigation": stage_meta["defensive_action"],
            "key_indicators": stage_meta["indicator_signatures"],
            "supporting_features": supporting_features,
            "disclaimer": "Model-derived predictive assessment; requires validation against host/endpoint telemetry."
        }
