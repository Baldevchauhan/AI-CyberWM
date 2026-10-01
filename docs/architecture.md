# AI World Model for Predictive Cyber Defence: Architecture Specification

## 1. Executive Summary

Conventional Network Intrusion Detection Systems (NIDS) are reactive point-in-time classifiers. They operate under the static paradigm:
$$\text{Traffic} \longrightarrow \text{Classifier} \longrightarrow \{\text{Benign}, \text{Malicious}\}$$
Such classifiers lack temporal awareness: they cannot anticipate multi-stage attack progressions along the cyber kill-chain, cannot model how network states evolve, and fail when adversaries introduce temporal delays or jitter to bypass per-flow signatures.

This system introduces an **AI Predictive Network World Model**. Instead of isolated classification, it models the temporal dynamics of the network:
$$P(S(t+1) \mid S(t), S(t-1), \dots, S(t-n+1))$$
By rolling these dynamics forward autoregressively ($S(t) \to \hat{S}(t+1) \to \dots \to \hat{S}(t+K)$), the model forecasts future network states, projects future attack risk probabilities, anticipates next-stage MITRE ATT&CK tactics (e.g., Reconnaissance $\to$ Initial Access $\to$ Lateral Movement $\to$ C2 $\to$ Exfiltration), and provides path-integrated feature and temporal attributions.

---

## 2. Mathematical Formalization

Let the network at any continuous timestamp generate flow and packet telemetry $e_i = (\tau_i, \mathbf{f}_i)$.

### 2.1 State Space Representation
Continuous traffic is discretized into consecutive non-overlapping or sliding time windows of duration $\Delta t \in \{5\text{s}, 10\text{s}, 30\text{s}, 60\text{s}\}$. For window $t$, the aggregation operator $\Phi$ constructs a canonical $D$-dimensional network state vector:
$$S(t) = \Phi(\{e_i \mid t \le \tau_i < t + \Delta t\}) \in \mathbb{R}^{20}$$

The canonical state vector encodes 20 core operational dimensions:
1. `flow_count`: aggregate flow frequency
2. `unique_src_ips`: diversity of source nodes
3. `unique_dst_ips`: diversity of target nodes
4. `unique_dst_ports`: fan-out service access diversity
5. `tcp_ratio`: proportion of transmission control protocol flows
6. `udp_ratio`: proportion of user datagram protocol flows
7. `syn_rate`: SYN flag frequency (connection setup density)
8. `ack_rate`: ACK flag frequency
9. `rst_rate`: RST flag frequency (teardown / rejection density)
10. `fin_rate`: FIN flag frequency
11. `avg_packet_size`: mean packet payload volume
12. `avg_flow_duration`: mean flow lifecycle
13. `avg_iat`: mean flow inter-arrival time
14. `iat_variance`: temporal variance of inter-arrival times (beaconing detector)
15. `retransmission_rate`: estimated network loss / SYN floods
16. `ttl_mean`: mean IP time-to-live
17. `ttl_variance`: IP hop variance
18. `port_scan_score`: normalized port-to-host dispersion heuristic
19. `inbound_bytes`: total ingress traffic volume
20. `outbound_bytes`: total egress traffic volume

### 2.2 Neural World Model Formulation
Given historical trajectory $\mathbf{H}_t = [S(t-n+1), \dots, S(t)] \in \mathbb{R}^{n \times D}$:

1. **Feature Projection**:
   $$h_i = \text{GELU}(\text{LayerNorm}(W_e S(i) + b_e)) \in \mathbb{R}^{H}$$
2. **Temporal Encoder** (LSTM or Transformer with causal attention):
   $$z_t = \text{RNN}(h_{t-n+1:t}) \in \mathbb{R}^{H}$$
3. **Latent Network State Bottleneck**:
   $$z_{\text{latent}} = \text{GELU}(\text{LayerNorm}(W_z z_t + b_z)) \in \mathbb{R}^{L}$$
4. **State Transition Decoder**:
   $$\hat{S}(t+1) = S(t) + \text{MLP}_{\text{trans}}([z_{\text{latent}} \,\|\, S(t)])$$
5. **Attack Risk Head**:
   $$\hat{p}_{\text{attack}}(t+1) = \sigma(W_r z_{\text{latent}} + b_r) \in [0, 1]$$
6. **Attack Progression Stage Head**:
   $$\hat{y}_{\text{stage}}(t+1) = \text{Softmax}(W_s z_{\text{latent}} + b_s) \in \Delta^5$$

### 2.3 Training Objective
The model is trained via multi-task composite loss balancing physical trajectory modeling with security objectives:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{state}} \cdot \text{MSE}(\hat{S}(t+1), S(t+1)) + \lambda_{\text{risk}} \cdot \text{BCE}(\hat{p}_{\text{risk}}, y_{\text{risk}}) + \lambda_{\text{stage}} \cdot \text{CE}(\hat{y}_{\text{stage}}, y_{\text{stage}})$$

---

## 3. Forward Simulation Rollout (K-Step Dynamics)

To evaluate forward trajectories without observing future telemetry, the model autoregressively rolls its latent transitions forward:

$$\begin{aligned}
\mathbf{H}^{(0)} &= [S(t-n+1), \dots, S(t)] \\
\hat{S}(t+1), \hat{p}(t+1), \hat{s}(t+1) &= \mathcal{M}(\mathbf{H}^{(0)}) \\
\mathbf{H}^{(1)} &= [S(t-n+2), \dots, S(t), \hat{S}(t+1)] \\
\hat{S}(t+2), \hat{p}(t+2), \hat{s}(t+2) &= \mathcal{M}(\mathbf{H}^{(1)}) \\
&\;\;\vdots \\
\hat{S}(t+K), \hat{p}(t+K), \hat{s}(t+K) &= \mathcal{M}(\mathbf{H}^{(K-1)})
\end{aligned}$$

This yields a forward **Risk Timeline** $[p(t), \hat{p}(t+1), \dots, \hat{p}(t+K)]$ allowing Security Operations Center (SOC) defenders to intercept attacks **before** exfiltration or damage manifests.

---

## 4. MITRE ATT&CK Kill-Chain Mapping

The system categorizes enterprise progression along 6 discrete stages:

| Stage ID | Cyber Phase | MITRE Tactic ID | Representative Techniques | Key Behavioral Indicators |
|---|---|---|---|---|
| **0** | Benign Normal | TA0000 (Normal Ops) | None | Balanced TCP/UDP, normal IAT variance |
| **1** | Reconnaissance | TA0043 (Reconnaissance) | T1046 (Network Service Discovery), T1595.001 (Active IP Sweeps) | High SYN rate, rapid port fanout, short flows |
| **2** | Initial Access | TA0001 (Initial Access) | T1190 (Exploit Public Application), T1133 (External Remote Services) | High ingress bytes to ports 80/443/22, failed attempts |
| **3** | Lateral Movement | TA0008 (Lateral Movement) | T1021.002 (SMB/Shares), T1021.001 (RDP) | Internal subnet fanout, port 445/135/3389 pivots |
| **4** | Command & Control | TA0011 (Command & Control) | T1071.001 (Web Beaconing), T1095 (Non-App Protocols) | Regular beaconing, ultra-low IAT variance to external IP |
| **5** | Exfiltration | TA0010 (Exfiltration) | T1048 (Alt Protocol Exfil), T1567 (Web Exfiltration) | Massive surge in outbound bytes, large packet payloads |

---

## 5. Explainability via Integrated Gradients

To satisfy strict explainability requirements without claiming unverifiable causality:
1. **Feature Attribution**:
   $$\text{Attr}_j = (x_j - x_j') \times \frac{1}{M} \sum_{k=1}^M \left. \frac{\partial \mathcal{M}_{\text{risk}}}{\partial x_j} \right|_{x' + \frac{k}{M}(x - x')}$$
   where $x'$ is the neutral zero-state baseline.
2. **Temporal Window Attribution**:
   Norm of attributions across the time horizon $\tau \in [t-n+1, \dots, t]$ exposes which historical time window triggered the alert escalation.

---

## 6. System Architecture Diagram

```
┌────────────────────────────────────────────────────────┐
│            Network Telemetry Ingestion                 │
│         CSV / PCAP / NetFlow / CIC-IDS / CTU-13        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               Network State Builder                    │
│      Discretize into Time Windows S(t) in R^20         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Neural Temporal World Model                │
│    FeatureEncoder -> LSTM/Transformer -> LatentState   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             K-Step Autoregressive Rollout              │
│       S(t) -> S_hat(t+1) -> ... -> S_hat(t+K)          │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
                ▼                        ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│     Attack Risk Head      │ │     Attack Stage Head    │
│  P(Attack) in [0, 1]      │ │  MITRE ATT&CK Projection │
└───────────────┬───────────┘ └──────────┬───────────────┘
                │                        │
                └───────────┬────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            Attribution & Explainability                │
│    Integrated Gradients / Feature & Window Ranking     │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│     FastAPI REST      │       │  Streamlit Dashboard  │
│  Offline Microservice │       │  Interactive Cyber UI │
└───────────────────────┘       └───────────────────────┘
```
