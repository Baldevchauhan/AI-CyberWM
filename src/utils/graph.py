"""
Network Graph Generator and Topology Analyzer.
Constructs NetworkX directed multigraphs from network flow records,
extracts topological metrics (fan-out, degree centrality, flow volumes),
and provides Plotly interactive graph visualizers.
"""

from typing import Dict, Any, List, Optional
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go


class NetworkGraphBuilder:
    """Constructs communication graphs from canonical flow records."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_flows(self, df: pd.DataFrame, max_nodes: int = 40) -> nx.DiGraph:
        """
        Builds directed graph from flows.
        Nodes are IP addresses; Edges represent communication with aggregated bytes and flow counts.
        """
        G = nx.DiGraph()
        if df.empty:
            return G

        # Group by (src_ip, dst_ip)
        grouped = df.groupby(["src_ip", "dst_ip"]).agg({
            "bytes": "sum",
            "packets": "sum",
            "dst_port": lambda x: list(x.unique())[:5],
            "protocol": lambda x: x.mode().iloc[0] if not x.empty else "TCP",
            "label": "max",
            "attack_stage": "max"
        }).reset_index()

        # If too many pairs, take top by total bytes
        if len(grouped) > max_nodes * 2:
            grouped = grouped.sort_values(by="bytes", ascending=False).head(max_nodes * 2)

        for _, row in grouped.iterrows():
            src = str(row["src_ip"])
            dst = str(row["dst_ip"])
            bytes_val = int(row["bytes"])
            packets_val = int(row["packets"])
            ports = row["dst_port"]
            proto = str(row["protocol"])
            is_malicious = int(row["label"])
            stage = int(row["attack_stage"])

            # Add source node
            if not G.has_node(src):
                G.add_node(src, type="host", is_attacker=(is_malicious == 1 and stage > 0))
            # Add destination node
            if not G.has_node(dst):
                G.add_node(dst, type="service", is_target=(is_malicious == 1))

            G.add_edge(
                src, dst,
                bytes=bytes_val,
                packets=packets_val,
                ports=ports,
                protocol=proto,
                is_malicious=is_malicious,
                stage=stage
            )

        self.graph = G
        return G

    def to_plotly_figure(self) -> go.Figure:
        """Renders the NetworkX communication graph as an interactive Plotly figure."""
        if len(self.graph.nodes) == 0:
            fig = go.Figure()
            fig.update_layout(
                title="Network Communication Topology (Empty)",
                template="plotly_dark"
            )
            return fig

        # Compute Spring Layout
        pos = nx.spring_layout(self.graph, k=0.5, iterations=40, seed=42)

        edge_x = []
        edge_y = []
        edge_colors = []

        for edge in self.graph.edges(data=True):
            src, dst, data = edge
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.2, color="#4A5568"),
            hoverinfo="none",
            mode="lines"
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in self.graph.nodes(data=True):
            name, data = node
            x, y = pos[name]
            node_x.append(x)
            node_y.append(y)

            in_deg = self.graph.in_degree(name)
            out_deg = self.graph.out_degree(name)
            is_attk = data.get("is_attacker", False)
            is_tgt = data.get("is_target", False)

            if is_attk:
                node_color.append("#EF4444")  # Red: suspicious / attacker
                node_size.append(18)
                role = "Suspicious Host (Attacker)"
            elif is_tgt:
                node_color.append("#F59E0B")  # Amber: targeted host
                node_size.append(16)
                role = "Target Host"
            else:
                node_color.append("#3B82F6")  # Blue: normal internal/external node
                node_size.append(12)
                role = "Standard Node"

            node_text.append(f"IP: {name}<br>Role: {role}<br>Outbound Links: {out_deg}<br>Inbound Links: {in_deg}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=[str(n).split(".")[-1] if "." in str(n) else str(n) for n in self.graph.nodes()],
            textposition="top center",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=2, color="#1E293B")
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="Network Communication Topology & Pivots",
                    font=dict(size=16, color="#F8FAFC")
                ),
                showlegend=False,
                hovermode="closest",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A",
                margin=dict(b=20, l=10, r=10, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        return fig
