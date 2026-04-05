#!/usr/bin/env python3
"""
Build publication data for the explorable GNN intuition essay.

The app no longer trains a model live. Instead it uses a few small,
hand-authored graph motifs that make message passing legible:

- a one-hop homophily case
- a two-hop witness case
- a bridge graph that shows oversmoothing with depth

Each scene stores a scalar state per node. That scalar stands in for one
channel of a node embedding so the publication can show the mechanics
clearly without burying the reader in vector notation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_PATH = DATA_DIR / "experiment-data.json"

SELF_WEIGHT = 0.35
NEIGHBOR_WEIGHT = 0.65
MAX_DEPTH = 6
LOGIT_SCALE = 10.0


RAW_SCENES = [
    {
        "id": "homophily",
        "figure": "message",
        "name": "Homophily Signal",
        "title": "One layer is enough when neighbors already agree",
        "dek": "The target starts uncertain, but its immediate neighborhood is dominated by orange evidence.",
        "focus_node": "t",
        "recommended_depth": 1,
        "task": "Predict whether the selected node belongs to the orange community.",
        "depth_prompt": "This motif is solved by a one-layer GNN because the decisive evidence is one edge away.",
        "stage_notes": {
            "features": "At depth 0 the target only knows its own feature. A value near 0.50 is almost neutral, so an MLP-like baseline hesitates.",
            "messages": "Each incoming edge scales the sender state. Strong orange neighbors send large positive messages into the target.",
            "aggregate": "The neighborhood summary is a weighted average, not a raw sum. That keeps degree from dominating the update by itself.",
            "update": "The target keeps part of its own feature but mostly follows the neighborhood summary, which pushes it into the orange class.",
            "depth": "A second layer barely changes the answer. Once the first hop is informative, deeper propagation mostly smooths the graph.",
        },
        "depth_notes": {
            "0": "Depth 0 is MLP-like: the target only sees its own ambiguous feature.",
            "1": "After one layer the target absorbs strong orange evidence from its neighbors.",
            "2": "The answer is already stable. Additional depth mostly smooths nearby nodes together.",
            "4": "At four layers the graph is noticeably smoother, but the target remains orange.",
            "6": "The graph is still separable, yet the states are less contrasted than they were at shallow depth.",
        },
        "nodes": [
            {"id": "a", "label": "a", "name": "orange neighbor a", "x": 0.18, "y": 0.22, "initial": 0.92, "group": "orange", "kind": "support"},
            {"id": "b", "label": "b", "name": "orange neighbor b", "x": 0.80, "y": 0.24, "initial": 0.86, "group": "orange", "kind": "support"},
            {"id": "c", "label": "c", "name": "blue neighbor c", "x": 0.26, "y": 0.78, "initial": 0.24, "group": "blue", "kind": "counter"},
            {"id": "d", "label": "d", "name": "blue neighbor d", "x": 0.74, "y": 0.76, "initial": 0.18, "group": "blue", "kind": "counter"},
            {"id": "t", "label": "v", "name": "target node v", "x": 0.50, "y": 0.50, "initial": 0.48, "group": "orange", "kind": "target"},
        ],
        "edges": [
            {"source": "a", "target": "t", "weight": 0.95},
            {"source": "b", "target": "t", "weight": 0.90},
            {"source": "c", "target": "t", "weight": 0.25},
            {"source": "d", "target": "t", "weight": 0.35},
            {"source": "a", "target": "b", "weight": 0.82},
            {"source": "b", "target": "a", "weight": 0.82},
            {"source": "c", "target": "d", "weight": 0.80},
            {"source": "d", "target": "c", "weight": 0.80},
            {"source": "t", "target": "a", "weight": 0.40},
            {"source": "t", "target": "b", "weight": 0.40},
            {"source": "t", "target": "c", "weight": 0.25},
            {"source": "t", "target": "d", "weight": 0.25},
        ],
    },
    {
        "id": "twohop",
        "figure": "message",
        "name": "Two-Hop Witness",
        "title": "Two layers reveal evidence that is hidden two hops away",
        "dek": "The target listens to two relay nodes. Those relays only become informative after they have listened to their own neighborhoods.",
        "focus_node": "t",
        "recommended_depth": 2,
        "task": "Predict whether the selected node should turn orange once two-hop context is included.",
        "depth_prompt": "This motif shows why stacking message-passing layers expands the receptive field.",
        "stage_notes": {
            "features": "The target again starts almost neutral. Looking only at its own feature is not enough to classify it confidently.",
            "messages": "At the first step the target only hears from the relay nodes below the true signal. Their messages are still weak.",
            "aggregate": "The first-hop aggregate is close to neutral because the relays have not yet incorporated the upstream orange sources.",
            "update": "After one layer the relays are prepared, but the target is still uncertain. The useful evidence is one step further out.",
            "depth": "After two layers the target can finally receive the upstream signal through the relays. This is the core advantage of stacked GNN layers.",
        },
        "depth_notes": {
            "0": "Depth 0 sees only the target feature, so the prediction is nearly a coin flip.",
            "1": "After one layer the target has only listened to the relays, which are still close to neutral.",
            "2": "After two layers the orange sources have reached the target through the relay nodes.",
            "4": "More layers keep amplifying the same motif, but they also start smoothing the entire graph.",
            "6": "At six layers the answer is very confident, but the graph is much less locally distinct than it was at depth two.",
        },
        "nodes": [
            {"id": "s1", "label": "s1", "name": "source s1", "x": 0.16, "y": 0.16, "initial": 0.96, "group": "orange", "kind": "source"},
            {"id": "s2", "label": "s2", "name": "source s2", "x": 0.84, "y": 0.18, "initial": 0.92, "group": "orange", "kind": "source"},
            {"id": "u1", "label": "u1", "name": "blue distractor u1", "x": 0.12, "y": 0.56, "initial": 0.18, "group": "blue", "kind": "distractor"},
            {"id": "u2", "label": "u2", "name": "blue distractor u2", "x": 0.88, "y": 0.56, "initial": 0.16, "group": "blue", "kind": "distractor"},
            {"id": "b1", "label": "r1", "name": "relay r1", "x": 0.34, "y": 0.36, "initial": 0.48, "group": "orange", "kind": "relay"},
            {"id": "b2", "label": "r2", "name": "relay r2", "x": 0.66, "y": 0.36, "initial": 0.46, "group": "orange", "kind": "relay"},
            {"id": "l", "label": "l", "name": "low evidence node l", "x": 0.50, "y": 0.82, "initial": 0.22, "group": "blue", "kind": "distractor"},
            {"id": "t", "label": "v", "name": "target node v", "x": 0.50, "y": 0.62, "initial": 0.48, "group": "orange", "kind": "target"},
        ],
        "edges": [
            {"source": "s1", "target": "b1", "weight": 1.00},
            {"source": "u1", "target": "b1", "weight": 0.18},
            {"source": "s2", "target": "b2", "weight": 0.98},
            {"source": "u2", "target": "b2", "weight": 0.16},
            {"source": "b1", "target": "t", "weight": 1.00},
            {"source": "b2", "target": "t", "weight": 0.96},
            {"source": "l", "target": "t", "weight": 0.18},
        ],
    },
    {
        "id": "oversmooth",
        "figure": "depth",
        "name": "Bridge Graph",
        "title": "Extra depth can wash out a boundary",
        "dek": "Two communities are connected by a bridge node. Repeated averaging makes both sides drift toward the same state.",
        "focus_node": "b",
        "recommended_depth": 4,
        "task": "Track how community contrast shrinks as message passing is applied again and again.",
        "depth_prompt": "This motif is not about missing information. It is about too much mixing.",
        "stage_notes": {
            "features": "The two communities start far apart, with the bridge node exactly in the middle.",
            "messages": "Messages flow across the bridge in both directions. Local smoothing now becomes global mixing.",
            "aggregate": "Because the bridge sees both communities, its aggregate stays almost neutral while its neighbors get pulled inward.",
            "update": "Every update shrinks the contrast between the left and right halves of the graph.",
            "depth": "At larger depth the graph becomes smoother and smoother. Distinctions do not disappear instantly, but they keep compressing.",
        },
        "depth_notes": {
            "0": "At depth 0 the orange and blue communities are sharply separated.",
            "1": "One round of averaging mostly smooths within each community.",
            "2": "By two layers the bridge has started to pull both sides inward.",
            "4": "At four layers the two communities are still separable, but the gap has already narrowed substantially.",
            "6": "At six layers the graph is much flatter. This is the start of oversmoothing: node states are becoming too similar.",
        },
        "nodes": [
            {"id": "l1", "label": "l1", "name": "left node l1", "x": 0.12, "y": 0.26, "initial": 0.92, "group": "orange", "kind": "community"},
            {"id": "l2", "label": "l2", "name": "left node l2", "x": 0.28, "y": 0.50, "initial": 0.88, "group": "orange", "kind": "community"},
            {"id": "l3", "label": "l3", "name": "left node l3", "x": 0.12, "y": 0.74, "initial": 0.83, "group": "orange", "kind": "community"},
            {"id": "b", "label": "b", "name": "bridge node b", "x": 0.50, "y": 0.50, "initial": 0.50, "group": "neutral", "kind": "bridge"},
            {"id": "r1", "label": "r1", "name": "right node r1", "x": 0.88, "y": 0.26, "initial": 0.18, "group": "blue", "kind": "community"},
            {"id": "r2", "label": "r2", "name": "right node r2", "x": 0.72, "y": 0.50, "initial": 0.12, "group": "blue", "kind": "community"},
            {"id": "r3", "label": "r3", "name": "right node r3", "x": 0.88, "y": 0.74, "initial": 0.08, "group": "blue", "kind": "community"},
        ],
        "edges": [
            {"source": "l1", "target": "l2", "weight": 0.92},
            {"source": "l2", "target": "l1", "weight": 0.92},
            {"source": "l2", "target": "l3", "weight": 0.90},
            {"source": "l3", "target": "l2", "weight": 0.90},
            {"source": "l2", "target": "b", "weight": 0.72},
            {"source": "b", "target": "l2", "weight": 0.72},
            {"source": "b", "target": "r2", "weight": 0.72},
            {"source": "r2", "target": "b", "weight": 0.72},
            {"source": "r1", "target": "r2", "weight": 0.90},
            {"source": "r2", "target": "r1", "weight": 0.90},
            {"source": "r2", "target": "r3", "weight": 0.92},
            {"source": "r3", "target": "r2", "weight": 0.92},
        ],
    },
]


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def probability_from_state(state: float) -> float:
    return logistic((state - 0.5) * LOGIT_SCALE)


def build_incoming(edges: list[dict]) -> dict[str, list[dict]]:
    incoming: dict[str, list[dict]] = {}
    for edge in edges:
        incoming.setdefault(edge["target"], []).append(edge)
    return incoming


def build_layer_snapshot(depth: int, states: dict[str, float], nodes: list[dict]) -> dict:
    return {
        "depth": depth,
        "nodes": [
            {
                "id": node["id"],
                "state": round(states[node["id"]], 6),
                "probability": round(probability_from_state(states[node["id"]]), 6),
            }
            for node in nodes
        ],
        "spread": round(max(states.values()) - min(states.values()), 6),
    }


def compute_scene(scene: dict) -> dict:
    nodes = scene["nodes"]
    edges = scene["edges"]
    focus_node = scene["focus_node"]
    incoming = build_incoming(edges)
    states = {node["id"]: float(node["initial"]) for node in nodes}

    layers = [build_layer_snapshot(0, states, nodes)]
    transitions = []

    for depth in range(MAX_DEPTH):
        messages = []
        aggregates = []
        updates = []
        next_states: dict[str, float] = {}

        for edge in edges:
            message_value = states[edge["source"]] * edge["weight"]
            messages.append({
                "source": edge["source"],
                "target": edge["target"],
                "weight": round(edge["weight"], 6),
                "value": round(message_value, 6),
            })

        for node in nodes:
            node_id = node["id"]
            incoming_edges = incoming.get(node_id, [])
            if incoming_edges:
                numerator = sum(states[edge["source"]] * edge["weight"] for edge in incoming_edges)
                denominator = sum(edge["weight"] for edge in incoming_edges)
                aggregate = numerator / denominator
            else:
                denominator = 0.0
                aggregate = states[node_id]

            self_term = SELF_WEIGHT * states[node_id]
            neighbor_term = NEIGHBOR_WEIGHT * aggregate
            next_state = self_term + neighbor_term
            next_states[node_id] = next_state

            aggregates.append({
                "id": node_id,
                "value": round(aggregate, 6),
                "normalizer": round(denominator, 6),
            })
            updates.append({
                "id": node_id,
                "self_term": round(self_term, 6),
                "neighbor_term": round(neighbor_term, 6),
                "next_state": round(next_state, 6),
            })

        transitions.append({
            "from_depth": depth,
            "to_depth": depth + 1,
            "messages": messages,
            "aggregates": aggregates,
            "updates": updates,
        })
        states = next_states
        layers.append(build_layer_snapshot(depth + 1, states, nodes))

    depth_metrics = []
    for layer in layers:
        layer_map = {item["id"]: item["state"] for item in layer["nodes"]}
        orange_nodes = [node["id"] for node in nodes if node["group"] == "orange"]
        blue_nodes = [node["id"] for node in nodes if node["group"] == "blue"]
        orange_mean = sum(layer_map[node_id] for node_id in orange_nodes) / max(1, len(orange_nodes))
        blue_mean = sum(layer_map[node_id] for node_id in blue_nodes) / max(1, len(blue_nodes))
        focus_state = layer_map[focus_node]
        depth_metrics.append({
            "depth": layer["depth"],
            "focus_state": round(focus_state, 6),
            "focus_probability": round(probability_from_state(focus_state), 6),
            "spread": layer["spread"],
            "orange_mean": round(orange_mean, 6),
            "blue_mean": round(blue_mean, 6),
            "community_gap": round(orange_mean - blue_mean, 6),
        })

    return {
        **scene,
        "layers": layers,
        "transitions": transitions,
        "depth_metrics": depth_metrics,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scenes = [compute_scene(scene) for scene in RAW_SCENES]
    payload = {
        "meta": {
            "title": "How a Graph Neural Network Sees a Node",
            "subtitle": "An explorable publication about message passing, receptive fields, and oversmoothing.",
            "self_weight": SELF_WEIGHT,
            "neighbor_weight": NEIGHBOR_WEIGHT,
            "max_depth": MAX_DEPTH,
            "logit_scale": LOGIT_SCALE,
            "state_label": "orange confidence",
            "message_equation": "m_{u→v}^{(k)} = w_{uv} · h_u^{(k)}",
            "aggregate_equation": "a_v^{(k)} = Σ_u m_{u→v}^{(k)} / Σ_u w_{uv}",
            "update_equation": "h_v^{(k+1)} = 0.35 · h_v^{(k)} + 0.65 · a_v^{(k)}",
            "note": "The essay uses one scalar feature channel to make message passing visible. Real GNNs usually propagate vectors.",
        },
        "scenes": scenes,
    }
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
