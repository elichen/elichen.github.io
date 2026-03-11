#!/usr/bin/env python3
"""
Train a small edge-aware GNN on synthetic conversation graphs.

The task is node classification:
predict which participants are likely to feel overloaded in a group conversation.

Each graph contains:
- node features: conversational preferences and sensitivities
- directed edge features: aggregated interaction patterns between participants

Labels depend on graph structure, not just node traits. In particular, a participant's
risk depends on whether the people pressuring them are themselves socially reinforced
by the rest of the group. That hidden two-hop dependency gives the GNN something real
to learn over a node-only baseline.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_PATH = DATA_DIR / "experiment-data.json"

SEED = 7
TRAIN_GRAPHS = 1600
VAL_GRAPHS = 320
TEST_GRAPHS = 320
MAX_EPOCHS = 42
BATCH_SIZE = 32
PATIENCE = 10

NODE_FEATURES = [
    ("expressiveness", "Floor hunger"),
    ("interruptiveness", "Interrupt drive"),
    ("reciprocity", "Reciprocity instinct"),
    ("processing_need", "Processing buffer"),
    ("detail_preference", "Detail appetite"),
    ("resilience", "Repair resilience"),
]

EDGE_FEATURES = [
    ("airtime", "Airtime pressure"),
    ("interrupt", "Interrupt rate"),
    ("support", "Repair support"),
    ("redirect", "Topic redirect"),
    ("long_turn", "Turn length"),
]

FEATURE_KEYS = [key for key, _ in NODE_FEATURES]
EDGE_KEYS = [key for key, _ in EDGE_FEATURES]

NAMES = [
    "Mira", "Jonah", "Sora", "Leah", "Milo", "Noor", "Iris", "Theo", "Vera", "Jules",
    "Cleo", "Nina", "Arlo", "Mae", "Kian", "Rhea", "June", "Ezra", "Lina", "Cole",
    "Asha", "Remy", "Drew", "Tess", "Hugo", "Pia", "Niko", "Elle", "Rafi", "Skye",
]


def clamp01(value: np.ndarray | float) -> np.ndarray | float:
    return np.clip(value, 0.0, 1.0)


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def beta_feature(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.beta(a, b))


def sample_nodes(rng: np.random.Generator, graph_idx: int) -> tuple[list[str], np.ndarray]:
    count = int(rng.integers(5, 9))
    offset = (graph_idx * 5) % len(NAMES)
    names = [NAMES[(offset + i * 3) % len(NAMES)] for i in range(count)]

    features = np.zeros((count, len(FEATURE_KEYS)), dtype=np.float32)
    for i in range(count):
        features[i] = np.array([
            beta_feature(rng, 2.5, 2.1),  # expressiveness
            beta_feature(rng, 1.8, 3.0),  # interruptiveness
            beta_feature(rng, 2.8, 1.8),  # reciprocity
            beta_feature(rng, 2.2, 2.0),  # processing need
            beta_feature(rng, 2.1, 2.1),  # detail preference
            beta_feature(rng, 2.6, 1.9),  # resilience
        ], dtype=np.float32)
    return names, features


def build_interaction_graph(rng: np.random.Generator, graph_idx: int) -> dict:
    names, nodes = sample_nodes(rng, graph_idx)
    count = len(names)

    meeting_intensity = float(0.12 + rng.random() * 0.55)
    airtime = np.zeros((count, count), dtype=np.float32)
    interrupt = np.zeros((count, count), dtype=np.float32)
    support = np.zeros((count, count), dtype=np.float32)
    redirect = np.zeros((count, count), dtype=np.float32)
    long_turn = np.zeros((count, count), dtype=np.float32)

    for src in range(count):
        src_feat = nodes[src]
        for dst in range(count):
            if src == dst:
                continue

            dst_feat = nodes[dst]
            detail_gap = abs(src_feat[4] - dst_feat[4])
            pace_gap = max(0.0, src_feat[0] - (1.0 - 0.7 * dst_feat[3]))
            noise = rng.normal(0.0, 0.045, size=5)

            airtime[src, dst] = clamp01(
                0.28
                + 0.42 * src_feat[0]
                + 0.16 * src_feat[1]
                + 0.14 * (1.0 - src_feat[2])
                + 0.18 * meeting_intensity
                + noise[0]
            )
            interrupt[src, dst] = clamp01(
                0.44 * src_feat[1]
                + 0.18 * src_feat[0]
                + 0.24 * dst_feat[3]
                + 0.14 * detail_gap
                + 0.18 * pace_gap
                + 0.10 * meeting_intensity
                + noise[1]
            )
            support[src, dst] = clamp01(
                0.42 * src_feat[2]
                + 0.22 * (1.0 - src_feat[1])
                + 0.18 * (1.0 - detail_gap)
                + 0.10 * dst_feat[4]
                - 0.09 * src_feat[0]
                - 0.08 * meeting_intensity
                + noise[2]
            )
            redirect[src, dst] = clamp01(
                0.30 * src_feat[1]
                + 0.20 * (1.0 - src_feat[2])
                + 0.22 * detail_gap
                + 0.18 * src_feat[0]
                + 0.12 * meeting_intensity
                + noise[3]
            )
            long_turn[src, dst] = clamp01(
                0.26
                + 0.34 * src_feat[0]
                + 0.24 * src_feat[4]
                + 0.14 * (1.0 - src_feat[2])
                + 0.10 * dst_feat[3]
                + 0.10 * meeting_intensity
                + noise[4]
            )

    floor_control = airtime.sum(axis=1) / max(1, count - 1)
    endorsement = support.sum(axis=0) / max(1, count - 1)
    disruption = (interrupt + redirect).sum(axis=1) / max(1, 2 * (count - 1))
    coalition_pull = (support * endorsement[None, :]).sum(axis=1) / max(1, count - 1)
    social_pressure = clamp01(
        0.24 * floor_control
        + 0.34 * endorsement
        + 0.22 * disruption
        + 0.24 * coalition_pull
        + 0.18 * meeting_intensity
    )

    pair_pressure = (0.35 + 1.05 * social_pressure[:, None]) * (
        0.92 * interrupt
        + 0.75 * redirect
        + 0.30 * long_turn
        + 0.12 * airtime
    )
    pair_relief = support * (0.65 + 0.35 * nodes[:, 2][:, None])

    labels = np.zeros(count, dtype=np.float32)
    latent_probability = np.zeros(count, dtype=np.float32)
    contributors = []
    node_metrics = []

    for dst in range(count):
        incoming_pressure = float(pair_pressure[:, dst].sum() / max(1, count - 1))
        incoming_relief = float(pair_relief[:, dst].sum() / max(1, count - 1))
        pressure_sources = pair_pressure[:, dst].copy()
        pressure_sources[dst] = 0.0
        top_sources = np.argsort(pressure_sources)[-2:][::-1]
        pile_on = float(pressure_sources[top_sources].mean()) if top_sources.size else 0.0
        redirect_load = float(redirect[:, dst].sum() / max(1, count - 1))
        interrupt_load = float(interrupt[:, dst].sum() / max(1, count - 1))
        long_turn_load = float(long_turn[:, dst].sum() / max(1, count - 1))

        strongest_source = float(pressure_sources[top_sources[0]]) if top_sources.size else 0.0
        score = (
            1.28 * nodes[dst, 3] * incoming_pressure
            + 0.94 * nodes[dst, 2] * (0.76 * redirect_load + 0.24 * interrupt_load)
            + 0.94 * (1.0 - nodes[dst, 5]) * strongest_source
            + 0.62 * (1.0 - nodes[dst, 5]) * pile_on
            + 0.28 * (1.0 - nodes[dst, 4]) * long_turn_load
            + 0.55 * meeting_intensity * strongest_source
            - 1.12 * incoming_relief * (0.38 + 0.62 * nodes[dst, 5])
            - 0.16 * nodes[dst, 0]
            + float(rng.normal(0.0, 0.065))
        )

        probability = float(sigmoid(3.7 * (score - 1.14)))
        latent_probability[dst] = probability
        labels[dst] = float(rng.random() < probability)

        contributors.append([
            {
                "source": int(source_idx),
                "weight": float(pressure_sources[source_idx]),
            }
            for source_idx in top_sources
            if int(source_idx) != dst and pressure_sources[source_idx] > 0.0
        ])

        node_metrics.append({
            "incoming_pressure": incoming_pressure,
            "incoming_relief": incoming_relief,
            "redirect_load": redirect_load,
            "interrupt_load": interrupt_load,
            "long_turn_load": long_turn_load,
            "hidden_probability": probability,
            "pressure_gap": incoming_pressure - incoming_relief,
            "strongest_source": strongest_source,
        })

    edge_index = []
    edge_attr = []
    edge_payload = []
    for src in range(count):
        for dst in range(count):
            if src == dst:
                continue
            features = [
                float(airtime[src, dst]),
                float(interrupt[src, dst]),
                float(support[src, dst]),
                float(redirect[src, dst]),
                float(long_turn[src, dst]),
            ]
            edge_index.append([src, dst])
            edge_attr.append(features)
            edge_payload.append({
                "source": src,
                "target": dst,
                "airtime": features[0],
                "interrupt": features[1],
                "support": features[2],
                "redirect": features[3],
                "long_turn": features[4],
                "pressure": float(pair_pressure[src, dst]),
                "relief": float(pair_relief[src, dst]),
            })

    label_names = [names[i] for i, label in enumerate(labels) if label > 0.5]
    dominant_idx = int(np.argmax(social_pressure))
    anchor_idx = int(np.argmax(endorsement))
    overloaded_count = int(labels.sum())
    summary = (
        f"{names[dominant_idx]} holds the floor hardest. "
        f"{names[anchor_idx]} receives the most repair. "
        f"{'Nobody crosses the overload line.' if overloaded_count == 0 else f'{overloaded_count} of {count} participants tip into overload.'}"
    )

    return {
        "id": f"graph-{graph_idx:04d}",
        "names": names,
        "node_features": nodes,
        "edge_index": np.array(edge_index, dtype=np.int64).T,
        "edge_attr": np.array(edge_attr, dtype=np.float32),
        "labels": labels,
        "meeting_intensity": meeting_intensity,
        "social_pressure": social_pressure.astype(np.float32),
        "endorsement": endorsement.astype(np.float32),
        "floor_control": floor_control.astype(np.float32),
        "contributors": contributors,
        "node_metrics": node_metrics,
        "summary": summary,
        "dominant_idx": dominant_idx,
        "anchor_idx": anchor_idx,
        "export_nodes": [
            {
                "id": idx,
                "name": names[idx],
                "features": {
                    key: float(nodes[idx, key_idx])
                    for key_idx, key in enumerate(FEATURE_KEYS)
                },
                "label": int(labels[idx]),
                "social_pressure": float(social_pressure[idx]),
                "endorsement": float(endorsement[idx]),
                "floor_control": float(floor_control[idx]),
                **node_metrics[idx],
            }
            for idx in range(count)
        ],
        "export_edges": edge_payload,
    }


def build_split(rng: np.random.Generator, count: int, offset: int) -> list[dict]:
    return [build_interaction_graph(rng, offset + index) for index in range(count)]


def compute_normalization(graphs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_stack = np.concatenate([graph["node_features"] for graph in graphs], axis=0)
    edge_stack = np.concatenate([graph["edge_attr"] for graph in graphs], axis=0)
    node_mean = node_stack.mean(axis=0)
    node_std = node_stack.std(axis=0) + 1e-6
    edge_mean = edge_stack.mean(axis=0)
    edge_std = edge_stack.std(axis=0) + 1e-6
    return node_mean, node_std, edge_mean, edge_std


def attach_tensors(
    graphs: list[dict],
    node_mean: np.ndarray,
    node_std: np.ndarray,
    edge_mean: np.ndarray,
    edge_std: np.ndarray,
) -> None:
    for graph in graphs:
        graph["x"] = torch.tensor((graph["node_features"] - node_mean) / node_std, dtype=torch.float32)
        graph["edge_x"] = torch.tensor((graph["edge_attr"] - edge_mean) / edge_std, dtype=torch.float32)
        graph["edge_index_t"] = torch.tensor(graph["edge_index"], dtype=torch.long)
        graph["y"] = torch.tensor(graph["labels"], dtype=torch.float32)


def make_batches(graphs: list[dict], batch_size: int, rng: np.random.Generator) -> list[dict]:
    order = rng.permutation(len(graphs))
    batches = []
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch_graphs = [graphs[int(idx)] for idx in indices]

        x_parts = []
        edge_parts = []
        edge_index_parts = []
        y_parts = []
        graph_ptr = []
        offset = 0

        for batch_graph in batch_graphs:
            x_parts.append(batch_graph["x"])
            edge_parts.append(batch_graph["edge_x"])
            edge_index_parts.append(batch_graph["edge_index_t"] + offset)
            y_parts.append(batch_graph["y"])
            graph_ptr.extend([len(graph_ptr)] * batch_graph["x"].shape[0])
            offset += batch_graph["x"].shape[0]

        batches.append({
            "x": torch.cat(x_parts, dim=0),
            "edge_x": torch.cat(edge_parts, dim=0),
            "edge_index": torch.cat(edge_index_parts, dim=1),
            "y": torch.cat(y_parts, dim=0),
            "graph_ids": torch.tensor(graph_ptr, dtype=torch.long),
        })
    return batches


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    labels = labels.float()

    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())

    total = max(1.0, tp + tn + fp + fn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    accuracy = (tp + tn) / total
    positive_rate = float(labels.mean())

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": positive_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


class NodeOnlyMLP(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_x: torch.Tensor) -> torch.Tensor:
        del edge_index, edge_x
        return self.net(x).squeeze(-1)


class EdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_x: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        messages = self.edge_proj(torch.cat([h[src], h[dst], edge_x], dim=-1))
        incoming = torch.zeros_like(h)
        outgoing = torch.zeros_like(h)
        incoming.index_add_(0, dst, messages)
        outgoing.index_add_(0, src, messages)

        in_deg = torch.bincount(dst, minlength=h.shape[0]).clamp(min=1).float().unsqueeze(-1)
        out_deg = torch.bincount(src, minlength=h.shape[0]).clamp(min=1).float().unsqueeze(-1)

        updated = self.update(torch.cat([
            h,
            incoming / in_deg,
            outgoing / out_deg,
        ], dim=-1))
        return self.norm(h + updated)


class ConversationGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        hidden_dim = 48
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer1 = EdgeMessageLayer(hidden_dim)
        self.layer2 = EdgeMessageLayer(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_x: torch.Tensor) -> torch.Tensor:
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_x)
        h = self.layer1(h, edge_index, e)
        h = self.layer2(h, edge_index, e)
        return self.classifier(h).squeeze(-1)


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict]
    best_metrics: dict


def run_model(
    model: nn.Module,
    graphs: list[dict],
    batch_size: int,
    rng: np.random.Generator,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict]:
    is_training = optimizer is not None
    batches = make_batches(graphs, batch_size=batch_size, rng=rng)
    losses = []
    logits_all = []
    labels_all = []

    if is_training:
        model.train()
    else:
        model.eval()

    for batch in batches:
        if is_training:
            optimizer.zero_grad()

        logits = model(batch["x"], batch["edge_index"], batch["edge_x"])
        loss = criterion(logits, batch["y"])

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        logits_all.append(logits.detach())
        labels_all.append(batch["y"].detach())

    logits_full = torch.cat(logits_all, dim=0)
    labels_full = torch.cat(labels_all, dim=0)
    metrics = classification_metrics(logits_full, labels_full)
    return float(np.mean(losses)), metrics


def train_model(
    name: str,
    model: nn.Module,
    train_graphs: list[dict],
    val_graphs: list[dict],
    pos_weight: float,
) -> TrainResult:
    del name
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    best_state = None
    best_val_f1 = -1.0
    patience_left = PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        train_rng = np.random.default_rng(SEED + epoch * 13)
        train_loss, train_metrics = run_model(
            model, train_graphs, BATCH_SIZE, train_rng, criterion, optimizer=optimizer
        )

        with torch.no_grad():
            val_rng = np.random.default_rng(SEED + epoch * 17)
            val_loss, val_metrics = run_model(
                model, val_graphs, BATCH_SIZE, val_rng, criterion, optimizer=None
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_record)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return TrainResult(model=model, history=history, best_metrics=best_metrics)


def predict_graphs(model: nn.Module, graphs: list[dict]) -> list[np.ndarray]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for graph in graphs:
            logits = model(graph["x"], graph["edge_index_t"], graph["edge_x"])
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return predictions


def summarize_subset(predictions: list[np.ndarray], graphs: list[dict]) -> dict:
    logits = []
    labels = []
    hard_cases_pred = []
    hard_cases_label = []

    for graph, probs in zip(graphs, predictions, strict=True):
        labels.extend(graph["labels"].tolist())
        logits.extend(probs.tolist())
        for node, prob in enumerate(probs):
            strongest_pressure = max((item["weight"] for item in graph["contributors"][node]), default=0.0)
            if strongest_pressure > 1.45:
                hard_cases_pred.append(prob)
                hard_cases_label.append(graph["labels"][node])

    overall = classification_metrics(
        torch.tensor(logits, dtype=torch.float32).logit(eps=1e-5),
        torch.tensor(labels, dtype=torch.float32),
    )

    if hard_cases_pred:
        hard = classification_metrics(
            torch.tensor(hard_cases_pred, dtype=torch.float32).logit(eps=1e-5),
            torch.tensor(hard_cases_label, dtype=torch.float32),
        )
    else:
        hard = {key: 0.0 for key in ("accuracy", "precision", "recall", "f1")}

    return {
        "overall": overall,
        "high_pressure_subset": {
            "accuracy": hard["accuracy"],
            "f1": hard["f1"],
            "count": len(hard_cases_pred),
        },
    }


def select_samples(
    graphs: list[dict],
    gnn_probs: list[np.ndarray],
    mlp_probs: list[np.ndarray],
    count: int = 12,
) -> list[dict]:
    scored = []
    for graph, gnn, mlp in zip(graphs, gnn_probs, mlp_probs, strict=True):
        labels = graph["labels"]
        gnn_acc = float(((gnn >= 0.5) == labels).mean())
        mlp_acc = float(((mlp >= 0.5) == labels).mean())
        central_pressure = float(np.max([
            max((item["weight"] for item in contributor), default=0.0)
            for contributor in graph["contributors"]
        ]))
        interestingness = (gnn_acc - mlp_acc) + 0.35 * central_pressure + 0.12 * graph["meeting_intensity"]
        scored.append((interestingness, graph, gnn, mlp))

    scored.sort(key=lambda item: item[0], reverse=True)
    samples = []
    for _, graph, gnn, mlp in scored[:count]:
        nodes = []
        for node in graph["export_nodes"]:
            idx = node["id"]
            enriched = dict(node)
            enriched["gnn_probability"] = float(gnn[idx])
            enriched["mlp_probability"] = float(mlp[idx])
            enriched["gnn_prediction"] = int(gnn[idx] >= 0.5)
            enriched["mlp_prediction"] = int(mlp[idx] >= 0.5)
            enriched["contributors"] = graph["contributors"][idx]
            nodes.append(enriched)

        samples.append({
            "id": graph["id"],
            "title": graph["summary"],
            "meeting_intensity": graph["meeting_intensity"],
            "dominant_name": graph["names"][graph["dominant_idx"]],
            "anchor_name": graph["names"][graph["anchor_idx"]],
            "nodes": nodes,
            "edges": graph["export_edges"],
        })
    return samples


def export_artifacts(
    train_result_gnn: TrainResult,
    train_result_mlp: TrainResult,
    test_graphs: list[dict],
    gnn_probs: list[np.ndarray],
    mlp_probs: list[np.ndarray],
) -> None:
    gnn_summary = summarize_subset(gnn_probs, test_graphs)
    mlp_summary = summarize_subset(mlp_probs, test_graphs)

    payload = {
        "meta": {
            "title": "Conversation GNN",
            "seed": SEED,
            "train_graphs": TRAIN_GRAPHS,
            "val_graphs": VAL_GRAPHS,
            "test_graphs": TEST_GRAPHS,
            "node_features": [{"key": key, "label": label} for key, label in NODE_FEATURES],
            "edge_features": [{"key": key, "label": label} for key, label in EDGE_FEATURES],
            "task": "Predict which participants become socially overloaded in a synthetic group conversation.",
        },
        "metrics": {
            "gnn": gnn_summary,
            "mlp": mlp_summary,
        },
        "histories": {
            "gnn": train_result_gnn.history,
            "mlp": train_result_mlp.history,
        },
        "samples": select_samples(test_graphs, gnn_probs, mlp_probs, count=14),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    train_graphs = build_split(rng, TRAIN_GRAPHS, 0)
    val_graphs = build_split(rng, VAL_GRAPHS, TRAIN_GRAPHS)
    test_graphs = build_split(rng, TEST_GRAPHS, TRAIN_GRAPHS + VAL_GRAPHS)

    node_mean, node_std, edge_mean, edge_std = compute_normalization(train_graphs)
    attach_tensors(train_graphs, node_mean, node_std, edge_mean, edge_std)
    attach_tensors(val_graphs, node_mean, node_std, edge_mean, edge_std)
    attach_tensors(test_graphs, node_mean, node_std, edge_mean, edge_std)

    train_labels = np.concatenate([graph["labels"] for graph in train_graphs])
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight = negatives / max(1.0, positives)

    gnn_result = train_model(
        "gnn",
        ConversationGNN(node_dim=len(NODE_FEATURES), edge_dim=len(EDGE_FEATURES)),
        train_graphs,
        val_graphs,
        pos_weight=pos_weight,
    )
    mlp_result = train_model(
        "mlp",
        NodeOnlyMLP(node_dim=len(NODE_FEATURES)),
        train_graphs,
        val_graphs,
        pos_weight=pos_weight,
    )

    gnn_probs = predict_graphs(gnn_result.model, test_graphs)
    mlp_probs = predict_graphs(mlp_result.model, test_graphs)

    export_artifacts(gnn_result, mlp_result, test_graphs, gnn_probs, mlp_probs)

    gnn_metrics = summarize_subset(gnn_probs, test_graphs)
    mlp_metrics = summarize_subset(mlp_probs, test_graphs)
    print("Saved", ARTIFACT_PATH)
    print("GNN accuracy:", round(gnn_metrics["overall"]["accuracy"], 4), "F1:", round(gnn_metrics["overall"]["f1"], 4))
    print("MLP accuracy:", round(mlp_metrics["overall"]["accuracy"], 4), "F1:", round(mlp_metrics["overall"]["f1"], 4))
    print(
        "High-pressure subset F1:",
        "GNN", round(gnn_metrics["high_pressure_subset"]["f1"], 4),
        "MLP", round(mlp_metrics["high_pressure_subset"]["f1"], 4),
    )


if __name__ == "__main__":
    main()
