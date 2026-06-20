#!/usr/bin/env python3
"""
Train and export a literal MovieLens graph recommender for Graph Signal Lab.

The browser app runs the same LightGCN forward pass exported here. The raw
MovieLens files are downloaded into data/raw/ for local training only; the site
ships a derived JSON artifact with learned embeddings, graph edges, selected
real examples, and attribution metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_PATH = DATA_DIR / "experiment-data.json"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_DIR = RAW_DIR / "ml-latest-small"

SEED = 7
POSITIVE_RATING = 4.0
DEPTHS = [0, 1, 2, 4, 6]
TRAIN_DEPTH = 2
GENRE_EDGE_WEIGHT = 0.42
DIM = 24


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_movielens() -> None:
    if (MOVIELENS_DIR / "ratings.csv").exists():
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ml-latest-small.zip"
    if not zip_path.exists():
        print(f"Downloading {MOVIELENS_URL}")
        urlretrieve(MOVIELENS_URL, zip_path)

    extract_dir = RAW_DIR / "_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()

    with ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    unpacked = extract_dir / "ml-latest-small"
    if MOVIELENS_DIR.exists():
        shutil.rmtree(MOVIELENS_DIR)
    unpacked.rename(MOVIELENS_DIR)
    shutil.rmtree(extract_dir)


def load_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:
    download_movielens()
    ratings = pd.read_csv(MOVIELENS_DIR / "ratings.csv")
    movies = pd.read_csv(MOVIELENS_DIR / "movies.csv")
    return ratings, movies


def filter_movielens(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    max_movies: int,
    min_user_positives: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = ratings[ratings.rating >= POSITIVE_RATING]
    top_movies = set(
        positives.groupby("movieId").size().sort_values(ascending=False).head(max_movies).index
    )
    filtered = ratings[ratings.movieId.isin(top_movies)].copy()
    filtered_positives = filtered[filtered.rating >= POSITIVE_RATING]
    user_counts = filtered_positives.groupby("userId").size()
    keep_users = set(user_counts[user_counts >= min_user_positives].index)
    filtered = filtered[filtered.userId.isin(keep_users)].copy()
    filtered_movies = movies[movies.movieId.isin(filtered.movieId.unique())].copy()
    return filtered, filtered_movies


def clean_title(title: str) -> str:
    return re.sub(r"\s+\(\d{4}\)$", "", title).strip()


def short_label(name: str, max_len: int = 9) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", clean_title(name)).strip()
    if not cleaned:
        return name[:max_len]
    words = cleaned.split()
    label = " ".join(words[:2])
    return label[:max_len]


def split_positive_edges(ratings: pd.DataFrame) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    positives = ratings[ratings.rating >= POSITIVE_RATING].sort_values(["userId", "timestamp", "movieId"])
    train_edges: list[tuple[int, int, float]] = []
    val_edges: list[tuple[int, int, float]] = []

    for _, group in positives.groupby("userId"):
        rows = list(group[["userId", "movieId", "rating"]].itertuples(index=False, name=None))
        if len(rows) >= 2:
            *train_rows, val_row = rows
            train_edges.extend((int(u), int(m), float(r)) for u, m, r in train_rows)
            val_edges.append((int(val_row[0]), int(val_row[1]), float(val_row[2])))
        else:
            train_edges.extend((int(u), int(m), float(r)) for u, m, r in rows)

    return train_edges, val_edges


def build_mappings(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], dict[str, int], list[dict], dict[int, dict]]:
    user_ids = sorted(ratings.userId.unique().tolist())
    movie_ids = sorted(ratings.movieId.unique().tolist())
    movie_info = movies.set_index("movieId").to_dict("index")

    nodes: list[dict] = []
    user_to_node: dict[int, int] = {}
    movie_to_node: dict[int, int] = {}
    genre_to_node: dict[str, int] = {}

    for user_id in user_ids:
        user_to_node[int(user_id)] = len(nodes)
        nodes.append({
            "kind": "user",
            "raw_id": int(user_id),
            "name": f"MovieLens user {int(user_id)}",
            "label": f"U{int(user_id)}",
        })

    for movie_id in movie_ids:
        info = movie_info.get(int(movie_id), {"title": f"Movie {movie_id}", "genres": "(no genres listed)"})
        movie_to_node[int(movie_id)] = len(nodes)
        nodes.append({
            "kind": "movie",
            "raw_id": int(movie_id),
            "name": str(info["title"]),
            "label": short_label(str(info["title"])),
            "genres": str(info["genres"]).split("|") if str(info["genres"]) != "(no genres listed)" else [],
        })

    genres = sorted({
        genre
        for genre_text in movies.genres.fillna("(no genres listed)")
        for genre in str(genre_text).split("|")
        if genre and genre != "(no genres listed)"
    })
    for genre in genres:
        genre_to_node[genre] = len(nodes)
        nodes.append({
            "kind": "genre",
            "raw_id": genre,
            "name": genre,
            "label": short_label(genre),
        })

    return user_to_node, movie_to_node, genre_to_node, nodes, movie_info


def build_graph(
    train_edges: list[tuple[int, int, float]],
    movies: pd.DataFrame,
    user_to_node: dict[int, int],
    movie_to_node: dict[int, int],
    genre_to_node: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], str], np.ndarray]:
    undirected: list[tuple[int, int, float, str]] = []
    relation_lookup: dict[tuple[int, int], str] = {}

    for user_id, movie_id, rating in train_edges:
        if user_id not in user_to_node or movie_id not in movie_to_node:
            continue
        u = user_to_node[user_id]
        m = movie_to_node[movie_id]
        weight = 1.0 + 0.08 * (rating - POSITIVE_RATING)
        undirected.append((u, m, weight, f"rated {rating:g}"))

    for row in movies.itertuples(index=False):
        movie_id = int(row.movieId)
        if movie_id not in movie_to_node:
            continue
        movie_node = movie_to_node[movie_id]
        for genre in str(row.genres).split("|"):
            if genre in genre_to_node:
                undirected.append((movie_node, genre_to_node[genre], GENRE_EDGE_WEIGHT, "genre"))

    directed_src: list[int] = []
    directed_dst: list[int] = []
    raw_weight: list[float] = []
    degree = np.zeros(len(user_to_node) + len(movie_to_node) + len(genre_to_node), dtype=np.float64)

    for src, dst, weight, relation in undirected:
        directed_src.extend([src, dst])
        directed_dst.extend([dst, src])
        raw_weight.extend([weight, weight])
        degree[src] += weight
        degree[dst] += weight
        relation_lookup[(src, dst)] = relation
        relation_lookup[(dst, src)] = relation

    src_arr = np.array(directed_src, dtype=np.int64)
    dst_arr = np.array(directed_dst, dtype=np.int64)
    raw_arr = np.array(raw_weight, dtype=np.float32)
    norm = raw_arr / np.sqrt(np.maximum(degree[src_arr] * degree[dst_arr], 1e-12))
    return src_arr, dst_arr, norm.astype(np.float32), relation_lookup, degree


class LightGCN(torch.nn.Module):
    def __init__(self, node_count: int, user_nodes: list[int], movie_nodes: list[int], dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(node_count, dim)
        self.bias = torch.nn.Embedding(node_count, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("user_mask_nodes", torch.tensor(user_nodes, dtype=torch.long))
        self.register_buffer("movie_mask_nodes", torch.tensor(movie_nodes, dtype=torch.long))
        torch.nn.init.normal_(self.embedding.weight, std=0.12)
        torch.nn.init.zeros_(self.bias.weight)

    def propagate(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        weight: torch.Tensor,
        layers: int,
    ) -> list[torch.Tensor]:
        states = [self.embedding.weight]
        current = self.embedding.weight
        for _ in range(layers):
            next_state = torch.zeros_like(current)
            messages = current[src] * weight.unsqueeze(1)
            next_state.index_add_(0, dst, messages)
            states.append(next_state)
            current = next_state
        return states

    def final_embedding(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        weight: torch.Tensor,
        layers: int,
    ) -> torch.Tensor:
        states = self.propagate(src, dst, weight, layers)
        return torch.stack(states, dim=0).mean(dim=0)

    def logits(
        self,
        user_nodes: torch.Tensor,
        movie_nodes: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        dot = (embeddings[user_nodes] * embeddings[movie_nodes]).sum(dim=1)
        return dot + self.bias(user_nodes).squeeze(1) + self.bias(movie_nodes).squeeze(1) + self.global_bias


def sample_negatives(
    users: np.ndarray,
    positive_by_user: dict[int, set[int]],
    movie_node_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = np.empty_like(users)
    movie_count = len(movie_node_ids)
    for index, user_node in enumerate(users):
        positives = positive_by_user[int(user_node)]
        while True:
            movie_node = int(movie_node_ids[int(rng.integers(0, movie_count))])
            if movie_node not in positives:
                negatives[index] = movie_node
                break
    return negatives


def train_model(
    node_count: int,
    user_nodes: list[int],
    movie_nodes: list[int],
    train_pairs: np.ndarray,
    positive_by_user: dict[int, set[int]],
    src_arr: np.ndarray,
    dst_arr: np.ndarray,
    weight_arr: np.ndarray,
    epochs: int,
    lr: float,
) -> LightGCN:
    device = torch.device("cpu")
    model = LightGCN(node_count, user_nodes, movie_nodes, DIM).to(device)
    src = torch.tensor(src_arr, dtype=torch.long, device=device)
    dst = torch.tensor(dst_arr, dtype=torch.long, device=device)
    weight = torch.tensor(weight_arr, dtype=torch.float32, device=device)
    train_users = torch.tensor(train_pairs[:, 0], dtype=torch.long, device=device)
    train_movies = torch.tensor(train_pairs[:, 1], dtype=torch.long, device=device)
    movie_node_ids = np.array(movie_nodes, dtype=np.int64)
    rng = np.random.default_rng(SEED + 101)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(1, epochs + 1):
        model.train()
        neg_movies_np = sample_negatives(train_pairs[:, 0], positive_by_user, movie_node_ids, rng)
        neg_movies = torch.tensor(neg_movies_np, dtype=torch.long, device=device)

        embeddings = model.final_embedding(src, dst, weight, TRAIN_DEPTH)
        pos_logits = model.logits(train_users, train_movies, embeddings)
        neg_logits = model.logits(train_users, neg_movies, embeddings)
        loss = -F.logsigmoid(pos_logits - neg_logits).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                margin = (pos_logits - neg_logits).mean().item()
            print(f"epoch {epoch:03d} loss={loss.item():.4f} margin={margin:.4f}")

    return model


def compute_depth_embeddings(
    model: LightGCN,
    src_arr: np.ndarray,
    dst_arr: np.ndarray,
    weight_arr: np.ndarray,
    max_depth: int,
) -> list[torch.Tensor]:
    model.eval()
    with torch.no_grad():
        src = torch.tensor(src_arr, dtype=torch.long)
        dst = torch.tensor(dst_arr, dtype=torch.long)
        weight = torch.tensor(weight_arr, dtype=torch.float32)
        states = model.propagate(src, dst, weight, max_depth)
        finals = []
        for depth in range(max_depth + 1):
            finals.append(torch.stack(states[: depth + 1], dim=0).mean(dim=0))
    return finals


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def score_pairs(
    model: LightGCN,
    embeddings_by_depth: list[torch.Tensor],
    pairs: np.ndarray,
    depth: int,
) -> np.ndarray:
    with torch.no_grad():
        users = torch.tensor(pairs[:, 0], dtype=torch.long)
        movies = torch.tensor(pairs[:, 1], dtype=torch.long)
        logits = model.logits(users, movies, embeddings_by_depth[depth])
        return torch.sigmoid(logits).cpu().numpy()


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    best_acc = -1.0
    best = 0.5
    for threshold in candidates:
        acc = ((scores >= threshold) == labels).mean()
        if acc > best_acc:
            best_acc = float(acc)
            best = float(threshold)
    return best, best_acc


def choose_negative_pairs(
    val_edges: list[tuple[int, int, float]],
    user_to_node: dict[int, int],
    movie_to_node: dict[int, int],
    positive_by_user: dict[int, set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    movie_nodes = np.array(list(movie_to_node.values()), dtype=np.int64)
    pairs = []
    for user_id, _, _ in val_edges:
        user_node = user_to_node[user_id]
        while True:
            movie_node = int(movie_nodes[int(rng.integers(0, len(movie_nodes)))])
            if movie_node not in positive_by_user[user_node]:
                pairs.append((user_node, movie_node))
                break
    return np.array(pairs, dtype=np.int64)


def node_to_movie_id(movie_to_node: dict[int, int]) -> dict[int, int]:
    return {node: movie_id for movie_id, node in movie_to_node.items()}


def node_to_user_id(user_to_node: dict[int, int]) -> dict[int, int]:
    return {node: user_id for user_id, node in user_to_node.items()}


def depth_scores_for_pair(
    model: LightGCN,
    embeddings_by_depth: list[torch.Tensor],
    user_node: int,
    movie_node: int,
) -> dict[str, float]:
    pair = np.array([[user_node, movie_node]], dtype=np.int64)
    return {
        str(depth): round(float(score_pairs(model, embeddings_by_depth, pair, depth)[0]), 6)
        for depth in DEPTHS
    }


def pick_scenarios(
    model: LightGCN,
    embeddings_by_depth: list[torch.Tensor],
    val_pairs: np.ndarray,
    val_ratings: dict[tuple[int, int], float],
    neg_pairs: np.ndarray,
    threshold: float,
) -> list[dict]:
    positive_scores = {depth: score_pairs(model, embeddings_by_depth, val_pairs, depth) for depth in DEPTHS}
    negative_scores = {depth: score_pairs(model, embeddings_by_depth, neg_pairs, depth) for depth in DEPTHS}

    scenarios: list[dict] = []
    used: set[tuple[int, int, int]] = set()

    def add_positive(metric: np.ndarray, payload: dict) -> None:
        for idx in np.argsort(metric)[::-1]:
            user_node = int(val_pairs[int(idx), 0])
            movie_node = int(val_pairs[int(idx), 1])
            key = (user_node, movie_node, 1)
            if key in used:
                continue
            used.add(key)
            scenarios.append({
                **payload,
                "user_node": user_node,
                "movie_node": movie_node,
                "label": 1,
                "rating": float(val_ratings[(user_node, movie_node)]),
            })
            return

    def add_negative(metric: np.ndarray, payload: dict) -> None:
        for idx in np.argsort(metric)[::-1]:
            user_node = int(neg_pairs[int(idx), 0])
            movie_node = int(neg_pairs[int(idx), 1])
            key = (user_node, movie_node, 0)
            if key in used:
                continue
            used.add(key)
            scenarios.append({
                **payload,
                "user_node": user_node,
                "movie_node": movie_node,
                "label": 0,
                "rating": None,
            })
            return

    lift = positive_scores[2] - positive_scores[0]
    add_positive(lift, {
        "id": "heldout-like",
        "title": "Held-out like",
        "kicker": "link prediction",
        "lesson": "This is a real held-out MovieLens rating. The edge is not in the training graph; the score comes from learned embeddings and neighboring movie/genre structure.",
    })

    one_hop = positive_scores[1] - positive_scores[0]
    add_positive(one_hop, {
        "id": "one-hop-lift",
        "title": "One-hop lift",
        "kicker": "near neighbors",
        "lesson": "The trained graph convolution quickly raises this held-out like once nearby users, movies, and genres start sharing signal.",
    })

    false_alarm = negative_scores[0] - negative_scores[2]
    add_negative(false_alarm, {
        "id": "false-alarm",
        "title": "False alarm",
        "kicker": "negative sample",
        "lesson": "This is an unrated user-movie pair sampled as a negative. The same trained model can lower a tempting candidate when graph context does not support it.",
    })

    over = positive_scores[2] - positive_scores[6]
    add_positive(over, {
        "id": "oversmoothing-check",
        "title": "Oversmoothing check",
        "kicker": "depth sweep",
        "lesson": "The model was trained for two-hop inference. Sweeping deeper layers is literal inference too, but extra propagation can wash out the pair-specific signal.",
    })

    deduped = []
    final_seen: set[tuple[int, int, int]] = set()
    for scenario in scenarios:
        key = (scenario["user_node"], scenario["movie_node"], scenario["label"])
        if key in final_seen:
            continue
        final_seen.add(key)
        scenario["expected_scores"] = depth_scores_for_pair(
            model,
            embeddings_by_depth,
            scenario["user_node"],
            scenario["movie_node"],
        )
        scores = {int(depth): value for depth, value in scenario["expected_scores"].items()}
        scenario["best_depth"] = max(DEPTHS, key=lambda depth: scores[depth] if scenario["label"] else -scores[depth])
        scenario["answer"] = "recommend" if scenario["label"] else "hold"
        scenario["threshold"] = round(threshold, 6)
        deduped.append(scenario)

    return deduped[:4]


def build_display_graph(
    scenario: dict,
    nodes: list[dict],
    movie_to_node_map: dict[int, int],
    user_to_node_map: dict[int, int],
    relation_lookup: dict[tuple[int, int], str],
    degree: np.ndarray,
    train_pairs_by_user: dict[int, list[int]],
    users_by_movie: dict[int, list[int]],
    movie_genres: dict[int, list[int]],
) -> dict:
    user_node = scenario["user_node"]
    movie_node = scenario["movie_node"]
    candidate_genres = movie_genres.get(movie_node, [])
    user_movies = train_pairs_by_user.get(user_node, [])

    shared_movies = [
        movie for movie in user_movies
        if set(movie_genres.get(movie, [])).intersection(candidate_genres)
    ]
    if len(shared_movies) < 3:
        shared_movies.extend(movie for movie in user_movies if movie not in shared_movies)
    shared_movies = shared_movies[:3]

    similar_users = [user for user in users_by_movie.get(movie_node, []) if user != user_node][:2]
    selected = [user_node, movie_node]
    selected.extend(shared_movies)
    selected.extend(candidate_genres[:3])
    selected.extend(similar_users)

    # Preserve order and cap the small graph.
    ordered = []
    seen = set()
    for node in selected:
        if node not in seen:
            ordered.append(node)
            seen.add(node)
    selected = ordered[:10]

    position_slots = {
        user_node: (0.14, 0.48),
        movie_node: (0.82, 0.48),
    }
    movie_slots = [(0.34, 0.22), (0.34, 0.52), (0.34, 0.78)]
    genre_slots = [(0.58, 0.22), (0.58, 0.50), (0.58, 0.78)]
    user_slots = [(0.72, 0.18), (0.72, 0.78)]
    movie_i = genre_i = user_i = 0

    display_nodes = []
    for node in selected:
        info = nodes[node]
        if node not in position_slots:
            if info["kind"] == "movie":
                position_slots[node] = movie_slots[min(movie_i, len(movie_slots) - 1)]
                movie_i += 1
            elif info["kind"] == "genre":
                position_slots[node] = genre_slots[min(genre_i, len(genre_slots) - 1)]
                genre_i += 1
            else:
                position_slots[node] = user_slots[min(user_i, len(user_slots) - 1)]
                user_i += 1
        x, y = position_slots[node]
        description = info["name"]
        if node == user_node:
            description = "Target MovieLens user. The id is anonymized by the dataset."
        elif node == movie_node:
            description = "Candidate movie for this prediction."
        elif info["kind"] == "genre":
            description = "Genre node connected from MovieLens movie metadata."
        display_nodes.append({
            "idx": int(node),
            "id": f"n{int(node)}",
            "type": "item" if info["kind"] == "movie" else "topic" if info["kind"] == "genre" else "user",
            "label": info["label"],
            "name": info["name"],
            "description": description,
            "degree": round(float(degree[node]), 3),
            "x": x,
            "y": y,
        })

    selected_set = set(selected)
    display_edges = []
    edge_seen = set()
    for src in selected:
        neighbors = []
        for dst in selected:
            if src == dst:
                continue
            relation = relation_lookup.get((src, dst))
            if relation:
                key = tuple(sorted((src, dst)))
                if key not in edge_seen:
                    neighbors.append((src, dst, relation))
                    edge_seen.add(key)
        for src_node, dst_node, relation in neighbors:
            display_edges.append({
                "id": f"e{src_node}-{dst_node}",
                "source": int(src_node),
                "target": int(dst_node),
                "relation": relation,
                "weight": 1.0 if relation.startswith("rated") else GENRE_EDGE_WEIGHT,
            })

    # Ensure held-out positive candidates visibly connect through their genres.
    for genre_node in candidate_genres[:3]:
        if genre_node in selected_set:
            key = tuple(sorted((movie_node, genre_node)))
            if key not in edge_seen:
                display_edges.append({
                    "id": f"e{movie_node}-{genre_node}",
                    "source": int(movie_node),
                    "target": int(genre_node),
                    "relation": "genre",
                    "weight": GENRE_EDGE_WEIGHT,
                })
                edge_seen.add(key)

    return {
        "nodes": display_nodes,
        "edges": display_edges,
    }


def rounded_list(values: np.ndarray, decimals: int = 6) -> list[float]:
    return np.round(values.astype(np.float64), decimals).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--max-movies", type=int, default=600)
    parser.add_argument("--min-user-positives", type=int, default=8)
    args = parser.parse_args()

    seed_everything(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_ratings, raw_movies = load_movielens()
    ratings, movies = filter_movielens(
        raw_ratings,
        raw_movies,
        max_movies=args.max_movies,
        min_user_positives=args.min_user_positives,
    )
    train_edges, val_edges = split_positive_edges(ratings)
    user_to_node, movie_to_node, genre_to_node, nodes, movie_info = build_mappings(ratings, movies)

    src_arr, dst_arr, weight_arr, relation_lookup, degree = build_graph(
        train_edges,
        movies,
        user_to_node,
        movie_to_node,
        genre_to_node,
    )

    train_pairs = np.array(
        [(user_to_node[user_id], movie_to_node[movie_id]) for user_id, movie_id, _ in train_edges],
        dtype=np.int64,
    )
    positive_by_user: dict[int, set[int]] = defaultdict(set)
    for user_id, movie_id, _ in train_edges:
        positive_by_user[user_to_node[user_id]].add(movie_to_node[movie_id])
    for user_id, movie_id, _ in val_edges:
        positive_by_user[user_to_node[user_id]].add(movie_to_node[movie_id])

    user_nodes = list(user_to_node.values())
    movie_nodes = list(movie_to_node.values())

    print(
        f"MovieLens latest-small subset: {len(ratings)} ratings, {len(train_edges)} train positives, "
        f"{len(val_edges)} held-out positives, {len(nodes)} graph nodes, {len(src_arr)} directed edges"
    )
    model = train_model(
        len(nodes),
        user_nodes,
        movie_nodes,
        train_pairs,
        positive_by_user,
        src_arr,
        dst_arr,
        weight_arr,
        epochs=args.epochs,
        lr=args.lr,
    )

    embeddings_by_depth = compute_depth_embeddings(model, src_arr, dst_arr, weight_arr, max(DEPTHS))
    val_pairs = np.array(
        [(user_to_node[user_id], movie_to_node[movie_id]) for user_id, movie_id, _ in val_edges],
        dtype=np.int64,
    )
    val_ratings = {
        (user_to_node[user_id], movie_to_node[movie_id]): rating
        for user_id, movie_id, rating in val_edges
    }
    rng = np.random.default_rng(SEED + 202)
    neg_pairs = choose_negative_pairs(val_edges, user_to_node, movie_to_node, positive_by_user, rng)

    pos_scores = score_pairs(model, embeddings_by_depth, val_pairs, TRAIN_DEPTH)
    neg_scores = score_pairs(model, embeddings_by_depth, neg_pairs, TRAIN_DEPTH)
    eval_scores = np.concatenate([pos_scores, neg_scores])
    eval_labels = np.concatenate([np.ones_like(pos_scores, dtype=bool), np.zeros_like(neg_scores, dtype=bool)])
    threshold, accuracy = best_threshold(eval_scores, eval_labels)
    auc_like = float((pos_scores[:, None] > neg_scores[None, :]).mean())
    print(f"validation depth={TRAIN_DEPTH} threshold={threshold:.4f} accuracy={accuracy:.3f} rank_auc={auc_like:.3f}")

    scenarios = pick_scenarios(model, embeddings_by_depth, val_pairs, val_ratings, neg_pairs, threshold)

    train_pairs_by_user: dict[int, list[int]] = defaultdict(list)
    users_by_movie: dict[int, list[int]] = defaultdict(list)
    for user_node, movie_node in train_pairs.tolist():
        train_pairs_by_user[user_node].append(movie_node)
        users_by_movie[movie_node].append(user_node)

    movie_genres: dict[int, list[int]] = {}
    for movie_id, node in movie_to_node.items():
        info = movie_info.get(movie_id, {})
        movie_genres[node] = [
            genre_to_node[genre]
            for genre in str(info.get("genres", "")).split("|")
            if genre in genre_to_node
        ]

    for index, scenario in enumerate(scenarios, start=1):
        scenario["level"] = f"{index:02d}"
        scenario["display"] = build_display_graph(
            scenario,
            nodes,
            movie_to_node,
            user_to_node,
            relation_lookup,
            degree,
            train_pairs_by_user,
            users_by_movie,
            movie_genres,
        )
        user_raw = nodes[scenario["user_node"]]["raw_id"]
        movie_name = nodes[scenario["movie_node"]]["name"]
        if scenario["label"]:
            scenario["prompt"] = (
                f"Predict whether anonymized MovieLens user {user_raw} will like {movie_name}. "
                f"The held-out rating was {scenario['rating']:g} stars."
            )
        else:
            scenario["prompt"] = (
                f"Predict whether anonymized MovieLens user {user_raw} should get {movie_name}. "
                "This sampled pair has no positive rating in the dataset."
            )
        scenario["candidate_name"] = movie_name
        scenario["candidate_meta"] = f"MovieLens user {user_raw}"

    with torch.no_grad():
        initial_embedding = model.embedding.weight.cpu().numpy()
        bias = model.bias.weight.squeeze(1).cpu().numpy()

    payload = {
        "meta": {
            "title": "Graph Signal Lab",
            "dataset": "MovieLens latest-small",
            "dataset_filter": f"top {args.max_movies} positively rated movies; users with at least {args.min_user_positives} positive ratings in the subset",
            "dataset_url": MOVIELENS_URL,
            "dataset_readme": "https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html",
            "citation": "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.",
            "generated_by": "gnn-explainer/train.py",
            "seed": SEED,
            "positive_rating": POSITIVE_RATING,
            "model": "LightGCN link predictor",
            "embedding_dim": DIM,
            "train_depth": TRAIN_DEPTH,
            "depths": DEPTHS,
            "threshold": round(float(threshold), 6),
            "validation_accuracy": round(float(accuracy), 4),
            "validation_rank_auc": round(float(auc_like), 4),
            "train_positive_edges": len(train_edges),
            "heldout_positive_edges": len(val_edges),
            "source_rating_rows": len(raw_ratings),
            "subset_rating_rows": len(ratings),
            "subset_users": int(ratings.userId.nunique()),
            "subset_movies": int(ratings.movieId.nunique()),
            "node_count": len(nodes),
            "directed_edge_count": len(src_arr),
            "note": "Scores are generated by browser-side LightGCN inference from these learned embeddings and graph edges.",
        },
        "graph": {
            "dim": DIM,
            "nodeCount": len(nodes),
            "src": src_arr.astype(int).tolist(),
            "dst": dst_arr.astype(int).tolist(),
            "weight": rounded_list(weight_arr, 6),
            "embedding": rounded_list(initial_embedding.reshape(-1), 6),
            "bias": rounded_list(bias, 6),
            "globalBias": round(float(model.global_bias.detach().cpu().item()), 6),
        },
        "scenarios": scenarios,
    }

    ARTIFACT_PATH.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    size_mb = ARTIFACT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {ARTIFACT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
