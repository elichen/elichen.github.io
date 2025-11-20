#!/usr/bin/env python3
"""
Convert a saved PPO checkpoint into an ONNX graph that can be fed into a
TensorFlow.js pipeline.

Usage:
    python convert_to_tfjs.py \
        --checkpoint sb3_fullmap_model/checkpoints/grid20_4000000_steps.zip \
        --sample sb3_fullmap_model/sample_observation.json \
        --output snake_policy.onnx
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO


@dataclass
class SampleObservation:
    board: np.ndarray
    stats: np.ndarray


def load_sample(path: Path) -> SampleObservation:
    with path.open("r") as f:
        data = json.load(f)
    board = np.asarray(data["board"], dtype=np.float32)
    stats = np.asarray(data["stats"], dtype=np.float32)
    return SampleObservation(board=board, stats=stats)


def build_stacked_inputs(
    sample: SampleObservation,
    n_stack: int,
) -> Tuple[np.ndarray, np.ndarray]:
    board = np.tile(sample.board, (n_stack, 1, 1))
    stats = np.tile(sample.stats, n_stack)
    return board, stats


class PolicyExportWrapper(nn.Module):
    """Minimal module that outputs policy logits and value estimates."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, board: torch.Tensor, stats: torch.Tensor):
        obs = {"board": board, "stats": stats}
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        latent_vf = self.policy.mlp_extractor.forward_critic(features)
        logits = self.policy.action_net(latent_pi)
        values = self.policy.value_net(latent_vf)
        return logits, values


def main():
    parser = argparse.ArgumentParser(description="Export PPO policy to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to SB3 .zip checkpoint")
    parser.add_argument("--sample", type=Path, default=Path("sb3_fullmap_model/sample_observation.json"))
    parser.add_argument("--output", type=Path, default=Path("snake_policy.onnx"))
    parser.add_argument("--stack", type=int, default=4, help="Number of stacked frames used during training")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device=args.device)
    model.policy.eval()

    sample = load_sample(args.sample)
    board_np, stats_np = build_stacked_inputs(sample, args.stack)
    board_tensor = torch.from_numpy(board_np).unsqueeze(0)
    stats_tensor = torch.from_numpy(stats_np).unsqueeze(0)

    wrapper = PolicyExportWrapper(model.policy).to(args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (board_tensor, stats_tensor),
        args.output.as_posix(),
        input_names=["board", "stats"],
        output_names=["logits", "values"],
        dynamic_axes={
            "board": {0: "batch"},
            "stats": {0: "batch"},
            "logits": {0: "batch"},
            "values": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX policy graph saved to {args.output}")
    print(
        "\nNext steps:\n"
        "  onnx-tf convert -i {0} -o snake_policy_tf\n"
        "  tensorflowjs_converter --input_format=tf_saved_model snake_policy_tf snake_tfjs_model\n".format(
            args.output
        )
    )


if __name__ == "__main__":
    main()
