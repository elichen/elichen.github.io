import argparse
import os
import torch
import onnx
from stable_baselines3 import PPO

def export_to_onnx(model_path, output_path=None, obs_dim=None):
    model = PPO.load(model_path)
    policy = model.policy
    policy.eval()

    # Auto-detect observation dimension from model
    if obs_dim is None:
        obs_dim = model.observation_space.shape[0]

    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    if output_path is None:
        os.makedirs("models/onnx", exist_ok=True)
        base_name = os.path.basename(model_path).replace('.zip', '')
        output_path = f"models/onnx/{base_name}.onnx"

    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            features = self.policy.extract_features(obs, self.policy.features_extractor)
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            mean_actions = self.policy.action_net(latent_pi)
            return torch.tanh(mean_actions)

    wrapped_policy = PolicyWrapper(policy)
    wrapped_policy.eval()

    torch.onnx.export(wrapped_policy, dummy_input, output_path,
                     export_params=True, opset_version=12, do_constant_folding=True,
                     input_names=["observation"], output_names=["action"],
                     dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}})

    # Load and re-save to ensure weights are embedded
    onnx_model = onnx.load(output_path)
    onnx.save(onnx_model, output_path, save_as_external_data=False)

    print(f"✓ ONNX exported (embedded): {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    export_to_onnx(args.model, args.output)