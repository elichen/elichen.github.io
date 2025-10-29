import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch

class GradientMonitorCallback(BaseCallback):
    """Callback to monitor gradient clipping and report statistics"""

    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.grad_norms = []
        self.clip_count = 0
        self.total_updates = 0

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        # Only check periodically to avoid overhead
        if self.num_timesteps % self.check_freq == 0:
            # Check for NaN in parameters FIRST
            has_nan = False
            for name, p in self.model.policy.named_parameters():
                if torch.isnan(p.data).any():
                    print(f"\n❌ NaN DETECTED in parameter: {name}")
                    has_nan = True
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"\n❌ NaN DETECTED in gradient: {name}")
                    has_nan = True

            if has_nan:
                print(f"❌ CRITICAL: NaN values at timestep {self.num_timesteps}")
                print("Training will fail. Consider:")
                print("- Lower learning rate (current: 3e-5)")
                print("- Stricter gradient clipping (current: 0.3)")
                print("- Check reward scaling")
                raise ValueError("NaN detected in model parameters or gradients")

            # Get gradient norms from the model
            total_norm = 0.0
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5

            self.grad_norms.append(total_norm)
            self.total_updates += 1

            # Check if gradient was clipped (exceeds max_grad_norm)
            if total_norm > self.model.max_grad_norm:
                self.clip_count += 1

            # Log statistics periodically
            if self.verbose > 0 and len(self.grad_norms) >= 10:
                recent_norms = self.grad_norms[-10:]
                clip_rate = self.clip_count / self.total_updates if self.total_updates > 0 else 0
                print(f"\n[Gradient Monitor] Timestep: {self.num_timesteps}")
                print(f"  Avg norm (last 10): {np.mean(recent_norms):.4f}")
                print(f"  Max norm (last 10): {np.max(recent_norms):.4f}")
                print(f"  Clip rate: {clip_rate:.2%} ({self.clip_count}/{self.total_updates})")

                # Warn if clipping is too frequent
                if clip_rate > 0.5:
                    print("  ⚠️ WARNING: High gradient clipping rate - consider adjusting learning rate")
                elif clip_rate > 0.3:
                    print("  ⚠️ Note: Moderate gradient clipping occurring")

    def _on_training_end(self) -> None:
        """Called at the end of training"""
        if self.verbose > 0 and self.grad_norms:
            clip_rate = self.clip_count / self.total_updates if self.total_updates > 0 else 0
            print("\n" + "="*60)
            print("GRADIENT CLIPPING SUMMARY")
            print("="*60)
            print(f"Total gradient updates: {self.total_updates}")
            print(f"Clipped gradients: {self.clip_count} ({clip_rate:.2%})")
            print(f"Average gradient norm: {np.mean(self.grad_norms):.4f}")
            print(f"Max gradient norm: {np.max(self.grad_norms):.4f}")
            print(f"Min gradient norm: {np.min(self.grad_norms):.4f}")

            if clip_rate > 0.5:
                print("\n⚠️ HIGH CLIPPING RATE DETECTED")
                print("Recommendations:")
                print("- Reduce learning rate")
                print("- Increase max_grad_norm")
                print("- Check for numerical instabilities in rewards")
            elif clip_rate > 0.3:
                print("\n⚠️ MODERATE CLIPPING RATE")
                print("Training may benefit from learning rate adjustment")
            else:
                print("\n✓ Gradient clipping within normal range")

    def _on_step(self) -> bool:
        return True