"""BBF's network as trained (configs/BBF.gin), for export.py and the parity references."""
import sys, os
import jax, jax.numpy as jnp
import numpy as np

BBF_WORK = os.environ.get('BBF_WORK', os.path.expanduser('~/code/bbf-work'))
sys.path[:0] = [BBF_WORK, os.path.join(BBF_WORK, 'shims')]
from bigger_better_faster.bbf import spr_networks  # noqa: E402

NUM_ATOMS, VMAX = 51, 10.0
SUPPORT = jnp.linspace(-VMAX, VMAX, NUM_ATOMS)


def network(num_actions):
    return spr_networks.RainbowDQNNetwork(
        num_actions=num_actions, num_atoms=NUM_ATOMS, noisy=False, dueling=True,
        distributional=True, renormalize=True, encoder_type='impala', hidden_dim=2048, width_scale=4)


def q_values(net, params, states):
    """states: uint8 [N, 84, 84, 4] -> Q [N, A], as the agent computes it when acting (eval mode)."""
    def one(s):
        x = spr_networks.process_inputs(s, data_augmentation=False)
        return net.apply(params, x, support=SUPPORT, eval_mode=True, key=jax.random.PRNGKey(0),
                         rngs={'dropout': jax.random.PRNGKey(0)}).q_values
    return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(states)))
