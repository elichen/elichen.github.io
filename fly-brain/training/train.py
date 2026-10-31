"""PPO for the fly's takeoff on Nitro (Brax + MJX-Warp), after robotarm/training/train.py.
  OUT=/mnt/c/w/flyfly RUN_ID=<unique> python train.py     settings by environment: NUM_ENVS, NUM_TIMESTEPS, ...
Writes <OUT>/runs/<RUN_ID>/: log.json, checkpoints, params_<step>.pkl.
"""
import functools, json, os, sys, time
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('CUDA_MODULE_LOADING', 'EAGER')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax.numpy as jnp
_clip = jnp.clip
jnp.clip = lambda a, min=None, max=None, *, a_min=None, a_max=None: _clip(a, a_min if a_min is not None else min, a_max if a_max is not None else max)
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
import fly_tasks

OUT = Path(os.environ.get('OUT', '/mnt/c/w/flyfly'))
RUN_ID = os.environ.get('RUN_ID') or time.strftime('%Y%m%d-%H%M%S')
RUN = OUT / 'runs' / RUN_ID
RUN.mkdir(parents=True, exist_ok=False)
Task = getattr(fly_tasks, os.environ.get('TASK', 'Takeoff'))
env, eval_env = Task(), Task()
eval_env._config.naconmax = int(os.environ.get('EVAL_NACON', eval_env._config.naconmax))     # naconmax is total across worlds: the eval needs far fewer, and an unused buffer costs GPU memory
eval_env._config.naccdmax = None if not eval_env._config.naccdmax else min(eval_env._config.naccdmax, eval_env._config.naconmax)
E = lambda k, d, c=int: c(os.environ.get(k.upper(), d))
pp = dict(num_timesteps=E('num_timesteps', 30000000), num_evals=E('num_evals', 11), reward_scaling=1.0, episode_length=env._config.episode_length,
          normalize_observations=True, action_repeat=1, unroll_length=E('unroll_length', 20), num_minibatches=E('num_minibatches', 32), num_updates_per_batch=E('num_updates_per_batch', 4),
          discounting=E('discounting', 0.99, float), learning_rate=E('learning_rate', 3e-4, float), entropy_cost=E('entropy_cost', 1e-3, float), num_envs=E('num_envs', 2048), batch_size=E('batch_size', 256), num_eval_envs=E('num_eval_envs', 128), max_grad_norm=1.0)
net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(256, 256, 128), value_hidden_layer_sizes=(512, 256, 256))
if os.environ.get('RESTORE'):
    pp['restore_checkpoint_path'] = os.environ['RESTORE']
pp['save_checkpoint_path'] = RUN / 'checkpoints'
t0, log = time.time(), []
print(json.dumps({k: str(v) for k, v in pp.items()}), 'action', env.action_size, 'obs', env.observation_size, flush=True)
def progress(step, m):
    rec = {'step': int(step), 't': round(time.time() - t0, 1), **{k.replace('eval/episode_', ''): float(m[k]) for k in m if k.startswith('eval/episode_') and not k.endswith('std')}, 'ep_len': float(m.get('eval/avg_episode_length', 0))}
    log.append(rec); (RUN / 'log.json').write_text(json.dumps(log, indent=1))
    L = max(rec['ep_len'], 1)
    print(f"[{rec['t']:8.1f}s] step={rec['step']:>10} reward={rec.get('reward', 0):7.2f} ep_len={L:.0f} flying={rec.get('flying', 0) / L:.3f} pos={rec.get('r_pos', 0) / L:.3f} ori={rec.get('r_ori', 0) / L:.3f} legs={rec.get('r_legs', 0) / L:.3f} height={rec.get('height', 0) / L:.3f}", flush=True)
def save(step, make_policy, params):
    model.save_params(RUN / f'params_{int(step)}.pkl', params)
_, params, _ = ppo.train(environment=env, eval_env=eval_env, **pp, network_factory=net, seed=E('seed', 1), wrap_env_fn=wrapper.wrap_for_brax_training, progress_fn=progress, policy_params_fn=save)
model.save_params(RUN / 'params_final.pkl', params)
print(f'DONE {time.time() - t0:.1f}s -> {RUN}', flush=True)
