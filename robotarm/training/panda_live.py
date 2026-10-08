"""MuJoCo Playground's PandaPickCube, changed for a live page.

The stock task starts every episode from the home pose with an upright, unrotated
box and a fixed target. On the page the target can be dragged at any time, the box
can be pulled out of the gripper and land any way up, and after each delivery a new
box drops somewhere else while the arm is still out at the old target. So here:

- the box starts resting on any of its 6 faces at a random yaw, or (half the time)
  is dropped from 5-25 cm in a random orientation, the way the page drops it;
- the arm starts near home with every joint jittered by up to 0.25 rad;
- the target jumps somewhere new with probability 0.005 per step (about every 4 s);
- the box respawns the same way with probability 0.002 per step (about every
  10 s), wherever the arm is and even mid-carry;
- episodes are 300 steps (6 s) instead of 150;
- the reward ignores the box's orientation (the target is a position), and
  fine-tuning (BOX_FINE=<scale>) adds a sharp bonus for being within ~1 cm.

Physics, observation (66 numbers) and action (8 position-target deltas) are the
stock ones, so the page's panda-env.js mirrors pick.py plus this file.
"""

from typing import Any, Dict

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import pick

BOX_RANGE = 0.2                  # box x, y within ±0.2 m of (0.5, 0)
TARGET_RANGE = 0.2               # target x, y within ±0.2 m of (0.5, 0) ...
TARGET_Z = (0.2, 0.4)            # ... and 0.2-0.4 m above the box's resting height
UPRIGHT_Z, LYING_Z = 0.03, 0.02  # box half-sizes are 0.02 x 0.02 x 0.03
_S = 0.5 ** 0.5
FACES = jp.array([                # which face is down: (w, x, y, z) quaternions
    [1, 0, 0, 0], [0, 1, 0, 0],   # upright, upside down
    [_S, _S, 0, 0], [_S, -_S, 0, 0], [_S, 0, _S, 0], [_S, 0, -_S, 0],  # on a long side
])


def live_config() -> config_dict.ConfigDict:
  c = pick.default_config()
  c.episode_length = 300
  c.target_jump_prob = 0.005
  c.box_respawn_prob = 0.002
  c.drop_prob = 0.5
  c.arm_jitter = 0.25
  # Fine-tuning adds a sharp bonus for holding the box within ~1 cm of the target
  c.reward_config.scales.box_fine = 0.0
  return c


class PandaPickLive(pick.PandaPickCube):

  def __init__(self, config=None, config_overrides=None):
    super().__init__(config or live_config(), config_overrides)
    m = self._mj_model
    self._obj_dofadr = int(m.jnt_dofadr[m.body("box").jntadr[0]])
    self._arm_lo = jp.array(m.jnt_range[:7, 0])
    self._arm_hi = jp.array(m.jnt_range[:7, 1])

  def _sample_box(self, rng):
    k_xy, k_yaw, k_face, k_drop, k_q, k_h = jax.random.split(rng, 6)
    xy = self._init_obj_pos[:2] + jax.random.uniform(
        k_xy, (2,), minval=-BOX_RANGE, maxval=BOX_RANGE)
    # Resting: one of the 6 faces down, then a random yaw
    yaw = jax.random.uniform(k_yaw, minval=-jp.pi, maxval=jp.pi)
    q_yaw = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    face = jax.random.randint(k_face, (), 0, 6)
    quat = math.quat_mul(q_yaw, FACES[face])
    z = jp.where(face < 2, UPRIGHT_Z, LYING_Z)
    # Dropped: a uniformly random orientation, 5-25 cm above where it would rest upright
    drop = jax.random.bernoulli(k_drop, self._config.drop_prob)
    q_rand = jax.random.normal(k_q, (4,))
    q_rand = q_rand / jp.linalg.norm(q_rand)
    h = jax.random.uniform(k_h, minval=0.05, maxval=0.25)
    quat = jp.where(drop, q_rand, quat)
    z = jp.where(drop, UPRIGHT_Z + h, z)
    return jp.concatenate([xy, z[None]]), quat

  def _sample_target(self, rng):
    lo = jp.array([-TARGET_RANGE, -TARGET_RANGE, TARGET_Z[0]])
    hi = jp.array([TARGET_RANGE, TARGET_RANGE, TARGET_Z[1]])
    return self._init_obj_pos + jax.random.uniform(rng, (3,), minval=lo, maxval=hi)

  def _place_box(self, data, pos, quat):
    a, d = self._obj_qposadr, self._obj_dofadr
    qpos = data.qpos.at[a:a + 3].set(pos).at[a + 3:a + 7].set(quat)
    qvel = data.qvel.at[d:d + 6].set(0.0)
    return data.replace(qpos=qpos, qvel=qvel)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k_box, k_target, k_arm = jax.random.split(rng, 4)
    jitter = jax.random.uniform(k_arm, (7,), minval=-1.0, maxval=1.0) * self._config.arm_jitter
    arm = jp.clip(jp.array(self._init_q[:7]) + jitter, self._arm_lo, self._arm_hi)
    qpos = jp.array(self._init_q).at[:7].set(arm)
    ctrl = jp.array(self._init_ctrl).at[:7].set(arm)
    data = mjx_env.make_data(
        self._mj_model, qpos=qpos, qvel=jp.zeros(self._mjx_model.nv), ctrl=ctrl,
        impl=self._mjx_model.impl.value, naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax, njmax=self._config.njmax)
    pos, quat = self._sample_box(k_box)
    data = self._place_box(data, pos, quat)
    target = self._sample_target(k_target)
    data = data.replace(mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target))
    metrics = {
        "out_of_bounds": jp.array(0.0),
        "at_target": jp.array(0.0),
        "respawns": jp.array(0.0),
        **{k: jp.array(0.0) for k in self._config.reward_config.scales.keys()},
    }
    info = {"rng": rng, "target_pos": target, "reached_box": jp.array(0.0)}
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = dict(state.info)
    rng, k_tj, k_t, k_bj, k_b = jax.random.split(info["rng"], 5)
    jump = jax.random.bernoulli(k_tj, self._config.target_jump_prob)
    target = jp.where(jump, self._sample_target(k_t), info["target_pos"])
    respawn = jax.random.bernoulli(k_bj, self._config.box_respawn_prob)
    pos, quat = self._sample_box(k_b)
    data = state.data
    moved = self._place_box(data, pos, quat)
    data = data.replace(
        qpos=jp.where(respawn, moved.qpos, data.qpos),
        qvel=jp.where(respawn, moved.qvel, data.qvel),
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target))
    info.update(rng=rng, target_pos=target,
                reached_box=jp.where(respawn, 0.0, info["reached_box"]))
    out = super().step(state.replace(data=data, info=info), action)
    box_pos = out.data.xpos[self._obj_body]
    out.metrics.update(
        at_target=(jp.linalg.norm(target - box_pos) < 0.02).astype(float),
        respawns=respawn.astype(float))
    return out

  def _get_reward(self, data, info: Dict[str, Any]) -> Dict[str, Any]:
    rewards = super()._get_reward(data, info)
    # Position only: a randomly turned box shouldn't have to be turned back
    pos_err = jp.linalg.norm(info["target_pos"] - data.xpos[self._obj_body])
    rewards["box_target"] = (1 - jp.tanh(5 * pos_err)) * info["reached_box"]
    rewards["box_fine"] = (1 - jp.tanh(pos_err / 0.01)) * info["reached_box"]
    return rewards
