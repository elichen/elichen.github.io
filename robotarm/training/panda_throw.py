"""Throw the box as far as possible in a given direction.

Episodes start from states where the pick policy has just lifted the box
(collect_grasps.py saves thousands of them), so on the page the pick policy hands
the box over to this one. The direction is an input: an angle around the robot's
base, sampled uniformly.

Reward: 10 x the box's progress along the direction each step, so an episode's
return is 10 x the distance gained, minus (with LATERAL=<scale>) the growth of its
sideways distance from the line, so throws land along the arrow on the page. The
episode ends when the box first touches the floor after leaving the gripper;
sliding and tumbling after that don't count.
Penalties keep the hand off the floor and away from the robot's own base (this
model has no self-collisions, so nothing else stops the arm passing through it).

Observation (59): qpos, qvel, gripper position and orientation, box orientation,
box - gripper, position target - joint position, and (cos, sin) of the direction.
Physics and actions are those of PandaPickCube.
"""

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import pick

LANDED_Z = 0.045      # box center below this, after release = touching the floor
RELEASED = 0.08       # box this far from the gripper = let go
BASE_CLEARANCE = 0.18 # hand must stay this far (horizontally) from the base's axis


def throw_config() -> config_dict.ConfigDict:
  c = pick.default_config()
  c.episode_length = 150
  c.reward_config = config_dict.create(scales=config_dict.create(
      progress=10.0, lateral=0.0, floor=1.0, base=1.0))
  c.grasps = "grasps.npz"
  return c


class PandaThrow(pick.PandaPickCube):

  def __init__(self, config=None, config_overrides=None, grasps=None):
    super().__init__(config or throw_config(), config_overrides)
    g = np.load(grasps or self._config.grasps)
    self._g_qpos, self._g_qvel, self._g_ctrl = (jp.array(g[k]) for k in ("qpos", "qvel", "ctrl"))
    print(f"PandaThrow: {len(g['qpos'])} start states")

  @property
  def observation_size(self):
    return 59

  def _distance(self, data, heading):
    u = jp.array([jp.cos(heading), jp.sin(heading)])
    return jp.dot(data.xpos[self._obj_body][:2], u)

  def _sideways(self, data, heading):
    v = jp.array([-jp.sin(heading), jp.cos(heading)])
    return jp.abs(jp.dot(data.xpos[self._obj_body][:2], v))

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k_i, k_h = jax.random.split(rng, 3)
    i = jax.random.randint(k_i, (), 0, self._g_qpos.shape[0])
    data = mjx_env.make_data(
        self._mj_model, qpos=self._g_qpos[i], qvel=self._g_qvel[i], ctrl=self._g_ctrl[i],
        impl=self._mjx_model.impl.value, naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax, njmax=self._config.njmax)
    data = mjx.forward(self._mjx_model, data)  # positions for the first observation
    heading = jax.random.uniform(k_h, minval=-jp.pi, maxval=jp.pi)
    info = {"rng": rng, "heading": heading, "dist": self._distance(data, heading),
            "side": self._sideways(data, heading), "released": jp.array(0.0)}
    metrics = {"throw": jp.array(0.0), "landed": jp.array(0.0),
               **{k: jp.array(0.0) for k in self._config.reward_config.scales.keys()}}
    return mjx_env.State(data, self._get_obs(data, info), jp.array(0.0), jp.array(0.0), metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = dict(state.info)
    ctrl = jp.clip(state.data.ctrl + action * self._action_scale, self._lowers, self._uppers)
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    box = data.xpos[self._obj_body]
    gripper = data.site_xpos[self._gripper_site]
    released = jp.maximum(info["released"], (jp.linalg.norm(box - gripper) > RELEASED).astype(float))
    landed = (released > 0) & (box[2] < LANDED_Z)
    dist = self._distance(data, info["heading"])
    side = self._sideways(data, info["heading"])

    floor = sum(data.sensordata[self._mj_model.sensor_adr[s]] > 0 for s in self._floor_hand_found_sensor) > 0
    near_base = jp.linalg.norm(gripper[:2]) < BASE_CLEARANCE
    raw = {"progress": dist - info["dist"], "lateral": info["side"] - side,
           "floor": -floor.astype(float), "base": -near_base.astype(float)}
    scales = self._config.reward_config.scales
    reward = sum(v * scales[k] for k, v in raw.items())

    bad = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | (box[2] < -0.1)
    done = (landed | bad).astype(float)
    info.update(dist=dist, side=side, released=released)
    state.metrics.update(**raw, landed=landed.astype(float), throw=jp.where(landed, dist, 0.0))
    return mjx_env.State(data, self._get_obs(data, info), reward, done, state.metrics, info)

  def _get_obs(self, data, info):
    gripper = data.site_xpos[self._gripper_site]
    return jp.concatenate([
        data.qpos,
        data.qvel,
        gripper,
        data.site_xmat[self._gripper_site].ravel()[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - gripper,
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
        jp.array([jp.cos(info["heading"]), jp.sin(info["heading"])]),
    ])
