"""Takeoff and landing for the one-body fly (tools/pack_flyer.py FULL=1): flybody's physics, trained with Brax PPO on
MJX-Warp. flybody has walking and flight policies but nothing between them; these join them. Not the connectome.

Action, at 1 kHz: position targets for head, abdomen and legs (relative to where the fly started), claw adhesion, corrections
to the wings, and wingbeat frequency. The wings follow flybody's recorded wingbeat at 5 kHz, opened from the folded pose.
"""
import json
import os
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

CTRL_DT, SIM_DT, WING_STEPS, WING_SUB = 1e-3, 5e-5, 5, 4      # 20 physics steps per action, wings updated every 4
LEG_SCALE = 0.6
PITCH = np.deg2rad(47.5)


def default_config():
    return config_dict.create(ctrl_dt=CTRL_DT, sim_dt=SIM_DT, episode_length=int(os.environ.get('EP', 300)), action_repeat=1,
                              impl='warp', naconmax=int(os.environ.get('NACON', 60)), naccdmax=int(os.environ.get('NACCD', 0)) or None, njmax=int(os.environ.get('NJMAX', 400)),
                              rise=float(os.environ.get('RISE', 0.25)), goal_z=float(os.environ.get('GOAL_Z', 1.0)), speed=float(os.environ.get('SPEED', 14.0)))


def quat_mul(a, b):
    return jp.array([a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                     a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]])
def rot_inv(q, v):
    u, s = -q[1:], q[0]
    return v + 2 * (s * jp.cross(u, v) + jp.cross(u, jp.cross(u, v)))
smooth = lambda x: (lambda u: u * u * (3 - 2 * u))(jp.clip(x, 0, 1))


class Takeoff(mjx_env.MjxEnv):
    def __init__(self, body=None, data=None, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        here = os.path.dirname(os.path.abspath(__file__))
        body, data = body or f'{here}/fullbody', data or f'{here}/data'
        self._xml = f'{body}/fly.xml'
        m = mujoco.MjModel.from_xml_path(self._xml)
        m.opt.noslip_iterations = 0      # MJX-Warp has none; checked again in the browser's physics afterward
        m.opt.timestep = SIM_DT
        self._m = m
        self._mx = mjx.put_model(m, impl=self._config.impl)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        self._wq = jp.array([m.jnt_qposadr[jid(f'walker/wing_{a}_{s}')] for s in ('left', 'right') for a in ('yaw', 'roll', 'pitch')])
        self._wfold = jp.array(m.qpos_spring)[self._wq]
        self._legq = jp.array([m.jnt_qposadr[j] for j, n in enumerate(names) if any(k in (n or '') for k in ('coxa', 'femur', 'tibia', 'tarsus'))])
        self._tuck = jp.array(m.qpos_spring)[self._legq] * LEG_SCALE
        acts = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        wing = [i for i, n in enumerate(acts) if 'wing_' in n]
        adh = [i for i, n in enumerate(acts) if 'adhere_claw' in n]
        pos = [i for i, n in enumerate(acts) if 'wing_' not in n and 'adhere' not in n and not any(k in n for k in ('rostrum', 'haustellum', 'labrum'))]
        self._wing, self._adh, self._pos = jp.array([acts.index(f'walker/wing_{a}_{s}') for s in ('left', 'right') for a in ('yaw', 'roll', 'pitch')]), jp.array(adh), jp.array(pos)
        trn = m.actuator_trnid[pos, 0]
        self._pos_q = jp.array(m.jnt_qposadr[trn])
        self._lo, self._hi = jp.array(m.actuator_ctrlrange[pos, 0]), jp.array(m.actuator_ctrlrange[pos, 1])
        self._nu = m.nu
        self._action = len(pos) + len(adh) + 6 + 1
        self._pattern = jp.array(np.load(f'{data}/wing_pattern_fmech.npy'))
        self._fq = int(m.jnt_qposadr[jid('walker/')])
        self._fv = int(m.jnt_dofadr[jid('walker/')])
        tid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f'walker/touch_claw_T{l}_{s}') for l in (1, 2, 3) for s in ('left', 'right')]
        self._touch = jp.array([m.sensor_adr[t] for t in tid])
        self._gyro = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'walker/gyro')])
        # starts: the walking policy's own states, from flybody's task (tools/walk_ref.py); wings folded, mouth at rest
        wm = json.load(open(f'{data}/walk_meta.json'))
        Q = np.fromfile(f'{data}/straight_qpos.bin').reshape(600, -1)[50:]
        V = np.fromfile(f'{data}/straight_qvel.bin').reshape(600, -1)[50:]
        q0 = np.tile(m.qpos_spring, (len(Q), 1))
        q0[:, m.jnt_qposadr[jid('walker/')]:7] = 0
        v0 = np.zeros((len(V), m.nv))
        pq = pv = 0
        for n in wm['names_q']:
            free = n.endswith('/')
            k, kv = (7, 6) if free else (1, 1)
            j = jid(n)
            if j >= 0 and 'ghost' not in n:
                q0[:, m.jnt_qposadr[j]:m.jnt_qposadr[j] + k] = Q[:, pq:pq + k]
                v0[:, m.jnt_dofadr[j]:m.jnt_dofadr[j] + kv] = V[:, pv:pv + kv]
            pq += k; pv += kv
        q0[:, self._fq:self._fq + 2] = 0     # each start at the origin, heading kept
        self._Q0, self._V0 = jp.array(q0), jp.array(v0)

    @property
    def xml_path(self): return self._xml
    @property
    def action_size(self): return self._action
    @property
    def mj_model(self): return self._m
    @property
    def mjx_model(self): return self._mx

    def reset(self, rng):
        rng, a, b = jax.random.split(rng, 3)
        i = jax.random.randint(a, (), 0, self._Q0.shape[0])
        qpos, qvel = self._Q0[i], self._V0[i]
        q = qpos[self._fq + 3:self._fq + 7]
        h0 = 2 * jp.arctan2(q[3], q[0])
        data = mjx_env.make_data(self._m, qpos=qpos, qvel=qvel, impl=self._config.impl, naconmax=self._config.naconmax, naccdmax=self._config.naccdmax, njmax=self._config.njmax)
        data = mjx.forward(self._mx, data)
        ph = jax.random.uniform(b)
        info = {'rng': rng, 't': jp.array(0.0), 'phase': ph, 'freq': jp.array(218.0), 'hold': qpos[self._pos_q], 'h0': h0, 'p0': qpos[self._fq:self._fq + 3], 'last': jp.zeros(self._action)}
        metrics = {k: jp.array(0.0) for k in ('r_pos', 'r_ori', 'r_legs', 'r_wing', 'height', 'flying')}
        return mjx_env.State(data, self._obs(data, info), jp.array(0.0), jp.array(0.0), metrics, info)

    def _pattern_at(self, ph):
        n = self._pattern.shape[0]
        f = (ph - jp.floor(ph)) * n
        i = jp.floor(f).astype(int) % n
        a = f - jp.floor(f)
        v = self._pattern[i] * (1 - a) + self._pattern[(i + 1) % n] * a
        return jp.concatenate([v, v])

    def _open(self, t, info):
        return smooth(t / 0.03)       # the wings open in 30 ms, as a fly raises them before it jumps

    def _ref(self, t, info):
        c = self._config
        s = smooth(t / c.rise)
        dirx = jp.array([jp.cos(info['h0']), jp.sin(info['h0']), 0.0])
        # distance flown at the speed the flight policy will take over at, ramping in
        dist = c.speed * jp.maximum(t - c.rise * 0.5, 0.0) * smooth(t / c.rise)
        p = jp.array([info['p0'][0], info['p0'][1], info['p0'][2] + (c.goal_z - info['p0'][2]) * s]) + dist * dirx
        pitch = -PITCH * smooth(t / (c.rise * 0.6))
        y = jp.array([jp.cos(info['h0'] / 2), 0.0, 0.0, jp.sin(info['h0'] / 2)])
        q = quat_mul(y, jp.array([jp.cos(pitch / 2), 0.0, jp.sin(pitch / 2), 0.0]))
        return p, q

    def _obs(self, d, info):
        t = info['t']
        p, q = d.qpos[self._fq:self._fq + 3], d.qpos[self._fq + 3:self._fq + 7]
        rp, rq = self._ref(t + 0.01, info)
        conj = q * jp.array([1., -1., -1., -1.])
        return jp.concatenate([jp.array([p[2], t * 10, jp.sin(2 * jp.pi * info['phase']), jp.cos(2 * jp.pi * info['phase'])]),
                               rot_inv(q, jp.array([0., 0., 1.])), rot_inv(q, rp - p), quat_mul(conj, rq),
                               d.qpos[self._fq + 7:], jp.clip(d.qvel * 0.02, -10, 10), jp.clip(d.sensordata[self._touch] * 1e-2, 0, 10), info['last']])

    def step(self, state, action):
        c = self._config
        info = dict(state.info)
        a = jp.clip(action, -1, 1)
        np_, na = self._pos.shape[0], self._adh.shape[0]
        ctrl = state.data.ctrl
        target = jp.clip(info['hold'] + 1.2 * a[:np_], self._lo, self._hi)
        ctrl = ctrl.at[self._pos].set(target).at[self._adh].set((a[np_:np_ + na] + 1) / 2)
        wfix = a[np_ + na:np_ + na + 6] * 0.5
        rate = float(np.exp(-2e-4 / (0.5 / 218)))
        want = 218.0 * (1 + 0.05 * a[-1])
        d, ph, fr, t = state.data, info['phase'], info['freq'], info['t']
        for k in range(WING_STEPS):
            fr = fr * rate + want * (1 - rate)
            ph = ph + fr * 2e-4
            tt = t + (k + 1) * 2e-4
            openw = self._open(tt, info)
            tgt = self._wfold * (1 - openw) + self._pattern_at(ph) * openw
            ctrl = ctrl.at[self._wing].set(jp.clip(wfix * openw + tgt - d.qpos[self._wq], -3, 3))
            d = mjx_env.step(self._mx, d, ctrl, WING_SUB)
        info.update(t=t + CTRL_DT, phase=ph, freq=fr, last=a)
        t = info['t']
        p, q = d.qpos[self._fq:self._fq + 3], d.qpos[self._fq + 3:self._fq + 7]
        rp, rq = self._ref(t, info)
        e = jp.sum((p - rp) ** 2)
        dq = quat_mul(rq * jp.array([1., -1., -1., -1.]), q)
        ang = 2 * jp.arctan2(jp.linalg.norm(dq[1:]), jp.abs(dq[0]))
        lg = smooth((t - 0.05) / 0.1)
        legs = jp.mean((d.qpos[self._legq] - self._tuck) ** 2)
        r_pos, r_ori = jp.exp(-e / 0.15 ** 2), jp.exp(-ang ** 2 / 0.5 ** 2)
        r_legs = lg * jp.exp(-legs / 0.3 ** 2)
        rate_cost = jp.mean((a - state.info['last']) ** 2)
        gy = jp.sum(d.sensordata[self._gyro:self._gyro + 3] ** 2)
        r_wing = jp.exp(-gy / 60.0 ** 2)
        reward = 0.5 * r_pos + 0.3 * r_ori + 0.1 * r_legs + 0.1 * r_wing - 0.02 * rate_cost
        up = rot_inv(q, jp.array([0., 0., 1.]))[2]
        bad = (t > 0.1) & (up < -0.1) | (p[2] > c.goal_z + 1.5) | (jp.linalg.norm(p - rp) > 1.5) | ((t > c.rise) & (p[2] < 0.3))
        done = jp.where(bad | ~jp.isfinite(p).all() | ~jp.isfinite(d.qvel).all(), 1.0, 0.0)
        reward = jp.where(jp.isfinite(reward), reward, -1.0)
        obs = jp.nan_to_num(self._obs(d, info), posinf=10, neginf=-10)
        flying = jp.where(jp.linalg.norm(p - rp) < 0.3, 1.0, 0.0) * (t > c.rise)
        m = dict(state.metrics); m.update(r_pos=r_pos, r_ori=r_ori, r_legs=r_legs, r_wing=r_wing, height=p[2], flying=flying)
        return state.replace(data=d, obs=obs, reward=reward, done=done, metrics=m, info=info)


class Landing(Takeoff):
    """From flight onto the floor, ending in a stand the walking policy can take over: descend, level the body, legs out, wings shut."""
    def __init__(self, body=None, data=None, config=None, config_overrides=None):
        super().__init__(body, data, config, config_overrides)
        i = np.array(self._Q0)[:, self._legq].mean(0)
        self._stand = jp.array(i)
        self._stand_z = float(np.array(self._Q0)[:, self._fq + 2].mean())
        self._folded = float(os.environ.get('FOLD_AT', 0.22))

    def reset(self, rng):
        c = self._config
        rng, a, b, e, f = jax.random.split(rng, 5)
        z = jax.random.uniform(a, minval=0.3, maxval=0.7)
        speed = jax.random.uniform(e, minval=4.0, maxval=c.speed)
        h = jax.random.uniform(f, minval=-3.1, maxval=3.1)
        ph = jax.random.uniform(b)
        y = jp.array([jp.cos(h / 2), 0.0, 0.0, jp.sin(h / 2)])
        q = quat_mul(y, jp.array([jp.cos(PITCH / 2), 0.0, -jp.sin(PITCH / 2), 0.0]))
        qpos = jp.array(self._m.qpos_spring).at[self._legq].set(self._tuck)
        qpos = qpos.at[self._fq:self._fq + 3].set(jp.array([0., 0., z])).at[self._fq + 3:self._fq + 7].set(q)
        tgt, tgt2 = self._pattern_at(ph), self._pattern_at(ph + 218 * 2e-4)
        qpos = qpos.at[self._wq].set(tgt)
        qvel = jp.zeros(self._m.nv).at[self._fv].set(speed * jp.cos(h)).at[self._fv + 1].set(speed * jp.sin(h))
        wv = jp.array([self._m.jnt_dofadr[mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_JOINT, f'walker/wing_{x}_{s}')] for s in ('left', 'right') for x in ('yaw', 'roll', 'pitch')])
        qvel = qvel.at[wv].set((tgt2 - tgt) / 2e-4)
        data = mjx_env.make_data(self._m, qpos=qpos, qvel=qvel, impl=c.impl, naconmax=c.naconmax, naccdmax=c.naccdmax, njmax=c.njmax)
        data = mjx.forward(self._mx, data)
        info = {'rng': rng, 't': jp.array(0.0), 'phase': ph, 'freq': jp.array(218.0), 'hold': qpos[self._pos_q], 'h0': h, 'p0': jp.array([0., 0., z]), 'last': jp.zeros(self._action), 'speed0': speed}
        metrics = {k: jp.array(0.0) for k in ('r_pos', 'r_ori', 'r_legs', 'r_wing', 'height', 'flying')}
        return mjx_env.State(data, self._obs(data, info), jp.array(0.0), jp.array(0.0), metrics, info)

    def _open(self, t, info):
        return 1 - smooth((t - self._folded) / 0.03)    # wings fold once it should be down

    def _ref(self, t, info):
        T = self._config.rise
        s = smooth(t / T)
        dirx = jp.array([jp.cos(info['h0']), jp.sin(info['h0']), 0.0])
        # keep going, slowing down, and come down to standing height
        u = jp.clip(t / T, 0, 1)
        dist = info.get('speed0', 10.0) * T * (u - u * u / 2)
        p = jp.array([0., 0., info['p0'][2] + (self._stand_z - info['p0'][2]) * s]) + dist * dirx
        pitch = -PITCH * (1 - smooth(t / T))
        y = jp.array([jp.cos(info['h0'] / 2), 0.0, 0.0, jp.sin(info['h0'] / 2)])
        return p, quat_mul(y, jp.array([jp.cos(pitch / 2), 0.0, jp.sin(pitch / 2), 0.0]))

    def step(self, state, action):
        out = super().step(state, action)
        d, t = out.data, out.info['t']
        c = self._config
        p, q = d.qpos[self._fq:self._fq + 3], d.qpos[self._fq + 3:self._fq + 7]
        rp, rq = self._ref(t, out.info)
        e = jp.sum((p - rp) ** 2)
        dq = quat_mul(rq * jp.array([1., -1., -1., -1.]), q)
        ang = 2 * jp.arctan2(jp.linalg.norm(dq[1:]), jp.abs(dq[0]))
        lg = smooth(t / (c.rise * 0.7))
        legs = jp.mean((d.qpos[self._legq] - self._stand) ** 2)
        late = smooth((t - c.rise) / 0.03)
        slow = jp.exp(-jp.sum(d.qvel[self._fv:self._fv + 3] ** 2) / 4.0 ** 2)
        feet = jp.mean(jp.where(d.sensordata[self._touch] > 1e-6, 1.0, 0.0))
        r_pos, r_ori, r_legs = jp.exp(-e / 0.1 ** 2), jp.exp(-ang ** 2 / 0.4 ** 2), lg * jp.exp(-legs / 0.3 ** 2)
        reward = 0.3 * r_pos + 0.2 * r_ori + 0.15 * r_legs + late * (0.15 * slow + 0.2 * feet) - 0.02 * jp.mean((jp.clip(action, -1, 1) - state.info['last']) ** 2)
        up = rot_inv(q, jp.array([0., 0., 1.]))[2]
        bad = (up < 0.0) | (jp.linalg.norm(p - rp) > 1.0)
        done = jp.where(bad | ~jp.isfinite(p).all() | ~jp.isfinite(d.qvel).all(), 1.0, 0.0)
        landed = (t > c.rise + 0.03) * jp.where((feet > 0.6) & (up > 0.7) & (p[2] < 0.2) & (jp.linalg.norm(d.qvel[self._fv:self._fv + 3]) < 3.0), 1.0, 0.0)
        m = dict(out.metrics); m.update(r_pos=r_pos, r_ori=r_ori, r_legs=r_legs, flying=landed, height=p[2])
        return out.replace(reward=jp.where(jp.isfinite(reward), reward, -1.0), done=done, metrics=m)
