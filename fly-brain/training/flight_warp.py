"""Does flybody's trained flight policy fly on GPU physics (MJX-Warp)? If not, nothing trained there will fly in the browser.
Same observation and action as ../flight/flyer.js, batched. Run on Nitro:
    python flight_warp.py <flight/body dir> [flies=256] [seconds=0.5]
"""
import json
import os
import sys
import time
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_MODULE_LOADING'] = 'EAGER'
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

body, N, T = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 256, float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
meta = json.load(open(f'{body}/meta.json'))
w = np.fromfile(f'{body}/policy.bin', np.float32)
at = 0
def take(*shape):
    global at
    n = int(np.prod(shape)); x = jp.array(w[at:at + n].reshape(shape)); at += n; return x
W0, B0, OFF, SCALE = take(104, 256), take(256), take(256), take(256)
LAYERS = [(take(256, 256), take(256)) for _ in range(2)]
WM, BM = take(256, 12), take(12)
def policy(x):
    h = x @ W0 + B0
    h = jp.tanh((h - h.mean()) / jp.sqrt(h.var() + 1e-5) * SCALE + OFF)
    for a, b in LAYERS:
        h = jax.nn.elu(h @ a + b)
    return jp.clip(h @ WM + BM, -1, 1)

m = mujoco.MjModel.from_xml_path(f'{body}/fly.xml')
m.opt.noslip_iterations = 0
id_ = lambda t, n: mujoco.mj_name2id(m, getattr(mujoco.mjtObj, t), n)
jq = jp.array([m.jnt_qposadr[id_('mjOBJ_JOINT', n)] for n in meta['joints']])
jv = jp.array([m.jnt_dofadr[id_('mjOBJ_JOINT', n)] for n in meta['joints']])
wq = jp.array([m.jnt_qposadr[id_('mjOBJ_JOINT', n)] for n in meta['wing_joints']])
wv = jp.array([m.jnt_dofadr[id_('mjOBJ_JOINT', n)] for n in meta['wing_joints']])
fq, fv = int(m.jnt_qposadr[id_('mjOBJ_JOINT', 'walker/')]), int(m.jnt_dofadr[id_('mjOBJ_JOINT', 'walker/')])
sens = [int(m.sensor_adr[id_('mjOBJ_SENSOR', meta[k])]) for k in ('accelerometer', 'gyro', 'velocimeter')]
ctrl = jp.array([id_('mjOBJ_ACTUATOR', 'walker/' + n) for n in meta['action'] if not n.startswith('user')])
LO, HI = jp.array(meta['lo']), jp.array(meta['hi'])
PAT = jp.array(meta['pattern'])
WA = jp.array(meta['wing_action'])
UA = meta['user_action']
DT = 2e-4
COM = jp.array([-0.03697732, 0.00029205, -0.0142447])
PITCH = -47.5 * np.pi / 180
RPY = (27 * np.pi / 180) / 43

def mulq(a, b):
    return jp.array([a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                     a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]])
def axis(th, k):
    return jp.array([jp.cos(th/2), *[jp.sin(th/2) if i == k else 0.0 * th for i in range(3)]])
def rot(q, v):
    u, s = q[1:], q[0]
    return v + 2 * (s * jp.cross(u, v) + jp.cross(u, jp.cross(u, v)))
def pose(x, y, th, yaw, z=1.0):
    q = mulq(axis(th, 2), mulq(axis(-RPY * yaw, 0), axis(jp.asarray(PITCH), 1)))
    return jp.array([x, y, z]) - rot(q, COM), q
def wing(ph):
    n = PAT.shape[0]
    f = (ph - jp.floor(ph)) * n
    i = jp.floor(f).astype(int) % n
    a = f - jp.floor(f)
    v = PAT[i] * (1 - a) + PAT[(i + 1) % n] * a
    return jp.concatenate([v, v])
def obs(d, buf, ref, speed, yaw):
    p, q0 = d.qpos[fq:fq+3], d.qpos[fq+3:fq+7]
    conj = q0 * jp.array([1., -1., -1., -1.])
    disp, quats = [], []
    x, y, th = ref
    for _ in range(6):
        pp, qq = pose(x, y, th, yaw)
        disp.append(rot(conj, pp - p)); quats.append(mulq(conj, qq))
        x, y, th = x + speed*DT*jp.cos(th), y + speed*DT*jp.sin(th), th + yaw*DT
    wz = rot(conj, jp.array([0., 0., 1.]))
    return jp.concatenate([buf[0:3], buf[3:6], d.qpos[jq], d.qvel[jv], *disp, *quats, buf[6:9], wz])

mx = mjx.put_model(m, impl='warp')
d0 = mjx.make_data(m, impl='warp', naconmax=N, njmax=64)
RATE = float(np.exp(-DT / meta['ctrl_filter']))

@jax.jit
def control(state, yaw):
    d, buf, ref, ph, fr = state
    speed = 14.0
    a = jax.vmap(lambda s, b, r: policy(obs(s, b, r, speed, yaw)))(d, buf, ref)
    real = LO + (a + 1) / 2 * (HI - LO)
    fr = fr * RATE + meta['base_freq'] * (1 + meta['rel_range'] * real[:, UA]) * (1 - RATE)
    ph = ph + fr * DT
    tgt = jax.vmap(wing)(ph)
    real = real.at[:, WA].add(tgt - d.qpos[:, wq])
    d = d.replace(ctrl=d.ctrl.at[:, ctrl].set(real[:, :11]))
    s = jp.zeros_like(buf)
    for _ in range(4):
        d = jax.vmap(lambda x: mjx.step(mx, x))(d)
        s = s + jp.concatenate([d.sensordata[:, sens[0]:sens[0]+3], d.sensordata[:, sens[1]:sens[1]+3], d.sensordata[:, sens[2]:sens[2]+3]], 1)
    x, y, th = ref[:, 0], ref[:, 1], ref[:, 2]
    ref = jp.stack([x + speed*DT*jp.cos(th), y + speed*DT*jp.sin(th), th + yaw*DT], 1)
    return d, s / 4, ref, ph, fr

ph0 = jp.array(np.random.default_rng(0).random(N))
q0 = jp.array(meta['qpos0'])[:m.nq]   # the walker comes first
nm = meta['names_q']
qpos = np.array(m.qpos0)
pq = 0
for n in nm:
    free = n.endswith('/')
    k = 7 if free else 1
    j = id_('mjOBJ_JOINT', n)
    if j >= 0:
        qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j]+k] = meta['qpos0'][pq:pq+k]
    pq += k
p, q = pose(0., 0., 0., 0.)
qpos[fq:fq+3], qpos[fq+3:fq+7] = np.array(p), np.array(q)
def init(ph):
    a, b = wing(ph), wing(ph + meta['base_freq'] * DT)
    qp = jp.array(qpos).at[wq].set(a)
    qv = jp.zeros(m.nv).at[wv].set((b - a) / DT).at[fv].set(14.0)
    return qp, qv
qp, qv = jax.vmap(init)(ph0)
d = jax.vmap(lambda p_, v_: d0.replace(qpos=p_, qvel=v_))(qp, qv)
d = jax.vmap(lambda x: mjx.forward(mx, x))(d)
state = (d, jp.zeros((N, 9)), jp.zeros((N, 3)), ph0, jp.full(N, meta['base_freq']))
t0 = time.time()
sd = 0.04 / 2.355
for k in range(int(T / DT)):
    t = k * DT
    yaw = 43 * float(np.exp(-((t - T * 0.5) ** 2) / (2 * sd * sd)))
    state = control(state, yaw)
    if k == 0:
        print('first step (compile)', round(time.time() - t0, 1), flush=True); t0 = time.time()
    if (k + 1) % 500 == 0:
        d, _, ref, _, _ = state
        px = np.asarray(d.qpos[:, fq:fq+3]); r = np.asarray(ref)
        e = np.hypot(px[:, 0] - r[:, 0], px[:, 1] - r[:, 1])
        fin = np.isfinite(px).all(1)
        print(f't {t+DT:.2f} s: flying {int((px[:, 2] > 0.8).sum())}/{N}, finite {int(fin.sum())}, z {np.nanmin(px[:, 2]):.2f}-{np.nanmax(px[:, 2]):.2f}, xy error to path median {np.nanmedian(e):.3f} max {np.nanmax(e):.3f} cm; {N*(k+1)*4/(time.time()-t0):.0f} physics steps/s', flush=True)
