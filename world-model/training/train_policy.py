"""Train the 100% LANGUAGE-FREE latent-conditioned UR5e behavior policy pi(a_t | obs_t, z_beh),
then run a FULL CLOSED-LOOP MUJOCO EVALUATION driven strictly by:

    Unseen English Prompt -> Frozen MiniLM -> Single Orthogonal Rotation R -> pi(a_t | obs_t, z_beh) -> UR5e!

Zero text ever touches E_traj or pi during training!
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sim_skills import UR5eArena, SKILL_NAMES, UNSEEN_PROMPTS

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True


class FiLMResBlock(nn.Module):
    def __init__(self, d, dz):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.film = nn.Linear(dz, d * 2)
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x, z):
        h = self.ln(x)
        gb = self.film(z)
        g, b = gb.chunk(2, dim=-1)
        h = h * (1.0 + g) + b
        u, v = self.fc1(h).chunk(2, dim=-1)
        return x + self.fc2(F.silu(u) * v)


class BehaviorPolicy(nn.Module):
    """Compact (~2.0M params, ~2.1 MB int8 in browser!) recurrent latent-conditioned UR5e policy.
    Takes:
      obs_t  (30D: proprio + visually-read cube_pos)
      z_beh  (dz-D unit-normalized behavior embedding from E_traj OR rotated MiniLM!)
      h_prev (dh-D GRU state, zero-init on new command)
    Outputs:
      a_t    (7D in [-1, 1]: 6 joint deltas + gripper)
      h_next (dh-D)
    """
    def __init__(self, obs_dim=30, dz=64, d=384, nblocks=3, act_dim=7):
        super().__init__()
        self.obs_dim = obs_dim
        self.dz = dz
        self.d = d
        self.nblocks = nblocks
        self.act_dim = act_dim
        self.in_proj = nn.Linear(obs_dim + dz + 1, d)  # +1 for elapsed time frac t/80
        self.gru = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.blocks = nn.ModuleList([FiLMResBlock(d, dz) for _ in range(nblocks)])
        self.out_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, act_dim)

    def forward_seq(self, obs_seq, z_beh):
        """obs_seq: (B, T, 30), z_beh: (B, dz) -> (B, T, 7)."""
        B, T, _ = obs_seq.shape
        t_feat = torch.linspace(0.0, 1.0, T, device=obs_seq.device).view(1, T, 1).expand(B, T, 1)
        z_exp = z_beh.unsqueeze(1).expand(B, T, -1)
        h = F.silu(self.in_proj(torch.cat([obs_seq, z_exp, t_feat], dim=-1)))
        h, _ = self.gru(h)
        for blk in self.blocks:
            h = blk(h, z_exp)
        return torch.tanh(self.head(self.out_ln(h)))

    @torch.inference_mode()
    def step(self, obs_t, z_beh, h_prev, t_frac):
        """Single 20 Hz step for closed-loop execution (matches browser JS 1:1!)."""
        tf = torch.tensor([[t_frac]], dtype=obs_t.dtype, device=obs_t.device)
        x = F.silu(self.in_proj(torch.cat([obs_t, z_beh, tf], dim=-1))).unsqueeze(1)
        h_out, h_next = self.gru(x, h_prev)
        h = h_out.squeeze(1)
        for blk in self.blocks:
            h = blk(h, z_beh)
        return torch.tanh(self.head(self.out_ln(h))), h_next


def train_policy(obs_tr, act_tr, G_tr, steps=16000, bs=256, lr=1e-3, dz=112, d=512):
    """Train pi(a | obs, z_beh) strictly on unlabeled trajectories (ZERO text!)."""
    N, T, obs_dim = obs_tr.shape
    obs_mu = obs_tr.mean((0, 1), keepdims=True).astype(np.float32)
    obs_sd = np.maximum(obs_tr.std((0, 1), keepdims=True), 1e-3).astype(np.float32)
    O = torch.from_numpy(((obs_tr - obs_mu) / obs_sd).astype(np.float32)).to(DEV)
    # Precompute raw distance ||cube - eff|| (cols 26..29 in raw obs) so precision grasping steps get 4x loss weight!
    dist_ce = np.linalg.norm(obs_tr[:, :, 26:29], axis=-1).astype(np.float32)
    W_step = torch.from_numpy(np.where(dist_ce < 0.13, 4.0, 1.0).astype(np.float32)).to(DEV).unsqueeze(-1)
    # Realistic Readout2 visual cube noise in RAW METERS (0.5 cm), converted to standardized obs units
    sd_c = torch.from_numpy(obs_sd[0, 0, 23:26]).to(DEV).view(1, 1, 3)
    sd_rel = torch.from_numpy(obs_sd[0, 0, 26:29]).to(DEV).view(1, 1, 3)

    A = torch.from_numpy(act_tr.astype(np.float32)).to(DEV)
    Gu = (G_tr / np.maximum(np.linalg.norm(G_tr, axis=1, keepdims=True), 1e-8)).astype(np.float32)
    Z = torch.from_numpy(Gu).to(DEV)

    pol = BehaviorPolicy(obs_dim=obs_dim, dz=dz, d=d, nblocks=3, act_dim=7).to(DEV)
    ema = BehaviorPolicy(obs_dim=obs_dim, dz=dz, d=d, nblocks=3, act_dim=7).to(DEV)
    ema.load_state_dict(pol.state_dict())
    n_params = sum(p.numel() for p in pol.parameters())
    print(f"Training BehaviorPolicy ({n_params/1e6:.2f}M params, d={d}, dz={dz}) on {N} clips (ZERO text!)...", flush=True)

    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    act_w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], device=DEV).view(1, 1, 7)

    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (bs,), device=DEV)
        ob = O[idx].clone()
        ac = A[idx]
        zb = Z[idx]
        ws = W_step[idx]

        # 1) Realistic 5 mm Readout2 visual jitter on cube coords (70% of batch; 30% clean for sharp grasping!)
        mask_n = (torch.rand(bs, 1, 1, device=DEV) < 0.70).float()
        ob[:, :, :23] += torch.randn_like(ob[:, :, :23]) * 0.008 * mask_n
        raw_err = (torch.randn(bs, 1, 3, device=DEV) * 0.004 + torch.randn(bs, T, 3, device=DEV) * 0.003) * mask_n
        ob[:, :, 23:26] += raw_err / sd_c
        ob[:, :, 26:29] += raw_err / sd_rel

        # 2) Properly dimension-scaled isotropic latent noise on unit-sphere z_beh:
        #    ||noise||_2 ~ Uniform(0, 0.25) (dividing by sqrt(dz)!) so cos(zb_n, zb) stays in [0.97, 1.0]!
        sig = (torch.rand(bs, 1, device=DEV) * 0.25) / math.sqrt(dz)
        zb_n = F.normalize(zb + torch.randn_like(zb) * sig, dim=-1)

        with torch.autocast('cuda', torch.bfloat16):
            pred = pol.forward_seq(ob, zb_n)
            per_el = F.smooth_l1_loss(pred, ac, beta=0.04, reduction='none') + 0.5 * (pred - ac).pow(2)
            loss = (per_el * ws * act_w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 2.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            for pe, p in zip(ema.parameters(), pol.parameters()):
                pe.mul_(0.996).add_(p, alpha=0.004)

        if step % 2000 == 0 or step == steps:
            print(f"  [Policy] step {step:5d}/{steps} | loss={loss.item():.5f} ({time.time()-t0:.1f}s)", flush=True)

    ema.eval()
    return ema, obs_mu, obs_sd


@torch.inference_mode()
def eval_closed_loop_english(robot_dir, pol, obs_mu, obs_sd, st_model, mu_t, P_t, R, tag='N=10000', T=80):
    """Run REAL closed-loop MuJoCo physics driven ONLY by unseen English -> MiniLM -> R -> Policy!"""
    env = UR5eArena(robot_dir)
    obs_mu_t = torch.from_numpy(obs_mu[0, 0]).to(DEV)
    obs_sd_t = torch.from_numpy(obs_sd[0, 0]).to(DEV)
    rng = np.random.default_rng(123)

    test_cases = [
        ("throw cube", "throw"),
        ("yeet the red block", "throw"),
        ("toss it to the right", "throw"),
        ("rotate around your base", "rotate_base"),
        ("do a 360", "rotate_base"),
        ("put the cube in the front left corner", "place", (0.50, -0.18)),
        ("move the red block to the back right", "place", (0.34, 0.18)),
        ("set it down in the middle of the table", "place", (0.42, 0.0)),
        ("shove the cube to the left", "push", "left"),
        ("knock the red block to the right", "push", "right"),
        ("pick up the red block and keep it aloft", "lift_hold"),
        ("grab the block and let it fall", "lift_drop"),
        ("say hi", "wave"),
        ("touch the ceiling", "reach_up"),
        ("take a bow", "bow"),
        ("stir the pot over the table", "circle"),
        ("sweep side to side", "sweep"),
        ("wiggle your claws open and closed", "clap_gripper"),
        ("boop the red square", "poke_cube"),
        ("point over to the left side", "point", "left"),
        ("nod your head yes", "nod"),
        ("shake and party", "dance"),
        ("freeze at home position", "home"),
    ]

    print(f"\n=== CLOSED-LOOP MUJOCO EVAL ({tag}): UNSEEN ENGLISH -> MiniLM -> R -> UR5e ===", flush=True)
    embs = st_model.encode([c[0] for c in test_cases], normalize_embeddings=False).astype(np.float64)
    Q = (embs - mu_t) @ P_t @ R
    Qu = unit(Q).astype(np.float32)

    passed = 0
    for idx, case in enumerate(test_cases):
        prompt, sk = case[0], case[1]
        z_t = torch.from_numpy(Qu[idx:idx+1]).to(DEV)
        # Run 3 trials per prompt with random start poses!
        ok_trials = 0
        metrics_str = ''
        for tr in range(3):
            ob = env.reset(rng)
            c0 = env.cube_pos().copy()
            q0_start = float(env.d.qpos[0])
            h = torch.zeros(1, 1, pol.d, device=DEV)
            max_cz = 0.0; max_cv = 0.0; max_ez = 0.0; min_ez = 9.0
            grip_hist = []; pan_hist = []; eff_xy = []; w1_hist = []
            min_cube_eff = 9.0
            for t in range(T):
                ob_t = (torch.from_numpy(ob).to(DEV) - obs_mu_t) / obs_sd_t
                a_t, h = pol.step(ob_t.unsqueeze(0), z_t, h, t / float(T - 1))
                ob = env.step(a_t[0].cpu().numpy())
                c = env.cube_pos(); cv = env.cube_vel(); e = env.eff_pos()
                max_cz = max(max_cz, float(c[2]))
                max_cv = max(max_cv, float(np.linalg.norm(cv)))
                max_ez = max(max_ez, float(e[2]))
                min_ez = min(min_ez, float(e[2]))
                min_cube_eff = min(min_cube_eff, float(np.linalg.norm(c - e)))
                grip_hist.append(env.grip_closed())
                pan_hist.append(float(env.d.qpos[0]))
                eff_xy.append(e[:2].copy())
                w1_hist.append(float(env.d.qpos[3]))

            c_end = env.cube_pos()
            fly = float(np.linalg.norm(c_end[:2] - c0[:2]))
            dpan = max(pan_hist) - min(pan_hist)
            g_rng = max(grip_hist) - min(grip_hist)
            exy = np.array(eff_xy)
            y_span = float(exy[:, 1].max() - exy[:, 1].min())
            x_span = float(exy[:, 0].max() - exy[:, 0].min())

            ok = False
            if sk == 'throw':
                ok = (fly > 0.30 or max_cv > 1.0 or max_cz > 0.35) and (env.has_cube() < 0.5)
                metrics_str = f"fly={fly:.2f}m max_v={max_cv:.1f}m/s max_z={max_cz:.2f}m"
            elif sk == 'rotate_base':
                ok = dpan > 3.2
                metrics_str = f"spin={math.degrees(dpan):.0f} deg"
            elif sk == 'place':
                gx, gy = case[2]
                err = math.hypot(c_end[0] - gx, c_end[1] - gy) * 100
                ok = err < 10.0 and c_end[2] < 0.05
                metrics_str = f"end=({c_end[0]*100:.0f},{c_end[1]*100:+.0f})cm err={err:.1f}cm"
            elif sk == 'push':
                dy = c_end[1] - c0[1]
                ok = (dy < -0.07) if case[2] == 'left' else (dy > 0.07)
                metrics_str = f"dy={dy*100:+.1f}cm"
            elif sk == 'lift_hold':
                ok = c_end[2] > 0.11 and env.has_cube() > 0.5
                metrics_str = f"end_z={c_end[2]*100:.0f}cm in_hand={int(env.has_cube())}"
            elif sk == 'lift_drop':
                ok = max_cz > 0.15 and c_end[2] < 0.045
                metrics_str = f"peak_z={max_cz*100:.0f}cm end_z={c_end[2]*100:.0f}cm"
            elif sk == 'wave':
                ok = max_ez > 0.45 and (max(w1_hist[20:]) - min(w1_hist[20:])) > 0.5
                metrics_str = f"max_ez={max_ez*100:.0f}cm wrist_amp={max(w1_hist[20:])-min(w1_hist[20:]):.2f}rad"
            elif sk == 'reach_up':
                ok = max_ez > 0.68
                metrics_str = f"max_ez={max_ez*100:.0f}cm"
            elif sk == 'bow':
                ok = min_ez < 0.15
                metrics_str = f"min_ez={min_ez*100:.0f}cm"
            elif sk == 'circle':
                ok = x_span > 0.18 and y_span > 0.18
                metrics_str = f"span=({x_span*100:.0f}x{y_span*100:.0f})cm"
            elif sk == 'sweep':
                ok = y_span > 0.35
                metrics_str = f"y_span={y_span*100:.0f}cm"
            elif sk == 'clap_gripper':
                # Count open-close cycles
                g_arr = np.array(grip_hist) > 0.4
                flips = int(np.sum(g_arr[1:] != g_arr[:-1]))
                ok = g_rng > 0.5 and flips >= 3
                metrics_str = f"grip_flips={flips} rng={g_rng:.2f}"
            elif sk == 'poke_cube':
                ok = min_cube_eff < 0.065 and c_end[2] < 0.05
                metrics_str = f"min_dist={min_cube_eff*100:.1f}cm"
            elif sk == 'point':
                ok = abs(env.d.qpos[0] - (-math.pi)) < 0.7 or env.eff_pos()[1] < -0.35
                metrics_str = f"eff_y={env.eff_pos()[1]*100:+.0f}cm pan={math.degrees(env.d.qpos[0]):.0f}deg"
            elif sk == 'nod':
                amp = max(w1_hist[15:]) - min(w1_hist[15:])
                ok = amp > 0.6
                metrics_str = f"wrist_nod={amp:.2f}rad"
            elif sk == 'dance':
                ok = dpan > 0.6 and g_rng > 0.4
                metrics_str = f"pan_amp={dpan:.2f}rad grip_rng={g_rng:.2f}"
            elif sk == 'home':
                err_h = float(np.linalg.norm(env.d.qpos[:6] - np.array([-math.pi/2,-math.pi/2,math.pi/2,-math.pi/2,-math.pi/2,0])))
                ok = err_h < 0.6
                metrics_str = f"home_err={err_h:.2f}rad"
            ok_trials += int(ok)
        if ok_trials >= 2:
            passed += 1
        print(f"  [{ok_trials}/3] {prompt:44s} ({sk:12s}) | {metrics_str}", flush=True)
    print(f"  SUMMARY ({tag}): {passed}/{len(test_cases)} unseen English commands succeeded physically (>=2/3 trials)!", flush=True)
    return passed, len(test_cases)


def main():
    robot_dir = sys.argv[1] if len(sys.argv) > 1 else 'robot'
    data_dir = sys.argv[2] if len(sys.argv) > 2 else 'data_arena'
    d = np.load(os.path.join(data_dir, 'trajs.npz'))
    meta = json.load(open(os.path.join(data_dir, 'meta.json')))
    obs = d['obs'].astype(np.float32)
    act = d['act'].astype(np.float32)
    feat = d['feat'].astype(np.float32)
    caps = meta['captions']

    N_te = 2000
    obs_tr, act_tr, feat_tr = obs[:-N_te], act[:-N_te], feat[:-N_te]
    feat_te = feat[-N_te:]
    caps_tr, caps_te = caps[:-N_te], caps[-N_te:]

    ck = torch.load(os.path.join(data_dir, 'e_traj_pilot.pt'), map_location=DEV, weights_only=False)
    e_traj = TrajJEPA(fdim=feat.shape[2], d=256, layers=4, heads=4, dim=ck['dim'], max_t=feat.shape[1]).to(DEV).eval()
    e_traj.load_state_dict(ck['state'])
    Z_v_tr = encode_all_trajs(e_traj, feat_tr, ck['mu'], ck['sd'])
    Z_v_te = encode_all_trajs(e_traj, feat_te, ck['mu'], ck['sd'])

    k_v = 64
    mu_v, P_v, G_tr, G_te = whiten_pca(Z_v_tr, Z_v_te, k_v)

    pol, obs_mu, obs_sd = train_policy(obs_tr, act_tr, G_tr, steps=12000, bs=256, lr=8e-4, dz=k_v)
    torch.save({
        'state': pol.state_dict(),
        'obs_mu': obs_mu, 'obs_sd': obs_sd,
        'obs_dim': obs.shape[2], 'dz': k_v, 'd': pol.d, 'nblocks': pol.nblocks,
        'mu_v': mu_v, 'P_v': P_v,
    }, os.path.join(data_dir, 'behavior_policy.pt'))

    # Evaluate closed-loop unseen English -> MiniLM -> Single Orthogonal R -> UR5e!
    st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEV)
    E_tr = st.encode(caps_tr, batch_size=512, show_progress_bar=False).astype(np.float64)
    E_te = st.encode(caps_te, batch_size=512, show_progress_bar=False).astype(np.float64)
    k_t = 48
    mu_t, P_t, A_tr, A_te = whiten_pca(E_tr, E_te, k_t)
    R_10k = procrustes(A_tr[:10000], G_tr[:10000])
    rng = np.random.default_rng(7)
    R_shuf = procrustes(A_tr[:10000][rng.permutation(10000)], G_tr[:10000])

    eval_closed_loop_english(robot_dir, pol, obs_mu, obs_sd, st, mu_t, P_t, R_10k, tag='MiniLM N=10000')
    eval_closed_loop_english(robot_dir, pol, obs_mu, obs_sd, st, mu_t, P_t, R_shuf, tag='SHUFFLED CONTROL')
    print("\nTRAIN_POLICY_DONE", flush=True)


if __name__ == '__main__':
    main()
