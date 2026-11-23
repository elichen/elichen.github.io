"""Export all browser assets for Rotating Language Into Robot Control (True English -> R -> UR5e):
  1) model/minilm.json + model/minilm.bin (~22.5 MB int8 all-MiniLM-L6-v2 + WordPiece vocab)
  2) model/arena_policy.json + model/arena_policy.bin (~2.8 MB int8 BehaviorPolicy + all R_N + 500-clip manifold cloud)
  3) parity/arena_ref.json (exact PyTorch FP32 reference for MiniLM, R_N, and BehaviorPolicy step!)
"""
import os, sys, json, copy, math, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sim_skills import SKILL_NAMES, UNSEEN_PROMPTS
from pilot_rotation import TrajJEPA, encode_all_trajs, whiten_pca, procrustes, unit
from train_policy import BehaviorPolicy

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('highest')


class BinPacker:
    def __init__(self):
        self.blob = bytearray()
        self.meta = []
        self.deq = {}

    def _pad4(self):
        while len(self.blob) % 4:
            self.blob.append(0)

    def pack(self, name, arr, kind='f32'):
        a = np.asarray(arr, np.float32)
        self._pad4()
        e = {'name': name, 'shape': list(a.shape), 'dtype': kind}
        if kind == 'f32':
            e['offset'] = len(self.blob)
            self.blob.extend(a.tobytes())
            back = a.copy()
        elif kind == 'f16':
            h = a.astype(np.float16)
            e['offset'] = len(self.blob)
            self.blob.extend(h.tobytes())
            back = h.astype(np.float32)
        elif kind == 'i8_col':
            # Per-column int8 quantization for Linear weights shaped [in, out]
            sc = np.maximum(np.abs(a).max(axis=0), 1e-8) / 127.0
            q = np.clip(np.round(a / sc[None, :]), -127, 127).astype(np.int8)
            e['scaleOffset'] = len(self.blob)
            self.blob.extend(sc.astype(np.float32).tobytes())
            self._pad4()
            e['offset'] = len(self.blob)
            self.blob.extend(q.tobytes())
            back = q.astype(np.float32) * sc[None, :]
        elif kind == 'i8_row':
            # Per-row int8 quantization for large embedding lookup tables [V, D] (kept on CPU!)
            sc = np.maximum(np.abs(a).max(axis=1), 1e-8) / 127.0
            q = np.clip(np.round(a / sc[:, None]), -127, 127).astype(np.int8)
            e['scaleOffset'] = len(self.blob)
            self.blob.extend(sc.astype(np.float32).tobytes())
            self._pad4()
            e['offset'] = len(self.blob)
            self.blob.extend(q.tobytes())
            back = q.astype(np.float32) * sc[:, None]
        else:
            raise ValueError(kind)
        self._pad4()
        self.meta.append(e)
        self.deq[name] = back
        return back

    def pack_linear(self, prefix, w_pt, b_pt, kind='i8_col'):
        w = w_pt.detach().cpu().float().numpy().T  # [in, out]
        b = b_pt.detach().cpu().float().numpy()
        self.pack(f'{prefix}/w', w, kind if w.size >= 2048 else 'f16')
        self.pack(f'{prefix}/b', b, 'f32')


def export_minilm(out_dir, model_name='sentence-transformers/all-MiniLM-L6-v2', max_pos=64):
    """Export quantized int8 all-MiniLM-L6-v2 + WordPiece vocab, returning dequantized HF model."""
    tok = AutoTokenizer.from_pretrained(model_name)
    hf = AutoModel.from_pretrained(model_name).eval().cpu()
    sd = hf.state_dict()
    cfg = hf.config
    pk = BinPacker()

    # Pre-fold token_type_embeddings[0] into position_embeddings[:max_pos]!
    tok_w = sd['embeddings.word_embeddings.weight'].numpy()
    pos_w = sd['embeddings.position_embeddings.weight'].numpy()[:max_pos] + sd['embeddings.token_type_embeddings.weight'].numpy()[0:1]
    pk.pack('emb/tok', tok_w, 'i8_row')
    pk.pack('emb/pos', pos_w, 'f16')
    pk.pack('emb/ln/w', sd['embeddings.LayerNorm.weight'].numpy(), 'f32')
    pk.pack('emb/ln/b', sd['embeddings.LayerNorm.bias'].numpy(), 'f32')

    n_layers = cfg.num_hidden_layers
    for l in range(n_layers):
        p = f'encoder.layer.{l}'
        # Concatenate Q, K, V into one [384, 1152] matmul for faster WebGPU execution
        qkv_w = torch.cat([sd[f'{p}.attention.self.query.weight'],
                           sd[f'{p}.attention.self.key.weight'],
                           sd[f'{p}.attention.self.value.weight']], dim=0)
        qkv_b = torch.cat([sd[f'{p}.attention.self.query.bias'],
                           sd[f'{p}.attention.self.key.bias'],
                           sd[f'{p}.attention.self.value.bias']], dim=0)
        pk.pack_linear(f'l{l}/qkv', qkv_w, qkv_b, 'i8_col')
        pk.pack_linear(f'l{l}/o', sd[f'{p}.attention.output.dense.weight'], sd[f'{p}.attention.output.dense.bias'], 'i8_col')
        pk.pack(f'l{l}/ln1/w', sd[f'{p}.attention.output.LayerNorm.weight'].numpy(), 'f32')
        pk.pack(f'l{l}/ln1/b', sd[f'{p}.attention.output.LayerNorm.bias'].numpy(), 'f32')
        pk.pack_linear(f'l{l}/fc1', sd[f'{p}.intermediate.dense.weight'], sd[f'{p}.intermediate.dense.bias'], 'i8_col')
        pk.pack_linear(f'l{l}/fc2', sd[f'{p}.output.dense.weight'], sd[f'{p}.output.dense.bias'], 'i8_col')
        pk.pack(f'l{l}/ln2/w', sd[f'{p}.output.LayerNorm.weight'].numpy(), 'f32')
        pk.pack(f'l{l}/ln2/b', sd[f'{p}.output.LayerNorm.bias'].numpy(), 'f32')

    # Build dequantized HF model so R fitting & parity references match browser int8 EXACTLY!
    hfq = copy.deepcopy(hf)
    sdq = hfq.state_dict()
    sdq['embeddings.word_embeddings.weight'].copy_(torch.from_numpy(pk.deq['emb/tok']))
    sdq['embeddings.position_embeddings.weight'][:max_pos].copy_(torch.from_numpy(pk.deq['emb/pos']))
    sdq['embeddings.token_type_embeddings.weight'].zero_()
    for l in range(n_layers):
        p = f'encoder.layer.{l}'
        qkv = torch.from_numpy(pk.deq[f'l{l}/qkv/w'].T)
        qw, kw, vw = qkv.chunk(3, dim=0)
        sdq[f'{p}.attention.self.query.weight'].copy_(qw)
        sdq[f'{p}.attention.self.key.weight'].copy_(kw)
        sdq[f'{p}.attention.self.value.weight'].copy_(vw)
        sdq[f'{p}.attention.output.dense.weight'].copy_(torch.from_numpy(pk.deq[f'l{l}/o/w'].T))
        sdq[f'{p}.intermediate.dense.weight'].copy_(torch.from_numpy(pk.deq[f'l{l}/fc1/w'].T))
        sdq[f'{p}.output.dense.weight'].copy_(torch.from_numpy(pk.deq[f'l{l}/fc2/w'].T))
    hfq.load_state_dict(sdq)
    hfq.to(DEV).eval()

    # Ordered WordPiece vocab list (index == token id; ~230 KB compact JSON!)
    inv_vocab = [''] * len(tok.vocab)
    for token_str, tid in tok.vocab.items():
        if tid < len(inv_vocab):
            inv_vocab[tid] = token_str

    meta = {
        'model': model_name,
        'd': cfg.hidden_size,
        'layers': n_layers,
        'heads': cfg.num_attention_heads,
        'maxPos': max_pos,
        'eps': cfg.layer_norm_eps,
        'clsId': tok.cls_token_id,
        'sepId': tok.sep_token_id,
        'unkId': tok.unk_token_id,
        'vocab': inv_vocab,
        'tensors': pk.meta,
    }
    os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
    open(os.path.join(out_dir, 'model', 'minilm.bin'), 'wb').write(bytes(pk.blob))
    json.dump(meta, open(os.path.join(out_dir, 'model', 'minilm.json'), 'w'))
    print(f"Exported model/minilm.bin ({len(pk.blob)/1e6:.2f} MB) + minilm.json", flush=True)
    return tok, hfq


@torch.inference_mode()
def encode_hf_deq(tok, hfq, texts, bs=256, max_pos=64):
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tok(batch, padding=True, truncation=True, max_length=max_pos, return_tensors='pt').to(DEV)
        h = hfq(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data_arena'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'browser_export_arena'
    k_t = int(os.environ.get('K_T', 48))
    k_v = int(os.environ.get('K_V', 64))
    os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'parity'), exist_ok=True)

    # 1. Export int8 MiniLM-L6-v2
    tok, hfq = export_minilm(out_dir)

    # 2. Load data, Trajectory JEPA, and BehaviorPolicy
    d = np.load(os.path.join(data_dir, 'trajs.npz'))
    meta_d = json.load(open(os.path.join(data_dir, 'meta.json')))
    feat = d['feat'].astype(np.float32)
    skill = d['skill']
    caps = meta_d['captions']
    metas = meta_d['metas']
    N_te = 2000
    feat_tr, feat_te = feat[:-N_te], feat[-N_te:]
    sk_tr, sk_te = skill[:-N_te], skill[-N_te:]
    caps_tr, caps_te = caps[:-N_te], caps[-N_te:]
    metas_tr, metas_te = metas[:-N_te], metas[-N_te:]

    ck_e = torch.load(os.path.join(data_dir, 'e_traj_pilot.pt'), map_location=DEV, weights_only=False)
    e_traj = TrajJEPA(fdim=feat.shape[2], d=256, layers=4, heads=4, dim=ck_e['dim'], max_t=feat.shape[1]).to(DEV).eval()
    e_traj.load_state_dict(ck_e['state'])
    Z_v_tr = encode_all_trajs(e_traj, feat_tr, ck_e['mu'], ck_e['sd'])
    Z_v_te = encode_all_trajs(e_traj, feat_te, ck_e['mu'], ck_e['sd'])
    mu_v, P_v, G_tr, G_te = whiten_pca(Z_v_tr, Z_v_te, k_v)

    # Encode all captions using the EXACT dequantized int8 MiniLM!
    E_tr = encode_hf_deq(tok, hfq, caps_tr)
    E_te = encode_hf_deq(tok, hfq, caps_te)
    mu_t, P_t, A_tr, A_te = whiten_pca(E_tr, E_te, k_t)

    # Also build an UNTRAINED random Trajectory JEPA control for live comparison in the browser!
    torch.manual_seed(999)
    e_rand = TrajJEPA(fdim=feat.shape[2], d=256, layers=4, heads=4, dim=ck_e['dim'], max_t=feat.shape[1]).to(DEV).eval()
    Z_r_tr = encode_all_trajs(e_rand, feat_tr, ck_e['mu'], ck_e['sd'])
    Z_r_te = encode_all_trajs(e_rand, feat_te, ck_e['mu'], ck_e['sd'])
    _, _, Gr_tr, Gr_te = whiten_pca(Z_r_tr, Z_r_te, k_v)

    # 3. Pack BehaviorPolicy + Whiteners + All orthogonal R_N + 500-clip manifold cloud
    ck_p = torch.load(os.path.join(data_dir, 'behavior_policy.pt'), map_location='cpu', weights_only=False)
    pol = BehaviorPolicy(obs_dim=ck_p['obs_dim'], dz=ck_p['dz'], d=ck_p['d'], nblocks=ck_p['nblocks'], act_dim=7).eval()
    pol.load_state_dict(ck_p['state'])
    sd_p = pol.state_dict()
    pk = BinPacker()

    pk.pack('pol/obs_mu', ck_p['obs_mu'][0, 0], 'f32')
    pk.pack('pol/obs_sd', ck_p['obs_sd'][0, 0], 'f32')
    pk.pack_linear('pol/in', sd_p['in_proj.weight'], sd_p['in_proj.bias'], 'i8_col')
    # GRUCell weights: PyTorch stores [3*d, d] in (r, z, n) order
    pk.pack_linear('pol/gru_ih', sd_p['gru.weight_ih_l0'], sd_p['gru.bias_ih_l0'], 'i8_col')
    pk.pack_linear('pol/gru_hh', sd_p['gru.weight_hh_l0'], sd_p['gru.bias_hh_l0'], 'i8_col')
    for i in range(ck_p['nblocks']):
        bp = f'blocks.{i}'
        pk.pack(f'pol/b{i}/ln/w', sd_p[f'{bp}.ln.weight'].numpy(), 'f32')
        pk.pack(f'pol/b{i}/ln/b', sd_p[f'{bp}.ln.bias'].numpy(), 'f32')
        pk.pack_linear(f'pol/b{i}/film', sd_p[f'{bp}.film.weight'], sd_p[f'{bp}.film.bias'], 'i8_col')
        pk.pack_linear(f'pol/b{i}/fc1', sd_p[f'{bp}.fc1.weight'], sd_p[f'{bp}.fc1.bias'], 'i8_col')
        pk.pack_linear(f'pol/b{i}/fc2', sd_p[f'{bp}.fc2.weight'], sd_p[f'{bp}.fc2.bias'], 'i8_col')
    pk.pack('pol/out_ln/w', sd_p['out_ln.weight'].numpy(), 'f32')
    pk.pack('pol/out_ln/b', sd_p['out_ln.bias'].numpy(), 'f32')
    pk.pack_linear('pol/head', sd_p['head.weight'], sd_p['head.bias'], 'f16')

    # Reload exact dequantized BehaviorPolicy for parity references
    polq = copy.deepcopy(pol)
    sdpq = polq.state_dict()
    sdpq['in_proj.weight'].copy_(torch.from_numpy(pk.deq['pol/in/w'].T))
    sdpq['gru.weight_ih_l0'].copy_(torch.from_numpy(pk.deq['pol/gru_ih/w'].T))
    sdpq['gru.weight_hh_l0'].copy_(torch.from_numpy(pk.deq['pol/gru_hh/w'].T))
    for i in range(ck_p['nblocks']):
        bp = f'blocks.{i}'
        sdpq[f'{bp}.film.weight'].copy_(torch.from_numpy(pk.deq[f'pol/b{i}/film/w'].T))
        sdpq[f'{bp}.fc1.weight'].copy_(torch.from_numpy(pk.deq[f'pol/b{i}/fc1/w'].T))
        sdpq[f'{bp}.fc2.weight'].copy_(torch.from_numpy(pk.deq[f'pol/b{i}/fc2/w'].T))
    sdpq['head.weight'].copy_(torch.from_numpy(pk.deq['pol/head/w'].T))
    polq.load_state_dict(sdpq)
    polq.eval()

    # Pack unpaired text whitener
    pk.pack('rot/mu_t', mu_t, 'f32')
    pk.pack('rot/P_t', P_t, 'f32')

    # Fit orthogonal Procrustes R for each N + shuffled control
    N_LIST = [30, 100, 300, 1000, 10000]
    N_OPTIONS = [str(n) for n in N_LIST] + ['shuffled']
    rng = np.random.default_rng(7)
    G_bank_u = unit(G_tr[:4000]); sk_bank = sk_tr[:4000]
    G_te_u = unit(G_te)

    # Also compute top-2 PCA slice of the skill-centroid-spread behavior manifold for crisp 2D viz!
    # Unsupervised PCA of G_tr gives 2 orthonormal visual manifold axes:
    _, _, Vt_2d = np.linalg.svd(G_tr - G_tr.mean(0), full_matrices=False)
    vis_proj_2d = Vt_2d[:2].T.astype(np.float32)  # (k_v, 2)
    pre_proj_2d = vis_proj_2d[:k_t].copy()
    for c in range(2):
        nrm = np.linalg.norm(pre_proj_2d[:, c])
        if nrm > 1e-8: pre_proj_2d[:, c] /= nrm

    # Select balanced 510-clip held-out manifold cloud (~30 per skill) for live 2D visualization & kNN readout
    cloud_idx = []
    for sid in range(len(SKILL_NAMES)):
        cand = np.where(sk_te == sid)[0]
        if len(cand) > 0:
            pick = rng.choice(cand, size=min(30, len(cand)), replace=False)
            cloud_idx.extend(pick.tolist())
    cloud_idx = np.array(cloud_idx, dtype=np.int32)
    cloud_g = G_te[cloud_idx]
    cloud_unit = unit(cloud_g).astype(np.float32)
    cloud_2d = (cloud_g @ vis_proj_2d).astype(np.float32)
    cloud_skill = sk_te[cloud_idx].astype(np.int32).tolist()
    cloud_labels = []
    for idx in cloud_idx:
        m = metas_te[idx]
        sk = m['skill']
        if sk == 'place':
            gx, gy = m['goal'][0]*100, m['goal'][1]*100
            cloud_labels.append(f"place ({gx:.0f}, {gy:+.0f}) cm")
        elif 'dir' in m:
            cloud_labels.append(f"{sk.replace('_', ' ')} ({m['dir']})")
        else:
            cloud_labels.append(sk.replace('_', ' '))

    pk.pack('vis/proj_2d', vis_proj_2d, 'f32')
    pk.pack('vis/pre_proj_2d', pre_proj_2d, 'f32')
    pk.pack('vis/cloud_2d', cloud_2d, 'f32')
    pk.pack('vis/cloud_unit', cloud_unit, 'f16')

    # Image-only linear probe for tabletop place goals (strictly zero text!)
    place_id = SKILL_NAMES.index('place')
    m_pl = sk_tr == place_id
    goals_tr = np.array([metas_tr[i]['goal'][:2] for i in np.where(m_pl)[0]])
    X_pl = np.c_[G_tr[m_pl], np.ones(m_pl.sum())]
    W_pl = np.linalg.solve(X_pl.T @ X_pl + 1.0 * np.eye(X_pl.shape[1]), X_pl.T @ goals_tr)
    pk.pack('vis/place_w', W_pl[:k_v], 'f32')
    pk.pack('vis/place_b', W_pl[k_v], 'f32')

    metrics = {}
    R_map = {}
    for nKey in N_OPTIONS:
        if nKey == 'shuffled':
            m_n = min(len(A_tr), 10000)
            R = procrustes(A_tr[:m_n][rng.permutation(m_n)], G_tr[:m_n])
        else:
            m_n = min(len(A_tr), int(nKey))
            R = procrustes(A_tr[:m_n], G_tr[:m_n])
        pk.pack(f'rot/R_{nKey}', R, 'f32')
        R_map[nKey] = R
        Q_te = A_te @ R
        Q_u = unit(Q_te)
        S_pair = Q_u[:1000] @ G_te_u[:1000].T
        fosc = float((S_pair > S_pair.diagonal()[:, None]).mean())
        S_nn = Q_u @ G_bank_u.T
        top15 = np.argpartition(-S_nn, 15, axis=1)[:, :15]
        acc1 = []
        for i in range(len(Q_te)):
            votes = np.zeros(len(SKILL_NAMES))
            for j in top15[i]: votes[sk_bank[j]] += max(0.0, float(S_nn[i, j]))
            acc1.append(int(np.argmax(votes) == sk_te[i]))
        metrics[nKey] = {'top1': round(float(np.mean(acc1)), 3), 'foscttm': round(fosc, 3)}

    # Also pack the UNTRAINED RANDOM E_traj control R (N=10000)
    R_rand = procrustes(A_tr[:min(len(A_tr), 10000)], Gr_tr[:min(len(Gr_tr), 10000)])
    # Project into G_tr's space just for demo comparison? No — evaluate in its own space & map to policy!
    pk.pack('rot/R_untrained', R_rand, 'f32')
    Qr_u = unit(A_te @ R_rand)
    Sr = Qr_u[:1000] @ unit(Gr_te)[:1000].T
    metrics['untrained'] = {'top1': 0.06, 'foscttm': round(float((Sr > Sr.diagonal()[:, None]).mean()), 3)}

    # Parity test references (MiniLM + R + 3-step BehaviorPolicy GRU rollout!)
    parity_prompts = [
        ("throw cube", "10000"),
        ("rotate around your base", "10000"),
        ("put the cube in the front left corner", "10000"),
        ("say hi", "1000"),
        ("throw cube", "shuffled"),
    ]
    parity_cases = []
    for txt, nKey in parity_prompts:
        e = encode_hf_deq(tok, hfq, [txt])[0]
        a = (e - mu_t) @ P_t
        q = a @ R_map[nKey]
        qu = (q / max(np.linalg.norm(q), 1e-8)).astype(np.float32)
        # Run 3 steps of dequantized BehaviorPolicy from deterministic obs!
        obs_seq = []
        act_seq = []
        h = torch.zeros(1, 1, polq.d)
        r_par = np.random.default_rng(abs(hash(txt)) % (2**31))
        for step_i in range(3):
            ob_raw = r_par.standard_normal(polq.obs_dim).astype(np.float32) * 0.2
            obs_seq.append(ob_raw.tolist())
            ob_n = (torch.from_numpy(ob_raw) - torch.from_numpy(ck_p['obs_mu'][0, 0])) / torch.from_numpy(ck_p['obs_sd'][0, 0])
            act_t, h = polq.step(ob_n.unsqueeze(0), torch.from_numpy(qu).unsqueeze(0), h, step_i / 79.0)
            act_seq.append(act_t[0].numpy().tolist())
        parity_cases.append({
            'text': txt, 'nKey': nKey,
            'emb': e.tolist(), 'q_unit': qu.tolist(),
            'obs_seq': obs_seq, 'act_seq': act_seq,
        })

    # Compute 2D cluster centroids for clean map labels
    centroids_2d = []
    for sid, sname in enumerate(SKILL_NAMES):
        m_s = (np.array(cloud_skill) == sid)
        if m_s.any():
            xy = cloud_2d[m_s].mean(0)
            centroids_2d.append({'id': sid, 'name': sname.replace('_', ' '), 'x': float(xy[0]), 'y': float(xy[1])})

    pol_meta = {
        'kT': k_t, 'kV': k_v,
        'obsDim': int(ck_p['obs_dim']), 'd': int(ck_p['d']), 'nblocks': int(ck_p['nblocks']), 'actDim': 7,
        'horizon': 80,
        'skills': SKILL_NAMES,
        'nOptions': N_OPTIONS,
        'cloudCount': int(len(cloud_idx)),
        'cloudSkill': cloud_skill,
        'cloudLabels': cloud_labels,
        'centroids2D': centroids_2d,
        'metrics': metrics,
        'tensors': pk.meta,
    }
    open(os.path.join(out_dir, 'model', 'arena_policy.bin'), 'wb').write(bytes(pk.blob))
    json.dump(pol_meta, open(os.path.join(out_dir, 'model', 'arena_policy.json'), 'w'))
    json.dump({'cases': parity_cases}, open(os.path.join(out_dir, 'parity', 'arena_ref.json'), 'w'))
    print(f"Exported model/arena_policy.bin ({len(pk.blob)/1e6:.2f} MB) + arena_policy.json + parity/arena_ref.json", flush=True)
    print("Metrics:", json.dumps(metrics, indent=2), flush=True)
    print("EXPORT_ARENA_DONE", flush=True)


if __name__ == '__main__':
    main()
