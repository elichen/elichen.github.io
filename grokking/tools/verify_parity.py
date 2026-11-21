#!/usr/bin/env python3
"""Verify exact numerical parity between grok.wasm (via Node.js) and PyTorch,
and benchmark live WASM SIMD training speed."""
import json, math, subprocess, tempfile, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P, V, D, H, DH, DM = 59, 60, 64, 4, 16, 192
N_PARAMS = V*D + 3*D + 4*D*D + D*DM + DM + DM*D + D + D*P

WASM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "grok.wasm")

NODE_HELPER = r"""
const fs = require('fs');
const wasmBytes = fs.readFileSync(process.argv[2]);
const outPath = process.argv[3];

(async () => {
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const e = instance.exports;
  const mem = () => new Float32Array(e.memory.buffer);
  const memU16 = () => new Uint16Array(e.memory.buffer);

  e.init_model(0, 0.35, 3e-3, 2.2, 1);
  const nParams = e.get_n_params();
  const nTrain = e.get_n_train();
  const pPtr = e.get_params_ptr() >> 2;
  const gPtr = e.get_grads_ptr() >> 2;
  const aPtr = e.get_train_pairs_a_ptr() >> 1;
  const bPtr = e.get_train_pairs_b_ptr() >> 1;
  const yPtr = e.get_train_labels_ptr() >> 1;

  const initParams = Array.from(mem().subarray(pPtr, pPtr + nParams));
  const trA = Array.from(memU16().subarray(aPtr, aPtr + nTrain));
  const trB = Array.from(memU16().subarray(bPtr, bPtr + nTrain));
  const trY = Array.from(memU16().subarray(yPtr, yPtr + nTrain));

  const loss1 = e.step_n(1);
  const grads1 = Array.from(mem().subarray(gPtr, gPtr + nParams));
  const params1 = Array.from(mem().subarray(pPtr, pPtr + nParams));

  e.step_n(9); // total 10 steps
  const params10 = Array.from(mem().subarray(pPtr, pPtr + nParams));

  const evalPtr = e.evaluate() >> 2;
  const evalLen = e.get_eval_len();
  const eval10 = Array.from(mem().subarray(evalPtr, evalPtr + evalLen));

  // Benchmark 100 steps
  const t0 = performance.now();
  e.step_n(100);
  const dt = performance.now() - t0;

  // Run default grokking trajectory to 2500 steps (wd=2.2, seed=1) and time it
  e.init_model(0, 0.35, 3e-3, 2.2, 1);
  const tG0 = performance.now();
  const checkpoints = [];
  for (let ep = 100; ep <= 2500; ep += 100) {
    e.step_n(100);
    const epPtr = e.evaluate() >> 2;
    const ev = Array.from(mem().subarray(epPtr, epPtr + evalLen));
    checkpoints.push({
      ep,
      tr_acc: ev[1], te_acc: ev[3], tr_loss: ev[0], te_loss: ev[2],
      wnorm: ev[4], res_loss: ev[5], exc_loss: ev[6], top5: ev[7], gini: ev[8],
      top5_k: ev.slice(9, 14)
    });
  }
  const dtGrok = (performance.now() - tG0) / 1000;

  // Test causal ablation at ep=2500
  const last = checkpoints[checkpoints.length - 1];
  let top5Mask = 0;
  for (const k of last.top5_k) top5Mask |= (1 << (k - 1));
  const fullMask = (1 << 29) - 1;
  const soloPtr = e.eval_ablated(top5Mask) >> 2;
  const solo = Array.from(mem().subarray(soloPtr, soloPtr + 4));
  const mutePtr = e.eval_ablated(fullMask ^ top5Mask) >> 2;
  const mute = Array.from(mem().subarray(mutePtr, mutePtr + 4));

  fs.writeFileSync(outPath, JSON.stringify({
    nParams, nTrain, initParams, trA, trB, trY,
    loss1, grads1, params1, params10, eval10,
    msPer100: dt, epochsPerSec: 100 / (dt / 1000),
    dtGrokSec: dtGrok, checkpoints, solo, mute
  }));
})();
"""

with tempfile.TemporaryDirectory() as td:
    js_path = os.path.join(td, "run.js")
    out_path = os.path.join(td, "out.json")
    open(js_path, "w").write(NODE_HELPER)
    subprocess.run(["node", js_path, WASM_PATH, out_path], check=True)
    data = json.load(open(out_path))

print(f"WASM params={data['nParams']} (expected {N_PARAMS}), nTrain={data['nTrain']}")
print(f"Speed: {data['msPer100']:.1f} ms / 100 epochs = {data['epochsPerSec']:.1f} epochs/sec")
print(f"Full 2500-epoch live grokking + 25 full evals: {data['dtGrokSec']:.2f} s")

# Unpack params into PyTorch reference (using FULL sequence-level transformer, NOT factored!)
def unpack(flat):
    t = torch.tensor(flat, dtype=torch.float32)
    off = 0
    def take(shape):
        nonlocal off
        n = math.prod(shape)
        x = t[off:off+n].view(shape).clone()
        off += n
        return nn.Parameter(x)
    return {
        "W_E": take((V, D)),
        "W_pos": take((3, D)),
        "W_Q": take((D, D)),
        "W_K": take((D, D)),
        "W_V": take((D, D)),
        "W_O": take((D, D)),
        "W_in": take((D, DM)),
        "b_in": take((DM,)),
        "W_out": take((DM, D)),
        "b_out": take((D,)),
        "W_U": take((D, P)),
    }

def forward_full_seq(params, pairs):
    # Naive sequence-level 1-layer causal Transformer on [a, b, '=']!
    B = pairs.shape[0]
    eq = torch.full((B, 1), P, dtype=torch.long)
    seq = torch.cat([pairs, eq], dim=1) # [B, 3]
    x = params["W_E"][seq] + params["W_pos"].unsqueeze(0) # [B, 3, D]
    Q = (x @ params["W_Q"]).view(B, 3, H, DH).transpose(1, 2) # [B, H, 3, DH]
    K = (x @ params["W_K"]).view(B, 3, H, DH).transpose(1, 2)
    V_ = (x @ params["W_V"]).view(B, 3, H, DH).transpose(1, 2)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(DH) # [B, H, 3, 3]
    mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    w = F.softmax(scores, dim=-1)
    attn = (w @ V_).transpose(1, 2).reshape(B, 3, D) @ params["W_O"]
    x_mid = x[:, 2, :] + attn[:, 2, :] # read at pos 2 ('=')
    h = F.relu(x_mid @ params["W_in"] + params["b_in"])
    out = x_mid + h @ params["W_out"] + params["b_out"]
    return out @ params["W_U"]

pairs_tr = torch.tensor(list(zip(data["trA"], data["trB"])), dtype=torch.long)
y_tr = torch.tensor(data["trY"], dtype=torch.long)

params = unpack(data["initParams"])
order = ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "b_in", "W_out", "b_out", "W_U"]
opt = torch.optim.AdamW([params[k] for k in order], lr=3e-3, betas=(0.9, 0.98), weight_decay=2.2)

opt.zero_grad()
logits1 = forward_full_seq(params, pairs_tr)
loss1 = F.cross_entropy(logits1, y_tr)
loss1.backward()

g_torch = torch.cat([params[k].grad.flatten() for k in order]).numpy()
g_wasm = np.array(data["grads1"], dtype=np.float32)
print(f"Step 1 loss: torch={loss1.item():.7f} wasm={data['loss1']:.7f} diff={abs(loss1.item()-data['loss1']):.2e}")
print(f"Step 1 grad max abs diff (all 49,024 params vs naive full-seq Transformer): {np.max(np.abs(g_torch - g_wasm)):.2e}")

opt.step()
p1_torch = torch.cat([params[k].detach().flatten() for k in order]).numpy()
p1_wasm = np.array(data["params1"], dtype=np.float32)
print(f"Step 1 param max abs diff: {np.max(np.abs(p1_torch - p1_wasm)):.2e}")

for _ in range(9):
    opt.zero_grad()
    F.cross_entropy(forward_full_seq(params, pairs_tr), y_tr).backward()
    opt.step()

p10_torch = torch.cat([params[k].detach().flatten() for k in order]).numpy()
p10_wasm = np.array(data["params10"], dtype=np.float32)
print(f"Step 10 param max abs diff: {np.max(np.abs(p10_torch - p10_wasm)):.2e}")

# Verify restricted/excluded loss parity at step 10 against PyTorch 2D FFT
all_pairs = torch.tensor([(a, b) for a in range(P) for b in range(P)], dtype=torch.long)
all_y = torch.tensor([(a + b) % P for a in range(P) for b in range(P)], dtype=torch.long)
tr_set = set(zip(data["trA"], data["trB"]))
is_tr = torch.tensor([(a, b) in tr_set for a in range(P) for b in range(P)], dtype=torch.bool)
with torch.no_grad():
    L = forward_full_seq(params, all_pairs).view(P, P, P)
    top5 = [int(x) for x in data["eval10"][9:14]]
    F2 = torch.fft.fft2(L, dim=(0, 1))
    mask = torch.zeros(P, P, dtype=torch.bool)
    mask[0, 0] = True
    for k in top5:
        for s1 in (k, P - k):
            for s2 in (k, P - k):
                mask[s1, s2] = True
    L_res = torch.fft.ifft2(F2 * mask.unsqueeze(-1), dim=(0, 1)).real.reshape(P*P, P)
    L_exc = L.reshape(P*P, P) - L_res
    res_t = F.cross_entropy(L_res[~is_tr], all_y[~is_tr]).item()
    exc_t = F.cross_entropy(L_exc[is_tr], all_y[is_tr]).item()
print(f"Step 10 res_loss: torch={res_t:.6f} wasm={data['eval10'][5]:.6f} diff={abs(res_t-data['eval10'][5]):.2e}")
print(f"Step 10 exc_loss: torch={exc_t:.6f} wasm={data['eval10'][6]:.6f} diff={abs(exc_t-data['eval10'][6]):.2e}")

print("\n=== Live WASM Grokking Trajectory (op=add, p=59, frac=0.35, wd=2.2, seed=1) ===")
for c in data["checkpoints"]:
    if c["ep"] in (100, 200, 400, 700, 1000, 1300, 1600, 1900, 2200, 2500):
        print(f"ep={c['ep']:4d} tr_acc={c['tr_acc']:.3f} te_acc={c['te_acc']:.3f} te_loss={c['te_loss']:6.3f} res={c['res_loss']:5.3f} exc={c['exc_loss']:6.3f} wnorm={c['wnorm']:4.1f} top5={c['top5']:.2f} gini={c['gini']:.2f}")
print(f"Causal ablation @ ep 2500: solo_top5 te_acc={data['solo'][1]:.3f}, mute_top5 te_acc={data['mute'][1]:.3f}")
