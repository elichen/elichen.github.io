"""Export Gemma 3 270M + Gemma Scope 2 (layer 12, 16k medium) for the page, and references for tools/parity.mjs.

  model/      int8 weights, one fp32 scale per row (served from outside GitHub Pages; ~290 MB)
  data/       tokenizer, concept stats + steering strengths, example snippets
  tools/test/ references computed in PyTorch from the same int8 weights (dequantized), fp32 activations

python3 tools/export.py <strength table.json> <texts.json>
"""
import json, os, struct, sys
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL, SCOPE, LAYER = "google/gemma-3-270m", "google/gemma-scope-2-270m-pt", 12
SUB = f"resid_post/layer_{LAYER}_width_16k_l0_medium"
for d in ["model", "data", "tools/test"]:
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config
sae = load_file(hf_hub_download(SCOPE, f"{SUB}/params.safetensors"))
ex = safe_open(hf_hub_download(SCOPE, f"{SUB}/examples.safetensors"), "pt")
table = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}


def q8(w):
    """per-row symmetric int8: returns int8 rows, fp32 scales, and the dequantized matrix the page computes with"""
    s = (w.abs().amax(1) / 127).clamp(min=1e-12)
    q = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q, s.float(), q.float() * s[:, None]


class Writer:
    """tensors packed into files; offsets 16-byte aligned; the manifest says where each one is"""
    def __init__(self):
        self.tensors, self.files = {}, {}

    def add(self, file, name, t, dtype):
        a = t.contiguous().numpy().astype(dtype)
        f = self.files.setdefault(file, bytearray())
        f += b"\0" * (-len(f) % 16)
        self.tensors[name] = {"file": file, "offset": len(f), "bytes": a.nbytes, "dtype": np.dtype(dtype).name, "shape": list(a.shape)}
        f += a.tobytes()

    def save(self, extra):
        for name, data in self.files.items():
            open(f"{ROOT}/model/{name}", "wb").write(data)
        json.dump({**extra, "files": {k: len(v) for k, v in self.files.items()}, "tensors": self.tensors},
                  open(f"{ROOT}/model/manifest.json", "w"), indent=1)


W = Writer()
# embedding (= lm_head), 4 files of 65,536 rows so each GPU buffer stays well under 128 MB
q, s, dq = q8(model.model.embed_tokens.weight)
model.model.embed_tokens.weight.copy_(dq)
for c in range(4):
    W.add(f"embed{c}.bin", f"embed.{c}", q[c * 65536:(c + 1) * 65536], np.int8)
W.add("embed_scales.bin", "embed.scales", s, np.float32)
LIN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
NORMS = ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
         "self_attn.q_norm", "self_attn.k_norm"]
for i, layer in enumerate(model.model.layers):
    f = f"layers{i // 6}.bin"
    for n in LIN:
        lin = layer.get_submodule(n)
        q, s, dq = q8(lin.weight)
        lin.weight.copy_(dq)
        W.add(f, f"l{i}.{n}", q, np.int8)
        W.add(f, f"l{i}.{n}.scales", s, np.float32)
    for n in NORMS:
        W.add(f, f"l{i}.{n}", 1 + layer.get_submodule(n).weight, np.float32)     # Gemma norms scale by (1 + w)
W.add("layers2.bin", "norm", 1 + model.model.norm.weight, np.float32)
MAXPOS = 512                                    # the page caps a sequence at 512 tokens, so sliding windows never bind
for lt in ["sliding_attention", "full_attention"]:
    cos, sin = model.model.rotary_emb(torch.zeros(1), torch.arange(MAXPOS)[None], lt)
    W.add("layers2.bin", f"rope.{lt}", torch.stack([cos[0, :, :128], sin[0, :, :128]], 1), np.float32)   # (pos, cos|sin, 128)
# SAE: encoder as one row per concept (W_enc^T), decoder rows
qe, se, dqe = q8(sae["w_enc"].T)
qd, sd, dqd = q8(sae["w_dec"])
W.add("sae.bin", "sae.enc", qe, np.int8)
W.add("sae.bin", "sae.enc.scales", se, np.float32)
W.add("sae.bin", "sae.b_enc", sae["b_enc"], np.float32)
W.add("sae.bin", "sae.threshold", sae["threshold"], np.float32)
W.add("sae.bin", "sae.dec", qd, np.int8)
W.add("sae.bin", "sae.dec.scales", sd, np.float32)
W.save({"layer": LAYER, "hidden": cfg.hidden_size, "inter": cfg.intermediate_size, "heads": cfg.num_attention_heads,
        "kvHeads": cfg.num_key_value_heads, "headDim": cfg.head_dim, "layers": cfg.num_hidden_layers, "vocab": cfg.vocab_size,
        "layerTypes": cfg.layer_types, "maxPos": MAXPOS,
        "queryScalar": cfg.query_pre_attn_scalar, "eps": cfg.rms_norm_eps, "concepts": int(qe.shape[0])})
print("model:", {k: f"{len(v) / 1e6:.1f} MB" for k, v in W.files.items()})

# ---- tokenizer: vocab strings, merges as id pairs (rank order), added tokens matched before BPE
tj = json.load(open(hf_hub_download(MODEL, "tokenizer.json")))
vocab = tj["model"]["vocab"]
ids = [None] * len(vocab)
for k, v in vocab.items():
    ids[v] = k
merges = np.array([[vocab[a], vocab[b]] for a, b in (m if isinstance(m, list) else m.split(" ", 1) for m in tj["model"]["merges"])], np.int32)
open(f"{ROOT}/data/merges.bin", "wb").write(merges.tobytes())
json.dump({"vocab": ids, "added": [{"id": a["id"], "content": a["content"]} for a in tj["added_tokens"]],
           "bos": tok.bos_token_id, "eos": tok.eos_token_id}, open(f"{ROOT}/data/vocab.json", "w"), ensure_ascii=False, separators=(",", ":"))

# ---- concepts: stats, steering strengths, example snippets
best = ex.get_tensor("activations").max(1).values
freq = ex.get_tensor("feature_frequencies").float()
F = best.shape[0]
stats = np.zeros((4, F), np.float32)
stats[0], stats[1] = best.numpy(), freq.numpy()
for j, v in table.items():
    stats[2, j], stats[3, j] = v["s"], v["lift"]
open(f"{ROOT}/data/stats.bin", "wb").write(stats.tobytes())                  # best, freq, strength (x best, 0 = dead), lift
acts, seqs, poss = ex.get_tensor("activations"), ex.get_tensor("seq_ids"), ex.get_tensor("positions")
toks = ex.get_slice("tokens")
snips = []
for j in range(F):
    out, seen = [], set()
    if stats[2, j] > 0:
        for k in acts[j].argsort(descending=True).tolist():
            sq, p, a = int(seqs[j, k]), int(poss[j, k]), float(acts[j, k])
            if a <= 0 or len(out) >= 4:
                break
            if sq in seen:
                continue
            seen.add(sq)
            row = toks[sq:sq + 1, :][0].tolist()
            out.append([tok.decode(row[max(1, p - 14):p]), tok.decode(row[p:p + 1]), tok.decode(row[p + 1:p + 4]), round(a / best[j].item(), 2)])
    snips.append(out)
    if j % 2000 == 0:
        print("snippets", j, flush=True)
json.dump(snips, open(f"{ROOT}/data/examples.json", "w"), ensure_ascii=False, separators=(",", ":"))

# ---- references, from the dequantized model
cap = {}
model.model.layers[LAYER].register_forward_hook(lambda m, i, o: cap.update(h=(o[0] if isinstance(o, tuple) else o).clone()))
steer = {"vec": None}


def hook(m, i, o):
    if steer["vec"] is None:
        return
    h = (o[0] if isinstance(o, tuple) else o).clone()
    if h.shape[1] > 1:
        h[:, 1:] += steer["vec"]
    else:
        h += steer["vec"]
    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h


model.model.layers[LAYER].register_forward_hook(hook)
enc_b, thr = sae["b_enc"], sae["threshold"]
TOK_TESTS = ["Hello world", "  two  spaces and a\ttab\n\nnew paragraph", "naïve café — “quotes” 🐋 日本語",
             "def f(x):\n    return x**2  # square", "<start_of_turn>user\nhi<end_of_turn>", "", " leading space", "émoji 👩‍🚀 and <unused3>"]
texts = json.load(open(sys.argv[2]))
ref = {"tokenize": [{"text": t, "ids": tok(t).input_ids} for t in TOK_TESTS + list(texts.values())], "read": [], "generate": []}
for name in ["whales", "recipe"]:
    t = texts[name]
    ids = tok(t, return_tensors="pt").input_ids
    lg = model(ids).logits[0, -1]
    h = cap["h"][0]
    pre = h @ dqe.T + enc_b
    z = pre * (pre > thr) / best.clamp(min=1e-6)
    ok = (freq > 1e-6) & (freq < 1e-2)
    spread = torch.where(ok, z[1:].mean(0), torch.zeros(F))
    top = spread.topk(10)
    tl = lg.topk(10)
    ref["read"].append({"text": t, "ids": ids[0].tolist(), "resid": h[[1, -1]].tolist(), "spread": [top.indices.tolist(), top.values.tolist()],
                        "logits": [tl.indices.tolist(), tl.values.tolist()]})
# greedy, unsteered and steered by the whales passage's top concept at 0.8 x its table strength
j = ref["read"][0]["spread"][0][0]
vec = 0.8 * table[j]["s"] * best[j] * dqd[j]
for v in [None, vec]:
    steer["vec"] = v
    ids = tok("I went outside and", return_tensors="pt").input_ids
    out, margins = [], []
    for _ in range(30):
        lg = model(ids).logits[0, -1]
        t2 = lg.topk(2)
        out.append(t2.indices[0].item())
        margins.append((t2.values[0] - t2.values[1]).item())
        ids = torch.cat([ids, t2.indices[:1][None]], 1)
    ref["generate"].append({"prompt": "I went outside and", "concept": j if v is not None else None, "scale": 0.8, "ids": out, "margins": margins})
    print(("steered:  " if v is not None else "unsteered:"), tok.decode(out))
steer["vec"] = None
json.dump(ref, open(f"{ROOT}/tools/test/ref.json", "w"))
print("EXPORT DONE")
