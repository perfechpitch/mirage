"""A routing-flip audit across every layer: where does the chain's set of top-6 experts differ from the official's, and by
how much was the 6th/7th score separated when it did.

Why only the gate: the residual gap per layer is dominated by *discrete* events. When two experts' scores are within a
bfloat16 ulp, one rounding difference changes the top-6 set, and the MoE output jumps by O(1/K) even though every kernel
is right. Numerically that shows up as a step in the per-layer relRMS table, which is exactly what the chain has at
L7/L16/L18. A layer with a large gap and NO flip is the opposite -- that is a real bug and worth chasing. So the audit
pairs the two tables: gate margin and whether the set changed.

Both inputs are built the same way, from the two sides' own dumps, with the same formula (model.py Block.forward):

    x_moe(L) = bf16( norm( pre_attn(L) . R'_attn(L) ) * ffn_norm.weight[L] )

  * official: `L{LL}_apost` (its post-attention residual) and `L{LL}_ct_pre` (the mhc pre mix it computed)
  * chain:    attn{L}_streams_out and attn{L}_coef's pre -- the kernel's own numbers

This is also the convention `moe_ref.py --x` expects, so `--save-x` writes exactly what it wants and the two tools agree.

  python moe_audit.py --checkpoint CK --layers 0-39
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

H, E, K = 5120, 384, 6
GATE_TEMP, ROUTE_SCALE = 1.0, 1.5
PRE = "/tmp/dec_coef8/layers/step1"
CHAIN = "/tmp/chain40"


def rr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-30))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


class Ck:
    def __init__(self, ck):
        self.ck = ck
        self.idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]
        self.cache = {}

    def __call__(self, n):
        if n not in self.cache:
            with safe_open(os.path.join(self.ck, self.idx[n]), "pt") as sf:
                self.cache[n] = sf.get_tensor(n)
        return self.cache[n]


def x_from(R, pre, L, ck, f32=True):
    """model.py's attn -> ffn handover. hc_pre's bf16 mix, RMSNorm(norm_eps 1e-20), then the ffn norm weight."""
    r = R.astype(np.float64)
    y = (pre.astype(np.float64)[:, None] * r).sum(0)
    g = ck(f"layers.{L}.ffn_norm.weight").float().numpy().astype(np.float64)
    x = y / np.sqrt((y ** 2).mean() + 1e-20) * g
    return torch.from_numpy(x.astype(np.float32)).bfloat16()


def gate_scores(x, w):
    logits = (x.float() @ w["gate.weight"].float().t()) / GATE_TEMP
    return torch.nn.functional.softplus(logits).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layers", default=f"0-{39}")
    ap.add_argument("--dump", default=CHAIN)
    ap.add_argument("--golden", default=PRE)
    ap.add_argument("--save-x", default=None, help="write the official x for this layer here (moe_ref.py --x)")
    ap.add_argument("--layer-for-x", type=int, default=0)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.layers.split("-"))
    ck = Ck(a.checkpoint)

    print(f"{'L':>3} {'6th':>10} {'7th':>10} {'margin':>10} {'rel.':>9} {'flip':>5}  "
          f"{'x: relRMS':>9} {'cos':>9}  overlap")
    flips = []
    for L in range(lo, min(hi, 39) + 1):
        fo = os.path.join(a.golden, f"L{L:02d}_apost.f32")
        if not os.path.exists(fo):
            continue
        R_o = np.fromfile(fo, np.float32).reshape(4, H)
        pre_o = np.fromfile(os.path.join(a.golden, f"L{L:02d}_ct_pre.f32"), np.float32).reshape(-1)[:4]
        xo = x_from(R_o, pre_o, L, ck)
        # chain side: the same two quantities from the kernel's own dumps
        st = os.path.join(a.dump, f"attn{L}_streams_out.bin")
        if not os.path.exists(st):
            continue
        R_c = torch.frombuffer(bytearray(open(st, "rb").read()), dtype=torch.bfloat16).float().reshape(-1, 4, H)[0].numpy()
        c = np.fromfile(os.path.join(a.dump, f"attn{L}_coef.bin"), np.float32).reshape(-1, 32)[0]
        xc = x_from(R_c, c[:4], L, ck)

        w = {k: ck(f"layers.{L}.ffn.{k}") for k in ("gate.weight", "gate.bias")}
        so, sc = gate_scores(xo, w), gate_scores(xc, w)
        to = (so + w["gate.bias"].float()).topk(K).indices.tolist()
        tc = (sc + w["gate.bias"].float()).topk(K).indices.tolist()
        o = (so + w["gate.bias"].float()).sort(descending=True).values
        marg = float(o[K - 1] - o[K])
        relm = marg / max(abs(float(o[K - 1])), 1e-30)
        same = set(to) == set(tc)
        if not same:
            flips.append(L)
        ov = len(set(to) & set(tc))
        print(f"{L:>3} {float(o[K-1]):>10.5f} {float(o[K]):>10.5f} {marg:>10.6f} {relm:>9.2e} "
              f"{'ok' if same else 'FLIP':>5}  {rr(xc.float().numpy(), xo.float().numpy()):>9.5f} "
              f"{cos(xc.float().numpy(), xo.float().numpy()):>+9.6f}  {ov}/{K}")
        if a.save_x and L == a.layer_for_x:
            np.array(xo.float().numpy(), np.float32).tofile(a.save_x)
            print(f"    wrote the official x for L{L} -> {a.save_x}")
    print(f"\nflipped layers: {flips if flips else 'none'}  ({len(flips)} of {len(range(lo, min(hi,39)+1))})")


if __name__ == "__main__":
    main()
