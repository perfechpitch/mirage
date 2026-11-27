"""Which residual did SGLang compute a fused layer's FFN coefficients from?

The FFN post of layers 0, 1, 14 and 20 is fused into the next layer's boundary, so the SGLang hook records no
`L{L}_ct_*` for them, and `L{L}_mpost` is captured before the pending post is applied (`L{L+1}_in` is not the same
array). What is left is `L{L+1}_in`, the residual the next layer actually reads.

With coefficients c and the mixing residual R, the post is  R'_j = post_j s + sum_i comb[i][j] R_i  for ONE shared
s, so given R' the four streams over-determine s: a wrong c or a wrong R leaves a fit residual of order |R'|,
the right pair leaves bf16 rounding. This sweeps the candidate residuals for both roles and reports the fit, which
is how the pair SGLang actually used can be identified from the dump alone.

  python coef_fit.py --layer 0 --kind moe --target L01_in
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
from safetensors import safe_open

HC, H, EPS, HC_EPS, ITERS = 4, 5120, 1e-20, 1e-6, 20


def ckpt(ck, keys):
    idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]
    out = []
    for k in keys:
        with safe_open(os.path.join(ck, idx[k]), "pt") as sf:
            out.append(sf.get_tensor(k).to(torch.float32))
    return out


def triplet(R, fn, base, scale):
    x = R.to(torch.float32).reshape(-1, HC * H)
    m = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(m[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2 * torch.sigmoid(m[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = (m[:, 2 * HC:].view(-1, HC, HC) * scale[2] + base[2 * HC:].view(HC, HC)).softmax(-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    return (pre[0].numpy().astype(np.float64), post[0].numpy().astype(np.float64),
            comb[0].detach().numpy().astype(np.float64))


def fit(post, comb, R, Rp):
    """The shared s that best explains Rp, and how badly the four streams disagree about it."""
    mix = np.stack([sum(comb[i, j] * R[i] for i in range(HC)) for j in range(HC)])
    den = float(sum(post[j] ** 2 for j in range(HC)))
    s = sum(post[j] * (Rp[j] - mix[j]) for j in range(HC)) / den
    r = float(np.sqrt(sum(np.linalg.norm(Rp[j] - (post[j] * s + mix[j])) ** 2 for j in range(HC))))
    return s, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgl", default="/tmp/dec_coef2/layers/step1")
    ap.add_argument("--checkpoint", default="/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--kind", choices=("attn", "moe"), default="moe")
    ap.add_argument("--target", default=None, help="the residual to explain (default: L{layer+1}_in)")
    a = ap.parse_args()

    L = a.layer
    tgt = a.target or (f"L{L + 1:02d}_in" if a.kind == "moe" else f"L{L:02d}_apost")
    names = [f"L{L:02d}_in", f"L{L:02d}_apost", f"L{L:02d}_st", f"L{L + 1:02d}_in", f"L{L + 1:02d}_apost",
             f"L{L:02d}_mpost", f"L{L + 1:02d}_mpost"]
    arr = {}
    for n in names:
        p = os.path.join(a.sgl, n + ".f32")
        if os.path.exists(p):
            arr[n] = np.fromfile(p, dtype=np.float32).astype(np.float64).reshape(HC, H)
    Rp = arr[tgt]

    k = "hc_attn_" if a.kind == "attn" else "hc_ffn_"
    fn, base, scale = ckpt(a.checkpoint, [f"layers.{L}.{k}{n}" for n in ("fn", "base", "scale")])

    print(f"explaining {tgt} with layer {L} {a.kind} coefficients: |{tgt}| {np.round(np.linalg.norm(Rp, axis=1), 3)}")
    print(f"{'coef-input':16s} {'mix-input':16s} {'|s|':>8s} {'fit resid':>10s}")
    rows = []
    for cn, cm in itertools.product(arr, arr):
        pre, post, comb = triplet(torch.from_numpy(arr[cn]), fn, base, scale)
        s, r = fit(post, comb, arr[cm], Rp)
        rows.append((r, cn, cm, float(np.linalg.norm(s)), post, comb))
    rows.sort()
    for r, cn, cm, sn, post, comb in rows[:8]:
        print(f"{cn:16s} {cm:16s} {sn:8.3f} {r:10.4f}")
    r, cn, cm, sn, post, comb = rows[0]
    print(f"\nbest: coefficients from {cn}, mixing {cm}")
    print(f"  post {np.round(post, 5)}   comb {np.round(comb, 4).tolist()}")
    sa, ra = fit(*triplet(torch.from_numpy(arr[cn]), fn, base, scale)[1:], arr[cm], Rp) if False else (None, None)


if __name__ == "__main__":
    main()
