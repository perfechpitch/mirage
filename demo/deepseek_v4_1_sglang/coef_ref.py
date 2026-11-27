"""The mHC triplet (pre, post, comb), computed in fp32 from the official formulas.

`model.py Block.hc_mixes`, `hc_post`/`hc_pre`, `kernel.py hc_split_sinkhorn` -- the same expressions mhc_check.py
verifies the kernel against, here as a standalone tool so a coefficient set read from a dump can be checked against
the formula on ANY residual, in particular on SGLang's residual instead of the chain's.

Why it matters: the coefficients are a pure function of the residual, so comparing two runs' coefficients mixes two
causes (a different residual, a different formula). Feeding the formula the *same* residual separates them, and
feeding it SGLang's residual yields the triplet SGLang itself used -- which its own dump does not record for the
layers whose FFN post is fused into the next layer's boundary.

  python coef_ref.py --checkpoint DIR --layer L --kind attn|moe --residual FILE.f32 [--compare NAME_coef.bin]
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

HC, H, EPS, HC_EPS, ITERS = 4, 5120, 1e-20, 1e-6, 20


def read_ckpt(ck, keys):
    idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for k in keys:
        with safe_open(os.path.join(ck, idx[k]), "pt") as sf:
            out[k] = sf.get_tensor(k).to(torch.float32).cpu()
    return out


def triplet(R, fn, base, scale, rows=1):
    """R: [rows, HC*H] fp32 residual -> pre [rows,HC], post [rows,HC], comb [rows,HC,HC] (i = source, j = dest)."""
    x = R.float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2 * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = (mixes[:, 2 * HC:].view(-1, HC, HC) * scale[2] + base[2 * HC:].view(HC, HC)).softmax(-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    return pre, post, comb


def post_apply(R, s, pre, post, comb, gamma):
    """R' and the next sublayer's x, as hc_post_combine does them (bf16 stores)."""
    Rs = R.float().view(-1, HC, H)
    Rn = (post[:, :, None] * s.float()[:, None] + torch.einsum("tij,tic->tjc", comb, Rs)).bfloat16()
    y = (pre[:, :, None] * Rn.float()).sum(1)
    xn = (y.bfloat16().float() * torch.rsqrt(y.square().mean(-1, keepdim=True) + EPS) * gamma.float()).bfloat16()
    return Rn, xn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--kind", choices=("attn", "moe"), required=True)
    ap.add_argument("--residual", required=True, help="the residual R the coefficients come from ([HC*H] or [8, HC*H] f32)")
    ap.add_argument("--compare", default=None, help="a *_coef.bin from full_decode --dump-dir ([8,32] fp32)")
    ap.add_argument("--s", default=None, help="the sublayer's output ([HC*H] f32): apply post_apply and print R'/x")
    ap.add_argument("--gamma", default=None, help="the norm weight key (default: the post's next sublayer)")
    a = ap.parse_args()

    k = "hc_attn_" if a.kind == "attn" else "hc_ffn_"
    keys = [f"layers.{a.layer}.{k}{n}" for n in ("fn", "base", "scale")]
    w = read_ckpt(a.checkpoint, keys)
    fn, base, scale = w[keys[0]], w[keys[1]], w[keys[2]]

    R = np.fromfile(a.residual, dtype=np.float32).reshape(-1, HC * H)
    pre, post, comb = triplet(torch.from_numpy(R).float(), fn, base, scale)

    print(f"layer {a.layer} {a.kind}: formula on {os.path.basename(a.residual)} (rows {R.shape[0]}, using row 0)")
    print(f"  pre  {np.round(pre[0].numpy(), 5)}")
    print(f"  post {np.round(post[0].numpy(), 5)}")
    print(f"  comb {np.round(comb[0].numpy(), 5).tolist()}")

    if a.compare:
        c = np.fromfile(a.compare, dtype=np.float32).reshape(-1)[:32]
        d = np.abs(c[:24] - np.concatenate([pre[0].numpy(), post[0].numpy(), comb[0].numpy().reshape(-1)])).max()
        print(f"  vs {os.path.basename(a.compare)}: maxdiff {d:.3e}")
        if d > 1e-4:
            print(f"    dump pre  {np.round(c[:4], 5)}")
            print(f"    dump post {np.round(c[4:8], 5)}")
            print(f"    dump comb {np.round(c[8:24].reshape(HC, HC), 5).tolist()}")

    if a.s:
        s = np.fromfile(a.s, dtype=np.float32).reshape(-1, HC * H) if os.path.getsize(a.s) != HC * H * 4 else \
            np.fromfile(a.s, dtype=np.float32).reshape(1, HC * H)
        gk = a.gamma or (f"layers.{a.layer}.ffn_norm.weight" if a.kind == "attn" else f"layers.{a.layer + 1}.attn_norm.weight")
        g = read_ckpt(a.checkpoint, [gk])[gk]
        Rn, xn = post_apply(torch.from_numpy(R).float(), torch.from_numpy(s[:1]).float(), pre, post, comb, g)
        print(f"  R' from the formula: |R'| {np.round(np.linalg.norm(Rn[0].numpy(), axis=-1), 4)}")
        print(f"  x  from the formula: |x|  {np.linalg.norm(xn[0].numpy()):.4f}")


if __name__ == "__main__":
    main()
