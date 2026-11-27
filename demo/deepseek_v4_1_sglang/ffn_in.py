"""The FFN's input row for layer L, from SGLang's own post-attention residual -- the quantity `moe_ref.py` needs.

`hc_post_combine`'s second half, per model.py `hc_pre` + the boundary's norm:

    y = sum_j pre[j] * R'[j]                      pre = the triplet's pre, from _compute_triplet on R'
    x = RMSNorm(y) * post_attention_layernorm.weight

R' is `L{L}_apost.f32` (the attention post's output) and the coefficients are `L{L}_ct_*.f32`, both written by
dump_hook.py -- so this reads SGLang's numbers and never the chain's.

  python ffn_in.py --checkpoint CK --dump DIR --layer 0 --out ffn0_x.f32
"""
import argparse

import numpy as np
import torch

from coef_ref import HC, H, read_ckpt

EPS = 1e-20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d, L = a.dump, a.layer
    R = np.fromfile(f"{d}/L{L:02d}_apost.f32", dtype=np.float32).reshape(HC, H)
    pre = np.fromfile(f"{d}/L{L:02d}_ct_pre.f32", dtype=np.float32)[:HC]
    gk = f"layers.{L}.ffn_norm.weight"
    g = read_ckpt(a.checkpoint, [gk])[gk]

    y = torch.from_numpy((pre[:, None] * R).astype(np.float32)).sum(0)
    x = (y.bfloat16().float() * torch.rsqrt(y.square().mean() + EPS) * g).bfloat16()
    print(f"layer {L} ffn input: |y| {y.norm():.4f}  pre {np.round(pre, 5).tolist()}  |x| {x.norm():.4f}")
    np.array(x.float().numpy(), dtype=np.float32).tofile(a.out)


if __name__ == "__main__":
    main()
