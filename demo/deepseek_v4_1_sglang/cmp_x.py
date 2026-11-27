"""Compare the chain's per-sublayer OUTPUT (x_out, before the mHC post) with SGLang's `_apost_x` / `_mpost_x`.

cmp_sgl.py compares the residuals (after the post). The post mixes sum_c pre_c R with post x out, and for the attention
sublayer `post` can be tiny (0.1362 on stream 3 at L0), so a residual match can hide an output error and an output
match can coexist with a residual error. This splits the two:

  python cmp_x.py --dump /tmp/chain_b2 --sgl /tmp/dec_coef8/layers/step1
"""
import argparse
import glob
import os
import re

import numpy as np

HID = 5120


def chain_bf16(path):
    """full_decode writes bf16 rows as raw uint16 (it views int16 to keep the bits)."""
    a = np.fromfile(path, dtype=np.uint16)
    return ((a.astype(np.uint32) << 16).view(np.float32)).astype(np.float64)


def live(v, n=HID):
    return v.reshape(-1, n)[0] if v.size > n else v


def rr(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-30))


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--sgl", required=True)
    a = ap.parse_args()
    print(f"{'L':>3} {'sub':>5} {'ref rms':>9} {'chain':>9} {'relRMS':>7} {'cos':>8}  ref")
    for L in range(40):
        for sub, tag in (("attn", "apost_x"), ("moe", "mpost_x")):
            pf = os.path.join(a.sgl, f"L{L:02d}_{tag}.f32")
            pc = os.path.join(a.dump, f"{sub}{L}_x_out.bin")
            if not (os.path.exists(pf) and os.path.exists(pc)):
                continue
            s = np.fromfile(pf, dtype=np.float32).astype(np.float64)
            c = live(chain_bf16(pc))
            print(f"{L:3d} {sub:>5} {np.sqrt((s ** 2).mean()):9.5f} {np.sqrt((c ** 2).mean()):9.5f} "
                  f"{rr(c, s):7.3f} {cos(c, s):+8.4f}  {tag}")


if __name__ == "__main__":
    main()
