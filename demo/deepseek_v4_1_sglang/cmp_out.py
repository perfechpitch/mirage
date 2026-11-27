"""Recover each sublayer's own output from the chain's residuals and compare it with SGLang's.

The chain dumps, per sublayer, the residual *after* the mHC post (`streams_out`) and the coefficients that post used
(`_coef.bin`: pre|post|comb, as the kernel read them). R' = comb^T R + post * x, so x = (R' - comb^T R) / post per
stream, weighted by |post| -- no rebuild, and it isolates "this sublayer's output is wrong" from "the mix is wrong".

SGLang's side is `L{L}_apost_x` (attention) and `L{L+1}_mpost_x` (MoE: the reference's L{L}_mpost is the residual
before layer L, so layer L's MoE output is written under L{L+1}, see cmp_sgl.py).

  python cmp_out.py --dump /tmp/chain_b2 --sgl /tmp/dec_coef8/layers/step1
"""
import argparse
import os

import numpy as np

HID, HC = 5120, 4
S = HID * HC


def chain_streams(path):
    a = np.fromfile(path, dtype=np.uint16)
    return ((a.astype(np.uint32) << 16).view(np.float32)).astype(np.float64).reshape(-1, S)[0].reshape(HC, HID)


def chain_coef(path):
    a = np.fromfile(path, dtype=np.float32).reshape(8, 32)[0]
    return a[0:4], a[4:8], a[8:24].reshape(4, 4)


def rr(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-30))


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def recover(R, Rp, comb, post):
    """x from R' = comb^T R + post x, the streams weighted by |post| (a stream with post 0 says nothing)."""
    d = Rp - comb.T @ R
    w = np.abs(post)
    if w.sum() == 0:
        return None
    return (w[:, None] * d).sum(0) / (w ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--sgl", required=True)
    a = ap.parse_args()
    # layer 0's input residual: the embedding in every stream (SGLang's hc_broadcast), i.e. the reference's L00_in.
    # Taken from the reference because the chain's S0 is the same bits and the point is to isolate the sublayers.
    p0 = os.path.join(a.sgl, "L00_in.f32")
    prev = np.fromfile(p0, dtype=np.float32).astype(np.float64).reshape(HC, HID) if os.path.exists(p0) else None
    print(f"{'L':>3} {'sub':>5} {'ref rms':>9} {'chain':>9} {'relRMS':>7} {'cos':>8}  ref")
    for L in range(40):
        for sub, tag, nxt in (("attn", "apost_x", None), ("moe", "mpost_x", 1)):
            pd = os.path.join(a.dump, f"{sub}{L}_streams_out.bin")
            pc = os.path.join(a.dump, f"{sub}{L}_coef.bin")
            if not (os.path.exists(pd) and os.path.exists(pc) and prev is not None):
                continue
            R = prev
            Rp = chain_streams(pd)
            # the attention sublayer of an Engram-host layer reads R'' (its host MoE's streams_eg), and the post that
            # produces this residual mixed R'': prev is updated below from the right file
            _pre, post, comb = chain_coef(pc)
            x = recover(R, Rp, comb, post)
            pf = os.path.join(a.sgl, f"L{L + (nxt or 0):02d}_{tag}.f32")
            if x is None or not os.path.exists(pf):
                continue
            s = np.fromfile(pf, dtype=np.float32).astype(np.float64)
            print(f"{L:3d} {sub:>5} {np.sqrt((s ** 2).mean()):9.5f} {np.sqrt((x ** 2).mean()):9.5f} "
                  f"{rr(x, s):7.3f} {cos(x, s):+8.4f}  {tag}")
            prev = Rp
        # the next attention reads this MoE's R'' when it hosts an Engram (full_decode: streams_eg), R' otherwise
        pe = os.path.join(a.dump, f"moe{L}_streams_eg.bin")
        if os.path.exists(pe):
            prev = chain_streams(pe)


if __name__ == "__main__":
    main()
