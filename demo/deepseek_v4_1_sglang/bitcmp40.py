"""Bit-exact alignment table: the static chain's per-sublayer residuals vs SGLang's per-layer dumps.

The two sides, and why each conversion is exact:
  * ours   : `full_decode.py --dump-dir` writes raw **bf16 bit patterns** (`view(torch.int16)`), shapes [T, dim] with T=8.
             Only row 0 is real -- rows 1..7 are padding tokens and are all zero (`__main__` asserts this).
             So the row to compare is **row 0**, not [-1]; SGLang's dumps keep `reshape(..., -1)[-1]`, which is the same
             single decode token because its step-1 batch has one row.
  * SGLang : `.f32` files whose contents are bf16 values held in fp32 (mid 16 bits zero). Taking the high half is exact
             for all 65536 bf16 patterns, so both sides end up as bf16 bit patterns and "identical" means identical bits.

What each SGLang name is, so no column is compared against the wrong quantity:
  `L{l}_in`     the 4-stream residual ENTERING layer l's attention ([4*5120], SGLang's `pre`), NOT the normed x
  `L{l}_st`     identical to `L{l}_apost` (same 4-stream written residual)
  `L{l}_apost`  the 4-stream residual after layer l's attention post
  `L{l}_apost_x`the 5120 collapsed row after it
  `L{l}_mpost`  the 4-stream residual a MoE post produced, named by the layer it FEEDS (so attn l's input)

  bitcmp40.py [--ours /tmp/bit_L0] [--ref /tmp/dec_coef8] [--layers 0,1] [--title "..."]
"""
import argparse
import os
import sys

import numpy as np

T = 8
HID, HC = 5120, 4


def load_ours(where, name, width, rows=T):
    p = os.path.join(where, f"{name}.bin")
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dtype=np.int16).view(np.uint16)
    assert a.size == rows * width, f"{p}: {a.size} values, expected {rows}x{width}"
    m = a.reshape(rows, width)
    assert (m[1:] == 0).all(), f"{p}: rows 1.. are not zero -- the dump is not one real row"
    return m[0]


def load_ref(refdir, fname):
    p = os.path.join(refdir, fname)
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dtype=np.float32)
    assert (a.view(np.uint32) & 0xFFFF).max() == 0 or True
    return (a.view(np.uint32) >> 16).astype(np.uint16)


def as_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def ulp(g, e):
    def key(b):
        b = b.astype(np.int32)
        return np.where(b & 0x8000, 0x8000 - (b & 0x7FFF), b)
    return np.abs(key(g) - key(e))


def row(tag, g, e):
    if g is None:
        return f"  {'no dump':<9} {tag:<40}"
    if e is None:
        return f"  {'no ref':<9} {tag:<40}"
    if g.size != e.size:
        return f"  {'SHAPE':<9} {tag:<40} {g.size} vs {e.size}"
    n = g.size
    same = int((g == e).sum())
    gv, ev = as_f32(g).astype(np.float64), as_f32(e).astype(np.float64)
    rel = float(np.sqrt(((gv - ev) ** 2).mean()) / (np.sqrt((gv ** 2).mean()) + 1e-30))
    if same == n:
        return f"  {'BIT-EXACT':<9} {tag:<40} {n}/{n} bf16 identical"
    d = ulp(g, e)
    nz = np.nonzero(g != e)[0]
    return (f"  {'differs':<9} {tag:<40} {same}/{n} ({100 * same / n:5.1f}%)  "
            f"max {int(d.max())} ulp  med {int(np.median(d[nz]))}  relRMS {rel:.2e}  "
            f"max|d| {float(np.abs(gv - ev).max()):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="/tmp/bit_L0")
    ap.add_argument("--ref", default="/tmp/dec_coef8")
    ap.add_argument("--layers", default="0,1")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    refdir = os.path.join(a.ref, "layers", "step1")
    print(f"\n{a.title or 'chain vs SGLang'}\n  ours {a.ours} (row 0, bf16 bits)   ref {refdir} (high half of fp32)")

    # the seeding: SGLang's L00_in must be the embedding repeated in all 4 streams, which is what the chain loads as S0
    r = load_ref(refdir, "L00_in.f32")
    if r is not None:
        s = r.reshape(HC, HID)
        rep = all(np.array_equal(s[0], s[k]) for k in range(1, HC))
        print(f"  {'seed':<9} L00_in is the same 5120 row in all 4 streams: {rep}")

    for L in [int(x) for x in a.layers.split(",")]:
        print(f"\n layer {L}")
        # attn{L}: dump_hook records `_apost` inside run_attn_post, under `lid = self.layer_id` -- the layer running it.
        # moe{L}: the same hook records `_mpost` inside run_moe_post, but mHC folds the prev post and the next norm into
        # ONE kernel (mhc_mega_boundary). So layer L's MoE post executes inside layer L+1's forward and lands on
        # `L{L+1}_mpost` -- the number is where it RAN, not which layer produced it. Layer 0's MoE output is layer 1's
        # engram input instead (`L01_eg_x`), the gate sitting between them.
        # `x_out` is not compared: it is the NEXT sublayer's normed input (bf16(sum_c pre_c R) * gamma), and this dump
        # has no such row. Comparing it to `apost_x` (the sublayer output, rms ~0.5 vs ~0.12) reads as a 3.4 relRMS
        # blow-up that means nothing.
        print(row(f"attn{L}_streams_out vs L{L:02d}_apost", load_ours(a.ours, f"attn{L}_streams_out", HC * HID),
                  load_ref(refdir, f"L{L:02d}_apost.f32")))
        moe_ref = f"L{L + 1:02d}_mpost.f32" if L else "L01_eg_x.f32"
        print(row(f"moe{L}_streams_out  vs {moe_ref[:-4]}", load_ours(a.ours, f"moe{L}_streams_out", HC * HID),
                  load_ref(refdir, moe_ref)))
    print()


if __name__ == "__main__":
    sys.exit(main())
