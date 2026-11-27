"""The 40-layer verdict: the chain's per-sublayer residual against SGLang, next to SGLang's own spread.

Two things have to be true for a mismatch to mean anything, and this prints both:

  * the pairing has to be the right quantity. `attn{L}_streams_out` is the 4-stream residual after layer L's attention
    post, so it is `L{L:02d}_apost`. `moe{L}_streams_out` is the one after the MoE post, which SGLang names by the layer
    it FEEDS: `L{L+1:02d}_mpost`. (Layer 0 is the exception -- its MoE output is the engram gate's input at layer 1, so
    it is `L01_eg_x`.) `L{L}_in` is NOT either of them: it is SGLang's `pre`, the mixed residual, and comparing against
    it shows a fake 40-60% at every layer. `x_out` is the normed input of the NEXT sublayer, so it has no counterpart
    in this dump at all.
  * the bar has to be SGLang's own repeatability. `ref_spread.py` ran the same row through SGLang several times; from
    layer 2 on the `full_r2` attention's reducer is not bit-stable, so two of SGLang's own runs differ by ~1e-2 on
    `apost`. A chain value inside that band is as correct as anything SGLang can tell us.

  layerwise.py [--ours /tmp/chain_ok] [--ref /tmp/dec_coef8] [--spread /tmp/ref_spread.json] [--layers 0-39]
"""
import argparse
import json
import os
import sys

import numpy as np

T = 8
HID, HC = 5120, 4


def ours(where, name, width):
    p = os.path.join(where, f"{name}.bin")
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dtype=np.int16).view(np.uint16)
    assert a.size == T * width, f"{p}: {a.size} values, expected {T}x{width}"
    m = a.reshape(T, width)
    assert (m[1:] == 0).all(), f"{p}: rows 1.. are not zero"
    return m[0].astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)


def ref(where, name):
    p = os.path.join(where, name)
    return np.fromfile(p, dtype=np.float32).astype(np.float64) if os.path.exists(p) else None


def rr(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((b ** 2).mean()) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="/tmp/chain_ok")
    ap.add_argument("--ref", default="/tmp/dec_coef8")
    ap.add_argument("--spread", default="/tmp/ref_spread.json")
    ap.add_argument("--layers", default="0-39")
    a = ap.parse_args()
    lo, _, hi = a.layers.partition("-")
    lo, hi = int(lo), int(hi or lo)
    R = os.path.join(a.ref, "layers", "step1")
    sp = json.load(open(a.spread)) if os.path.exists(a.spread) else {}

    print(f"\n  chain {a.ours}   vs SGLang {a.ref}")
    print(f"  band = SGLang's own run-to-run relRMS (ref_spread.py); 'in' means the chain is inside it\n")
    print(f"  {'L':>3}  {'attn vs _apost':>15} {'(band)':>9} {'':>4}   {'moe vs _mpost':>15} {'(band)':>9} {'':>4}")
    print("  " + "-" * 76)
    for L in range(lo, hi + 1):
        row = f"  {L:>3}  "
        o = ours(a.ours, f"attn{L}_streams_out", HC * HID)
        e = ref(R, f"L{L:02d}_apost.f32")
        d1 = rr(o, e) if o is not None and e is not None else None
        b1 = (sp.get(str(L), {}).get("_apost") or {}).get("relRMS")
        row += f"{d1:>15.2e} " if d1 is not None else f"{'--':>15} "
        row += f"{b1:>9.2e} " if b1 else f"{'0 (exact)':>9} "
        row += f"{'in' if d1 is not None and (b1 is None or d1 <= max(b1, 3e-3)) else 'OUT':>4}   "
        o = ours(a.ours, f"moe{L}_streams_out", HC * HID)
        e = ref(R, f"L{L + 1:02d}_mpost.f32") if L else ref(R, "L01_eg_x.f32")
        d2 = rr(o, e) if o is not None and e is not None else None
        b2 = (sp.get(str(L + 1), {}).get("_apost") or {}).get("relRMS")
        row += f"{d2:>15.2e} " if d2 is not None else f"{'--':>15} "
        row += f"{b2:>9.2e} " if b2 else f"{'0 (exact)':>9} "
        row += f"{'in' if d2 is not None and (b2 is None or d2 <= max(b2, 3e-3)) else 'OUT':>4}"
        print(row)
    print()


if __name__ == "__main__":
    sys.exit(main())
