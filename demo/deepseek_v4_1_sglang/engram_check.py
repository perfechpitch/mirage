"""The Engram gate: does our `R''` equal SGLang's, given the same n-gram embedding rows?

The Engram sits between a MoE and the next layer's attention, and its embedding rows are an INPUT (SGLang does the table
lookup), so this is the one sublayer where the two sides read exactly the same bytes:

    L01_eg_emb / L14_eg_emb   the gathered rows (what the chain copies into `eg_emb`)
    L01_eg_x                 the residual the gate MIXES (== the MoE's output, R')
    L01_eg_y                 R'' -- what the gate produced, and what L1's attention reads (SGLang's `in` there)

Our side: the MoE kernel of the hosting layer (0 and 13) writes `streams_eg`, so `moe0_streams_eg` / `moe13_streams_eg`.

Why the comparison is not just "same numbers": the gate's own math is `y = sigmoid(q*k / sqrt(d)) * kv @ emb` style, and
a wrong scale on q*k, a wrong block-scale unpack, or reading the wrong gathered row all show up as a uniform ratio on
`R'' - R'` rather than as noise. So the ratio `(ours - R') / (ref - R')` is printed too: 1.0 means the gate's contribution
matches in both size and direction.

    engram_check.py [--ours /tmp/chain_m10] [--ref /tmp/dec_coef8] [--steps 1,2,3]
"""
import argparse
import os
import sys

import numpy as np

T = 8
HID, HC = 5120, 4
# hosting MoE layer -> the Engram's layer (the gate in MoE L feeds layer L+1's attention)
HOSTS = {0: 1, 13: 14}


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


def step_of(ours_dir, s):
    return ours_dir if s == 1 else os.path.join(ours_dir, f"step{s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="/tmp/chain_m10")
    ap.add_argument("--ref", default="/tmp/dec_coef8")
    ap.add_argument("--steps", default="1,2,3")
    a = ap.parse_args()
    steps = [int(x) for x in a.steps.split(",")]

    print(f"\n  chain {a.ours}   vs SGLang {a.ref}")
    print(f"\n  {'step':>4} {'engram':>7}  {'Rin rms':>9} {'Rout rms':>9}  "
          f"{'eg_y ours vs ref':>17} {'cos':>10}  {'gate ratio':>10}  gate ratio per stream")
    print("  " + "-" * 104)
    for s in steps:
        R = os.path.join(a.ref, "layers", f"step{s}")
        O = step_of(a.ours, s)
        if not os.path.isdir(R) or not os.path.isdir(O):
            continue
        for host, L in HOSTS.items():
            o = ours(O, f"moe{host}_streams_eg", HC * HID)
            e = ref(R, f"L{L:02d}_eg_y.f32")
            if o is None or e is None:
                continue
            xin = ref(R, f"L{L:02d}_eg_x.f32")          # R' -- the residual the gate mixed
            d_o = (o - xin).reshape(-1)                  # our gate's contribution
            d_r = (e - xin).reshape(-1)
            ratio = float(np.dot(d_o, d_r) / (np.dot(d_r, d_r) + 1e-30))
            cos = float(o @ e / (np.linalg.norm(o) * np.linalg.norm(e) + 1e-30))
            per = [float(np.dot(d_o[i * HID:(i + 1) * HID], d_r[i * HID:(i + 1) * HID])
                         / (np.dot(d_r[i * HID:(i + 1) * HID], d_r[i * HID:(i + 1) * HID]) + 1e-30))
                   for i in range(HC)]
            # a stream the gate does not write has a ~0 delta on the reference side, so its ratio is a 0/0 artifact:
            # print the DELTA's own relRMS instead, which is the meaningful number there (verified: the gate writes
            # streams 0 and 1 only, leaving 2 and 3 at 1e-6 and 0)
            drel = [rr(d_o[i * HID:(i + 1) * HID], d_r[i * HID:(i + 1) * HID]) if
                    np.sqrt((d_r[i * HID:(i + 1) * HID] ** 2).mean()) > 1e-5 else float("nan")
                    for i in range(HC)]
            cells = " ".join(f"{v:6.4f}" if np.isfinite(v) else "  ----" for v in per)
            cells += "   delta relRMS " + " ".join(f"{v:7.4f}" if np.isfinite(v) else "   ----" for v in drel)
            print(f"  {s:>4} {('L%d' % L):>7}  {np.sqrt((xin ** 2).mean()):>9.4f} {np.sqrt((o ** 2).mean()):>9.4f}  "
                  f"{rr(o, e):>17.2e} {cos:>10.7f}  {ratio:>10.4f}  {cells}")

    # the gate's input rows, which are supposed to be byte-identical (SGLang does the table lookup)
    print("\n  the gathered rows (`eg_emb`) are an input on both sides -- they must match byte for byte:")
    for s in steps:
        R = os.path.join(a.ref, "layers", f"step{s}")
        for host, L in HOSTS.items():
            e = ref(R, f"L{L:02d}_eg_emb.f32")
            if e is None:
                continue
            print(f"    step {s} L{L:02d}: eg_emb [{e.size}]  rms {np.sqrt((e ** 2).mean()):.5f}  "
                  f"nonzero {int((e != 0).sum())}/{e.size}")
    print()


if __name__ == "__main__":
    sys.exit(main())
