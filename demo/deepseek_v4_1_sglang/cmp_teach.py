"""DSV41_TEACH run: each sublayer's OWN residual vs SGLang's, with SGLang's residual as the input of every sublayer.

full_decode.py --dump-dir DIR with DSV41_TEACH=1 writes, per sublayer, what the launch actually wrote (captured right
after it, before the next launch overwrites the tensor -- the chain links `streams` to the previous `streams_out` by
reference):
    DIR/{name}_teach_out.f32    fp32 [20480]  the 4-stream residual this sublayer produced (row 0)
    DIR/{name}_teach_x.f32      fp32 [5120]   its x_out (the normed input of the next sublayer)

With the input exact, the residual difference IS the sublayer's own error: R' = comb^T R + post x, so the same
comb/post applied to a wrong x moves R' by post * dx. Mapping (see full_decode.teach_row):
    attn L: reads L{L}_mpost  (L00_in for L0, L{L}_eg_y for an Engram layer), SGLang's own row is L{L}_apost
    moe  L: reads L{L}_apost,                                                SGLang's own row is L{L+1}_mpost

  python cmp_teach.py --dump /tmp/chain_t --sgl /tmp/dec_coef8/layers/step1
"""
import argparse
import os

import numpy as np

HID, HC = 5120, 4
S = HID * HC


def rr(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-30))


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--sgl", required=True)
    ap.add_argument("--tol", type=float, default=0.01, help="relRMS a sublayer must stay under (1%%)")
    ap.add_argument("--json", default=None, help="also write {name: [relRMS, cos]} here")
    a = ap.parse_args()
    print(f"{'L':>3} {'sub':>5} {'chain rms':>10} {'sgl rms':>9} {'relRMS':>7} {'cos':>9}  {'ref':>6}  verdict")
    rows, bad = {}, []
    for L in range(40):
        for sub in ("attn", "moe"):
            pc = os.path.join(a.dump, f"{sub}{L}_teach_out.f32")
            tag = L + 1 if sub == "moe" else L
            key = "mpost" if sub == "moe" else "apost"
            pf = os.path.join(a.sgl, f"L{tag:02d}_{key}.f32")
            if not (os.path.exists(pc) and os.path.exists(pf)):
                continue
            c = np.fromfile(pc, dtype=np.float32).astype(np.float64)
            s = np.fromfile(pf, dtype=np.float32).astype(np.float64)
            r, k = rr(c, s), cos(c, s)
            rows[f"{sub}{L}"] = [r, k]
            ok = r < a.tol and k > 0.999
            if not ok:
                bad.append(f"L{L} {sub}")
            print(f"{L:3d} {sub:>5} {np.sqrt((c ** 2).mean()):10.5f} {np.sqrt((s ** 2).mean()):9.5f} "
                  f"{r:7.4f} {k:+9.6f}  {key:>6}  {'ok' if ok else 'BAD'}")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} sublayers within {a.tol:.1%} (relRMS) and cos > 0.999")
    if bad:
        print("outside: " + "; ".join(bad))
    if a.json:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
