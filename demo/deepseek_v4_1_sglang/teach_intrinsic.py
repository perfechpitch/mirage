"""Each sublayer's OWN error, with the reference residual fed in (DSV41_TEACH=1 DSV41_TEACH_FROM=0 --dump-dir D).

Why this is the right instrument: in the free chain a sublayer is graded on whatever the layers above it happened to
produce, so its score mixes "this kernel is wrong" with "its input had drifted". Teacher forcing removes the second
term: every sublayer reads SGLang's own residual, so the number below IS the kernel's own error.

The mapping (align3.py:57 -- the chain is mpost(L) --attn L--> apost(L) --moe L--> mpost(L+1)):
    attn{L}_teach_out  vs  L{L:02d}_apost
    moe{L}_teach_out   vs  L{L+1:02d}_mpost        (L = 39 has no L40 dump: the head's input, see --head)

  teach_intrinsic.py --dump /tmp/teach_dump [--golden /tmp/dec_coef8/layers/step1] [--a /tmp/dec_coef7/layers/step1]

The reference's own A-vs-B column is printed beside each row for scale: an intrinsic error at or below it is
indistinguishable from the reference's own spread, so it cannot be what to fix.
"""
import argparse
import os

import numpy as np

HC, HID = 4, 5120


def rr(x, y):
    return float(np.linalg.norm(np.asarray(x, np.float64) - np.asarray(y, np.float64))
                 / max(np.linalg.norm(np.asarray(y, np.float64)), 1e-30))


def cos(x, y):
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))


def load_golden(d, tag, key):
    f = os.path.join(d, f"L{tag:02d}_{key}.f32")
    if not os.path.exists(f):
        return None
    v = np.fromfile(f, dtype=np.float32)
    return v.reshape(-1, HID)[-HC:] if v.size == HC * HID else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/teach_dump")
    ap.add_argument("--golden", default="/tmp/dec_coef8/layers/step1")
    ap.add_argument("--a", default="/tmp/dec_coef7/layers/step1")
    ap.add_argument("--layers", type=int, default=40)
    a = ap.parse_args()
    print(f"{'sub':<8} {'intrinsic relRMS':>17} {'cos':>10} | {'ref A-B relRMS':>15} | per-stream relRMS")
    rows = []
    for L in range(a.layers):
        for sub in ("attn", "moe"):
            f = os.path.join(a.dump, f"{sub}{L}_teach_out.f32")
            if not os.path.exists(f):
                continue
            c = np.fromfile(f, dtype=np.float32).reshape(-1)
            if c.size != HC * HID:
                print(f"{sub}{L:<6} unexpected size {c.size}")
                continue
            c = c.reshape(HC, HID)
            tag, key = (L, "apost") if sub == "attn" else (L + 1, "mpost")
            g = load_golden(a.golden, tag, key)
            if g is None:
                print(f"{sub}{L:<6} {'-- (no L%d_%s)' % (tag, key):>17}")
                continue
            r, cs = rr(c, g), cos(c, g)
            ga = load_golden(a.a, tag, key)
            ab = rr(ga, g) if ga is not None else float("nan")
            per = " ".join(f"{rr(c[s], g[s]):8.1e}" for s in range(HC))
            rows.append((sub + str(L), r, ab))
            print(f"{sub}{L:<6} {r:>17.5f} {cs:>10.6f} | {ab:>15.5f} | {per}")
    # the summary that matters: is there a sublayer whose intrinsic error is above the reference's own spread?
    print()
    if rows:
        worst = sorted(rows, key=lambda t: -t[1])[:8]
        print("worst 8 by intrinsic error (sub, intrinsic, ref A-B):")
        for n, r, ab in worst:
            flag = "  <-- ABOVE the reference's spread" if r > max(ab, 0) * 1.5 and r > 0.01 else ""
            print(f"  {n:<8} {r:8.5f}   {ab:8.5f}{flag}")
        med = float(np.median([r for _, r, _ in rows]))
        print(f"\nmedian intrinsic {med:.5f}   mean {np.mean([r for _, r, _ in rows]):.5f}")


if __name__ == "__main__":
    main()
