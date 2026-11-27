"""The reference's own launch-to-launch spread, per sublayer, per stream -- the only honest yardstick past L2.

SGLang is bit-repeatable at L0 and L1 and NOT past L2: four independent launches (dec_coef2/6/7/8, same checkpoint, same
row, same settings) agree bit for bit through L1 and then differ in ~27% of the values at L2, with the difference sitting
inside the attention of L2 (`L02_st`/`L02_apost`/`L02_apost_x` move, `L02_in`/`L02_mpost`/`L02_ct_*` stay identical).
So no implementation can be bit-exact with "SGLang" at L2+ -- there is no single SGLang answer. What a correct kernel must
do is land inside that band.

This prints, per layer and per quantity, the spread across the launches (max pairwise relRMS and the max |d|), which
band cmp40.py then judges our chain against.

  ref_spread.py [--dirs dec_coef2,dec_coef6,dec_coef7,dec_coef8] [--layers 0..39] [--save /tmp/ref_spread.json]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np


def load(d, f):
    p = os.path.join("/tmp", d, "layers", "step1", f)
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="dec_coef2,dec_coef6,dec_coef7,dec_coef8")
    ap.add_argument("--layers", default=None, help="a,b or a..b; default all")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    runs = a.dirs.split(",")
    if a.layers:
        if ".." in a.layers:
            lo, hi = (int(x) for x in a.layers.split(".."))
            layers = list(range(lo, hi + 1))
        else:
            layers = [int(x) for x in a.layers.split(",")]
    else:
        layers = list(range(40))

    out = {}
    print(f"{'tensor':<18} {'n':>3}  {'spread relRMS':>13}  {'spread max|d|':>13}  {'bit-identical':>13}  verdict")
    for L in layers:
        for suf in ("_apost", "_apost_x", ""):
            f = f"L{L:02d}{suf}.f32"
            vs = [(r, load(r, f)) for r in runs]
            vs = [(r, v) for r, v in vs if v is not None]
            if len(vs) < 2:
                continue
            rms, mx, alleq = 0.0, 0.0, True
            n = vs[0][1].size
            for (ra, xa), (rb, xb) in itertools.combinations(vs, 2):
                if not np.array_equal(xa.view(np.int32), xb.view(np.int32)):
                    alleq = False
                d = xa.astype(np.float64) - xb.astype(np.float64)
                den = np.sqrt((xa.astype(np.float64) ** 2).mean()) + 1e-30
                rms = max(rms, float(np.sqrt((d ** 2).mean()) / den))
                mx = max(mx, float(np.abs(d).max()))
            out.setdefault(str(L), {})[suf or "x"] = {"relRMS": rms, "maxabs": mx, "bitexact": alleq, "n": n}
            if L < 4 or L % 8 == 0 or L == 39:
                print(f"L{L:02d}{suf or '':<12} {len(vs):>3}  {rms:>13.4e}  {mx:>13.4e}  {str(alleq):>13}  "
                      f"{'DETERMINISTIC' if alleq else 'nondeterministic'}")
    if a.save:
        json.dump(out, open(a.save, "w"), indent=1)
        print(f"\nwritten {a.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
