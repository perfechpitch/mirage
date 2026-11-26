"""Read the tensors DSV41_STATIC_DUMPDIR wrote and say what actually differs.

In check mode with DSV41_STATIC_DUMPDIR=<dir> every compared sublayer writes both sides:
  <dir>/step1/attn04_y_static.npy, ..._sglang.npy
  <dir>/manifest.jsonl   step, tag, shape, dtypes, relRMS per comparison
so the answer to "what is wrong at layer 4" can be worked out after the run, with no GPU.

  python dump_layers.py --dir /tmp/layerwise/dump                  # the table: one row per comparison
  python dump_layers.py --dir ... --entry attn04_y                # head/pair detail for one entry
  python dump_layers.py --dir ... --entry attn04_y --cols 0:24    # a contiguous window of the flattened row
  python dump_layers.py --dir ... --hist                          # relRMS per layer, as a crude plot

The classifications in the table are the ones that matter for a wrong-but-finite layer, because they have nothing in
common as bugs and need different fixes:
  equal      bit-identical
  scale      same argmax, every value off by one constant ratio (a missing or extra factor)
  permuted   same multiset, different order (an index/column mix-up)
  sparse      a few elements are large and the rest agree (a tail or a boundary row)
  spread      the differences are everywhere (rounding, or a different algorithm)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def entries(d):
    man = os.path.join(d, "manifest.jsonl")
    if os.path.exists(man):
        return [json.loads(l) for l in open(man) if l.strip()]
    out = []   # no manifest (an older dump): infer from the file names
    for f in sorted(glob.glob(os.path.join(d, "step*", "*_static.npy"))):
        base = f[: -len("_static.npy")]
        step = int(os.path.basename(os.path.dirname(f))[4:])
        g = np.load(f)
        out.append(dict(step=step, tag=os.path.basename(base), shape=list(g.shape), dtype_static="?", dtype_sglang="?"))
    return out


def classify(g, r):
    if g.shape != r.shape:
        return f"shape {g.shape} vs {r.shape}"
    if np.array_equal(g, r):
        return "equal"
    g, r = g.astype(np.float64), r.astype(np.float64)
    if not (np.isfinite(g).all() and np.isfinite(r).all()):
        return f"non-finite static {int((~np.isfinite(g)).sum())}/{g.size}, sglang {int((~np.isfinite(r)).sum())}/{r.size}"
    if g.size >= 4 and g.argmax() == r.argmax():
        nz = r != 0
        if nz.any():
            ratio = g[nz] / r[nz]
            if np.allclose(ratio, ratio.ravel()[0], rtol=1e-3):
                return f"scale x{ratio.ravel()[0]:.6g}"
    if g.size == r.size and np.allclose(np.sort(g.ravel()), np.sort(r.ravel()), atol=0):
        return "permuted"
    diff = np.abs(g - r)
    # A relative floor, not a median-based one: when only a few elements differ the median is 0 and any threshold
    # derived from it is ~1e-30, which reports every rounding difference as "large". 1e-6 of the tensor's scale is
    # below bf16's own spacing, so what is left over this floor is a real difference.
    floor = max(1e-6 * float(np.abs(r).max()), 1e-30)
    nbig = int((diff > floor).sum())
    frac = nbig / diff.size
    return (f"sparse {frac * 100:.2f}% ({nbig}/{diff.size}) over {floor:.3g}" if 0 < frac < 0.05
            else "spread")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--entry", default=None, help="a tag, e.g. attn04_y")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--cols", default="0:8", help="flattened index range to print, 'a:b' (default 0:8)")
    ap.add_argument("--hist", action="store_true", help="relRMS per (step, tag) as bars")
    a = ap.parse_args()
    ents = entries(a.dir)
    if not ents:
        print(f"no entries under {a.dir}: was DSV41_STATIC_DUMPDIR set on the run, and did it reach check mode?",
              file=sys.stderr)
        return 2

    if a.hist:
        # nan is a recorded value here (a non-finite tensor has no relRMS): it gets a marker, not a bar length.
        finite = [e["relrms"] for e in ents if e.get("relrms") is not None and np.isfinite(e["relrms"])]
        top = (max(finite) if finite else 1.0) or 1.0   # all-zero: one column, rather than divide by zero
        for e in ents:
            v = e.get("relrms")
            if v is None:
                continue
            if not np.isfinite(v):
                print(f"step {e['step']:>2} {e['tag']:<18} {'!':<40} non-finite")
                continue
            print(f"step {e['step']:>2} {e['tag']:<18} {'#' * max(1, int(40 * v / top)):<40} {v:.3e}")
        return 0

    if a.entry:
        want = [e for e in ents if e["tag"] == a.entry and (a.step is None or e["step"] == a.step)]
        if not want:
            print(f"no entry {a.entry!r}; tags here: " + " ".join(sorted({e['tag'] for e in ents}))[:500], file=sys.stderr)
            return 2
        parts = a.cols.split(":")
        lo = int(parts[0]) if parts[0] else 0
        hi = int(parts[1]) if len(parts) > 1 and parts[1] else None
        for e in want:
            d = os.path.join(a.dir, f"step{e['step']}")
            g = np.load(os.path.join(d, f"{e['tag']}_static.npy")).ravel()
            r = np.load(os.path.join(d, f"{e['tag']}_sglang.npy")).ravel()
            sl = slice(lo, hi)
            print(f"step {e['step']} {e['tag']}  shape {e['shape']}  {e['dtype_static']} vs {e['dtype_sglang']}"
                  f"  relRMS {e['relrms']:.3e}  -> {classify(g.reshape(e['shape']), r.reshape(e['shape']))}")
            print(f"  cols [{lo}:{hi if hi is not None else g.size}]")
            print("  static " + " ".join(f"{x:+.5f}" for x in g[sl]))
            print("  sglang " + " ".join(f"{x:+.5f}" for x in r[sl]))
            print("  diff   " + " ".join(f"{x:+.5f}" for x in (g[sl] - r[sl])))
        return 0

    print(f"{'step':>4} {'what':<20} {'shape':<18} {'relRMS':>10}  classification")
    for e in ents:
        d = os.path.join(a.dir, f"step{e['step']}")
        g = np.load(os.path.join(d, f"{e['tag']}_static.npy"))
        r = np.load(os.path.join(d, f"{e['tag']}_sglang.npy"))
        v = e.get("relrms")
        print(f"{e['step']:>4} {e['tag']:<20} {str(tuple(e['shape'])):<18} "
              f"{'-' if v is None else f'{v:.2e}':>10}  {classify(g, r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
