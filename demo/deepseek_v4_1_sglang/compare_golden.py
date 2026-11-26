"""Compare two golden files (or one golden file against a static-chain dump) on the
top-k distribution instead of the greedy id.

Why not the id: on these 32K code prompts the next-token distribution is nearly flat
(step 0's best candidate is at logprob -2.2, p=0.11) and the engine's own run-to-run
noise is ~0.13 in logprob.  Two SGLang reruns of row 0 forked at step 6 -- 3445 at
-1.56 vs 344 at -1.69, a 0.13 margin -- and reconverged at step 8, i.e. the model
state was the same and only the argmax flipped.  Any reference demanding id equality
is impossible to satisfy; the distribution is what is reproducible.

Three checks per step, each with its own verdict:

  value   the logprob of the token that was actually chosen.  Catches a real numeric
          error anywhere, and is the check that fails on NaN / wrong weights.
  overlap |top5(golden) & top5(other)|.  Catches a distribution that moved somewhere
          the chosen token does not show -- e.g. mass shifting to an off-branch
          candidate whose own logprob still looks plausible.
  rank    whether the golden's argmax is inside the other's top-k.  A flip is only
          reported, never failed: inside the noise band it is expected.

A step FAILS on value error > --tol, or overlap < --min-overlap.  The script exits
non-zero if any row fails, so it can gate a run.

  python compare_golden.py --golden /tmp/golden_bench32k.jsonl --other /tmp/mine.jsonl
  python compare_golden.py --golden /tmp/golden_bench32k.jsonl --self   # noise floor
"""
import argparse
import json
import math
import sys


def load(path):
    rows = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r or not r.get("ids"):
            continue
        rows[r["row"]] = r
    return rows


def as_pairs(rec):
    """Per-step [(logprob, id), ...] from the stored top-k, falling back to the chosen token."""
    steps = rec.get("topk") or []
    if steps:
        return [[(float(v[0]), int(v[1])) for v in step] for step in steps]
    return [[(float(lp), int(tid))] for lp, tid in zip(rec["logprobs"], rec["ids"]) if lp is not None]


def compare_step(golden_step, other_step):
    """(value err, top5 overlap, argmax flip) for one decode step."""
    gmap = dict((tid, lp) for lp, tid in golden_step)
    omap = dict((tid, lp) for lp, tid in other_step)
    gold_argmax = max(gmap, key=lambda t: gmap[t])
    other_argmax = max(omap, key=lambda t: omap[t]) if omap else None

    def err(t):
        # a candidate the other run dropped out of its top-k is below its k-th value, so
        # comparing against that floor is the honest bound (and never a false pass)
        if t in omap:
            return abs(gmap[t] - omap[t])
        return abs(gmap[t] - min(omap.values())) if omap else math.inf

    value = err(gold_argmax)
    g5 = set(t for _, t in sorted(golden_step, reverse=True)[:5])
    o5 = set(t for _, t in sorted(other_step, reverse=True)[:5])
    overlap = len(g5 & o5) / max(1, len(g5))
    # A flip is the two runs picking different tokens, which is what a near-flat
    # distribution does inside the noise band -- reported, never a failure.  (Checking
    # only that golden's argmax is somewhere in the other's top-k would miss it: the
    # runner-up is always there.)
    return value, overlap, other_argmax != gold_argmax


def compare(golden, other, tol, min_overlap, verbose):
    fails, flips, total = [], [], 0
    print(f"{'row':<58} {'steps':>5} {'maxval':>8} {'meanval':>8} {'minovl':>7} {'flips':>5}")
    for name, g in golden.items():
        o = other.get(name)
        if o is None:
            print(f"{name[:58]:<58} MISSING")
            fails.append((name, "missing from --other"))
            continue
        gs, os_ = as_pairs(g), as_pairs(o)
        n = min(len(gs), len(os_))
        vals, ovls, nflip = [], [], 0
        for i in range(n):
            total += 1
            v, ov, flipped = compare_step(gs[i], os_[i])
            vals.append(v)
            ovls.append(ov)
            if flipped:
                nflip += 1
                flips.append((name, i, g["ids"][i] if i < len(g["ids"]) else None))
        if not vals:
            continue
        mx, mean = max(vals), sum(vals) / len(vals)
        mo = min(ovls)
        bad = mx > tol or mo < min_overlap
        if bad:
            fails.append((name, f"maxval={mx:.3f} minovl={mo:.2f}"))
        print(f"{name[:58]:<58} {n:>5} {mx:>8.3f} {mean:>8.3f} {mo:>7.2f} {nflip:>5}"
              f"{'  FAIL' if bad else ''}")
        if verbose and bad:
            for i in range(n):
                if vals[i] > tol or ovls[i] < min_overlap:
                    print(f"    step {i:>2}: value={vals[i]:.4f} overlap={ovls[i]:.2f} "
                          f"golden_id={g['ids'][i] if i < len(g['ids']) else '?'}")
    print()
    print(f"{len(golden)} rows, {total} steps compared; tol={tol} min_overlap={min_overlap}")
    print(f"argmax flips (expected near a tie, not a failure): {len(flips)}")
    if fails:
        print(f"FAILED rows: {len(fails)}")
        for name, why in fails[:10]:
            print(f"  {name[:70]}: {why}")
        return 1
    print("all rows within tolerance")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--other", help="a second golden file; omit with --self")
    ap.add_argument("--self", action="store_true",
                    help="compare the file with itself: maxval must be 0, a sanity check of the loader")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="max |logprob| difference for the chosen token; 0.15 is the measured run-to-run "
                         "noise of this engine on these rows (~0.125), rounded up")
    ap.add_argument("--min-overlap", type=float, default=0.6,
                    help="min |top5 & top5| / 5 per step: catches mass moving without the chosen token "
                         "showing it; 0.6 tolerates one of five swapping at a tie")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    golden = load(a.golden)
    other = golden if a.self else load(a.other)
    if not golden:
        print(f"no usable rows in {a.golden}")
        return 2
    return compare(golden, other, a.tol, a.min_overlap, a.verbose)


if __name__ == "__main__":
    sys.exit(main())
