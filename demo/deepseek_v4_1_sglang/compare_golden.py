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
          error anywhere, and is the check that fails on NaN / wrong weights.  NaN is
          checked with isfinite, not with a comparison: nan > tol is False, so a run
          whose steps went NaN used to print "all rows within tolerance" and exit 0.
  overlap |top5(golden) & top5(other)|.  Catches a distribution that moved somewhere
          the chosen token does not show -- e.g. mass shifting to an off-branch
          candidate whose own logprob still looks plausible.
  rank    whether the golden's argmax is inside the other's top-k.  A flip is only
          reported, never failed: inside the noise band it is expected.

A step FAILS on a non-finite value, value error > --tol, or overlap < --min-overlap.  A
row also FAILS if it collapsed to a single argmax across every step while golden varies
(the "sinks to token 0" failure, which the tolerance alone does not see).  The script
exits non-zero if any row fails, so it can gate a run.

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


def _argmax(step):
    """The step's argmax token (its logprob's max); a NaN logprob wins, as a broken step should."""
    return max(step, key=lambda p: p[0])[1]


def compare_step(golden_step, other_step):
    """(value err, top5 overlap, argmax flip, other's argmax) for one decode step."""
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
    return value, overlap, other_argmax != gold_argmax, other_argmax


def compare(golden, other, tol, min_overlap, verbose, as_json=False):
    fails, flips, total, summ = [], [], 0, {}
    if not as_json:
        print(f"{'row':<58} {'steps':>5} {'maxval':>8} {'meanval':>8} {'minovl':>7} {'flips':>5}")
    for name, g in golden.items():
        o = other.get(name)
        if o is None:
            if not as_json:
                print(f"{name[:58]:<58} MISSING")
            fails.append((name, "missing from --other"))
            continue
        gs, os_ = as_pairs(g), as_pairs(o)
        n = min(len(gs), len(os_))
        vals, ovls, nflip, amax = [], [], 0, []
        for i in range(n):
            total += 1
            v, ov, flipped, oa = compare_step(gs[i], os_[i])
            vals.append(v)
            ovls.append(ov)
            amax.append(oa)
            if flipped:
                nflip += 1
                flips.append((name, i, g["ids"][i] if i < len(g["ids"]) else None))
        if not vals:
            continue
        # NaN must be a failure, not a value: `max()` drops a NaN it meets second (nan > x is False), so a
        # run whose last steps are NaN printed "all rows within tolerance" and exited 0 before this.
        nbad = sum(1 for v in vals if not math.isfinite(v))
        mx = math.inf if nbad else max(vals)
        mean = math.inf if nbad else sum(vals) / len(vals)
        mo = min(ovls)
        # a collapse to one token (the "sink to 0" failure): every step the same argmax while golden varies
        uniq = len(set(t for t in amax if t is not None))
        g_uniq = len(set(_argmax(s) for s in gs if s))
        sunk = n >= 8 and uniq <= 1 and g_uniq > 3
        bad = nbad > 0 or mx > tol or mo < min_overlap or sunk
        if bad:
            why = []
            if nbad:
                why.append(f"{nbad}/{n} steps non-finite")
            if mx > tol and not nbad:
                why.append(f"maxval={mx:.3f}")
            if mo < min_overlap:
                why.append(f"minovl={mo:.2f}")
            if sunk:
                why.append(f"collapsed to one argmax ({uniq} distinct over {n} steps)")
            fails.append((name, " ".join(why)))
        mxs = "nan" if nbad else f"{mx:.3f}"
        summ[name] = dict(steps=n, maxval=None if nbad else mx, meanval=None if nbad else mean,
                          minovl=mo, flips=nflip, nonfinite=nbad,
                          id0=sum(1 for t in amax if t == 0), ok=not bad)
        if not as_json:
            print(f"{name[:58]:<58} {n:>5} {mxs:>8} {'nan' if nbad else f'{mean:.3f}':>8} {mo:>7.2f} {nflip:>5}"
              f"  id0 {sum(1 for t in amax if t == 0)}/{n}"
              f"{'  FAIL' if bad else ''}")
        if verbose and bad and not as_json:
            for i in range(n):
                if not math.isfinite(vals[i]) or vals[i] > tol or ovls[i] < min_overlap:
                    print(f"    step {i:>2}: value={vals[i]:.4f} overlap={ovls[i]:.2f} "
                          f"other_id={amax[i]} golden_id={g['ids'][i] if i < len(g['ids']) else '?'}")
    if as_json:
        print(json.dumps(dict(rows=summ, steps=total, tol=tol, min_overlap=min_overlap,
                              flips=len(flips), ok=not fails,
                              failed=[n for n, _ in fails], why=dict(fails))))
        return 0 if not fails else 1
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
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="one JSON object instead of the table: per row maxval/meanval/minovl/flips/nonfinite/"
                         "id0/ok, plus ok and why for the file. Same exit code, so accept.sh can compare the "
                         "static run's maxval against the baseline's instead of a fixed --tol")
    a = ap.parse_args()

    golden = load(a.golden)
    other = golden if a.self else load(a.other)
    if not golden:
        print(f"no usable rows in {a.golden}")
        return 2
    return compare(golden, other, a.tol, a.min_overlap, a.verbose, a.as_json)


if __name__ == "__main__":
    sys.exit(main())
