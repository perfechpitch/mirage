"""Per-step view of two golden-format runs of the same row: where do they agree, where do they fork.

compare_golden.py compresses a row to maxval/meanval/minovl.  That hides the thing we need here:
the first step where the two runs disagree, and how the disagreement grows.  A static chain that
is correct agrees with the stock engine on step 0 (same prefill, same weights) and only forks
where a bf16 tie flips the argmax; a broken one is already wrong at step 1, or returns NaN.

  python stepwise.py --a golden_bench32k.jsonl --b /tmp/static_rows1.jsonl --row 0 --steps 32
"""
import argparse
import json
import math


def load(path, row_idx):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if r.get("ids")]
    return rows[row_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference file (same format)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()

    A, B = load(a.a, a.row), load(a.b, a.row)
    ta, tb = A.get("topk") or [], B.get("topk") or []
    ia, ib = A.get("ids") or [], B.get("ids") or []
    print(f"a {A.get('tag', '?')}: {len(ia)} ids   b {B.get('tag', '?')}: {len(ib)} ids")
    print(f"{'step':>4} {'a_argmax':>9} {'b_argmax':>9} {'a_lp':>8} {'b_lp':>8} {'|d|':>7} {'top5':>5}  ids")
    fork = None
    for i in range(min(a.steps, len(ta), len(tb))):
        sa, sb = ta[i], tb[i]
        ma = {int(t): float(v) for v, t in sa}
        mb = {int(t): float(v) for v, t in sb}
        aa = max(ma, key=lambda t: ma[t]) if ma else None
        ab = max(mb, key=lambda t: mb[t]) if mb else None
        la, lb = (ma.get(aa), mb.get(ab)) if aa and ab else (None, None)
        d = abs(ma.get(aa, float("nan")) - mb.get(aa, float("nan"))) if aa else float("nan")
        g5 = set(sorted(ma, key=lambda t: -ma[t])[:5])
        h5 = set(sorted(mb, key=lambda t: -mb[t])[:5])
        ov = len(g5 & h5) / 5
        same = "same" if ia[i:i + 1] == ib[i:i + 1] else ""
        if fork is None and (aa != ab or (not math.isfinite(float(la or 0))) or (not math.isfinite(float(lb or 0)))):
            fork = i
        print(f"{i:>4} {aa!s:>9} {ab!s:>9} {la!s:>8} {lb!s:>8} {d:>7.3f} {ov:>5.1f}  "
              f"{ia[i] if i < len(ia) else '?'}/{ib[i] if i < len(ib) else '?'} {same}")
    print(f"\nfirst divergence: step {fork}" if fork is not None else "\nno divergence in the compared steps")


if __name__ == "__main__":
    main()
