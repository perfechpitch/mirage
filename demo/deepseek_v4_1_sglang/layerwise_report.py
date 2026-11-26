"""Read a check-mode run's per-sublayer comparison and say which layer diverged first.

In check mode (DSV41_STATIC_CHECK=1) each sublayer runs SGLang's own post and the static one on the same input, and
_mhc_end prints one line per sublayer (dsv41_static_hook.py:987, rank 0, steps <= DSV41_STATIC_CHECK_STEPS):

  check mHC attn L4 step 1: R' relRMS 6.02e-02, pre 1.1e-02, x 5.9e-02 (next yes)

Three numbers, three different places a chain can break:
  R'   the residual stream's next value (mHC post: post_j s + sum_i comb[i][j] R_i)
  pre  this sublayer's collapse coefficients
  x    the next sublayer's input after the norm -- the residual is what the next layer consumes, so x is the number
       that decides whether the chain continues correctly

Why this exists: the end-to-end number (accept.sh, compare_golden) says "the 40 layers are wrong" and nothing about
where. This says where, on the real weights and the real 32K pools, without any offline alignment of dumps.

Read it as: the rows before the first jump are correct, the jumping row is the culprit, everything after inherits it.
A NaN or an enormous relRMS at one layer with the previous ones at 1e-2 is exactly the signature the goal describes
("L0-L3 finite, attn L4 emits NaN with a finite input").

  python layerwise_report.py --log /tmp/layerwise/kern.log
  python layerwise_report.py --log ... --cap 5e-2      # a different threshold for "jumped"
"""
import argparse
import math
import re
import sys

# the line _mhc_end prints. `x` is absent when the layer has no gamma_next (a norm is not fused into it) and `bits`
# can sit between the numbers and the "(next ...)" tail, so neither is required here.
NUM = r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"
LINE = re.compile(
    r"check mHC\s+(\w+)\s+L(\d+)\s+step\s+(\d+):\s*"
    r"R' relRMS\s+" + NUM + r",\s*"
    r"pre\s+" + NUM +
    r"(?:,\s*x\s+" + NUM + r")?")


def num(s):
    if s is None:
        return None
    v = float(s)
    return v if math.isfinite(v) else math.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--cap", type=float, default=1e-1,
                    help="a relRMS above this is a jump. The layers that agree sit at 1e-2 to 6e-2 (measured, "
                         "l2_pool3), so 1e-1 is above every healthy attention row; raise it if a whole run is "
                         "slightly worse and you only care about the one that explodes")
    ap.add_argument("--step", type=int, default=None, help="only this decode step (default: all printed)")
    a = ap.parse_args()

    rows = []
    for line in open(a.log, errors="replace"):
        m = LINE.search(line)
        if m:
            rows.append(dict(kind=m.group(1), layer=int(m.group(2)), step=int(m.group(3)),
                             rprime=num(m.group(4)), pre=num(m.group(5)), x=num(m.group(6))))
    if not rows:
        # the two ways this happens, which need opposite fixes
        txt = open(a.log, errors="replace").read()
        if "check attn L" in txt:
            print("this log has 'check attn ... y relRMS' lines, i.e. check mode WITHOUT mHC (MHC=0). Those compare\n"
                  "each sublayer's own y and are fine, but they never exercise the mHC post or the in-kernel\n"
                  "all-reduce -- the chain that goes NaN. Rerun with DSV41_STATIC_MHC=1.", file=sys.stderr)
        else:
            print(f"no 'check mHC' line in {a.log}: was DSV41_STATIC_CHECK=1 and DSV41_STATIC_MHC=1, and did the\n"
                  "step reach check mode at all (the hook logs 'NOT checked -- ...' with the reason when it skips)?",
                  file=sys.stderr)
        return 2

    steps = sorted({r["step"] for r in rows})
    if a.step is not None:
        rows = [r for r in rows if r["step"] == a.step]
        steps = [a.step]

    print(f"{'step':>4} {'sub':<5} {'layer':>5} {'R\' relRMS':>10} {'pre':>10} {'x':>10}  verdict")
    first_bad = []
    for step in steps:
        rs = [r for r in rows if r["step"] == step]
        seen_bad = False
        for r in rs:
            bad = [k for k, v in (("R'", r["rprime"]), ("pre", r["pre"]), ("x", r["x"]))
                   if v is not None and not math.isfinite(v)]
            jump = r["x"] is not None and math.isfinite(r["x"]) and r["x"] > a.cap
            if (bad or jump) and not seen_bad:
                seen_bad = True
                first_bad.append(r)
                verdict = ("NON-FINITE " + ",".join(bad)) if bad else f"JUMP > {a.cap:g}"
                verdict += "  <- first"
            elif seen_bad:
                verdict = "inherited"
            else:
                verdict = ""
            f = lambda v: "nan" if v is not None and not math.isfinite(v) else ("-" if v is None else f"{v:.2e}")
            print(f"{step:>4} {r['kind']:<5} {r['layer']:>5} {f(r['rprime']):>10} {f(r['pre']):>10} "
                  f"{f(r['x']):>10}  {verdict}")

    print()
    if not first_bad:
        print(f"no sublayer exceeded x {a.cap:g} and nothing was non-finite: every layer agrees with SGLang")
        return 0
    for r in first_bad:
        print(f"step {r['step']}: the chain first breaks at {r['kind']} L{r['layer']}"
              f" (R' {r['rprime']}, pre {r['pre']}, x {r['x']}); the layers above it are correct")
    print(f"\nFAIL: {len(first_bad)} step(s) diverged")
    return 1


if __name__ == "__main__":
    sys.exit(main())
