"""Judge a full_decode chain against the reference's OWN launch-to-launch spread, per sublayer.

Why this and not a fixed tolerance: SGLang is bit-repeatable through L1 and NOT past it. Four independent launches of
the same checkpoint, same token, same settings agree bit for bit at L0/L1 and then part ways -- 0.64% on L2's
residual, 8% at L10, 13% at L20, 11% at L39 (`ref_spread.py`). Past the point where the reference stops agreeing with
itself there is no single right answer, so "our relRMS vs dec_coef8 must be < 1%" is not a measurable bar: no
implementation, SGLang included, can clear it. The only honest question is whether our chain lands inside the band
the reference spans on its own.

Note this is NOT a weak criterion: the divergence is chaotic, so an implementation that is *wrong* in a kernel lands
far outside the band (the MoE at 0.6-1.6 before §12.38's buffer fix), while one that is right in every kernel lands
inside it, whatever the specific value.

    cmp_band.py --chain /tmp/chain_idx [--runs dec_coef6,dec_coef7,dec_coef8] [--slack 1.5]

Per quantity the band is the largest pairwise relRMS among the runs, and the row's ratio is
relRMS(ours, ref) / band. `ratio <= slack` = inside. Bit-exact layers (band = 0) must match bit for bit; those are
reported separately because there no band can excuse anything.

The MoE residual is mapped one layer up: the reference records layer L-1's FFN post under layer L (`L{L}_mpost`), so
`moe{L}_streams_out` is compared with `L{L+1}_mpost` -- with the band taken on that same quantity.
"""
import argparse
import itertools
import os

import numpy as np

HC, HID, T = 4, 5120, 8

# quantity -> (chain file pattern, reference suffix)
# `streams_eg` is the residual the chain actually follows after a MoE (the engram rewrites it); it only exists for the
# layers whose boundary is fused, and `full_decode.py` falls back to `streams_out` where it does not.
Q = {
    "apost":   (("attn{L}_streams_out.bin",), "_apost"),
    "apost_x": (("attn{L}_x_out.bin",), "_apost_x"),
    "mpost":   (("moe{L}_streams_out.bin",), "_mpost"),
}


def inverted_s(chain, L, ref_dir):
    """The chain's attention output `s`, inverted from its own residual and coefficients (cmp_all.infer_s).

    `attn{L}_x_out.bin` cannot be used directly: it is rank 0's o_proj partial sum, before the all-reduce (norm 8.9
    against the reference's 35.6 at layer 0), so a direct comparison reports ~1.2 for a kernel that is at 0.005. Only
    the inversion recovers the merged tensor, and it closes on `moe{L-1}_streams_out` at 1.3e-4 - 5e-4."""
    from cmp_all import as_streams, chain_f32, infer_s
    r = chain_vec(chain, L, ("attn{L}_streams_out.bin",), HC * HID)
    cf = os.path.join(chain, f"attn{L}_coef.bin")
    if r is None or not os.path.exists(cf):
        return None
    cf = chain_f32(cf)
    if cf.size < 32:
        return None
    if L == 0:
        p = os.path.join(ref_dir, "L00_in.f32")
        Rin = np.fromfile(p, dtype=np.float32).astype(np.float64).reshape(HC, HID) \
            if os.path.exists(p) else None
    else:
        # the chain's post reads the engram'd residual where one exists (`moe0_streams_eg` closes at 2.2e-4 against
        # 5.4e-1 for `moe0_streams_out` at L1) and `streams_out` everywhere else (L2+: 5.0e-4).
        Rin = chain_vec(chain, L - 1, ("moe{L}_streams_eg.bin",), HC * HID)
        if Rin is None:
            Rin = chain_vec(chain, L - 1, ("moe{L}_streams_out.bin",), HC * HID)
        Rin = as_streams(Rin) if Rin is not None else None
    if Rin is None:
        return None
    return infer_s(as_streams(r), cf[:32], Rin)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def ref_vec(run, L, suf):
    p = os.path.join("/tmp", run, "layers", "step1", f"L{L:02d}{suf}.f32")
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype=np.float32).astype(np.float64)


def chain_vec(d, L, pats, n):
    for pat in pats:
        p = os.path.join(d, pat.format(L=L))
        if os.path.exists(p):
            v = np.fromfile(p, dtype=np.uint16).astype(np.uint32) << 16
            v = v.view(np.float32).astype(np.float64)
            if v.size < n:
                continue
            return v[:n]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--runs", default="dec_coef6,dec_coef7,dec_coef8")
    ap.add_argument("--ref", default="dec_coef8", help="the run ours is measured against (the ratio's numerator)")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--slack", type=float, default=1.5)
    a = ap.parse_args()
    runs = a.runs.split(",")

    print(f"chain {a.chain}   band from {len(runs)} runs ({','.join(runs)})   measured against {a.ref}")
    print(f"{'layer quant':14s} {'band':>10s} {'ours':>10s} {'ratio':>7s}  verdict")
    out = []
    for L in range(a.layers):
        for q, (pats, suf) in Q.items():
            if q == "mpost":
                if L + 1 >= a.layers:
                    continue
                vsuf, vl = suf, L + 1
            else:
                vsuf, vl = suf, L
            vecs = [v for v in (ref_vec(r, vl, vsuf) for r in runs) if v is not None]
            n = HC * HID if q in ("apost", "mpost") else HID
            if len(vecs) < 2:
                continue
            band = max(rel(x, y) for x, y in itertools.combinations(vecs, 2))
            ref = ref_vec(a.ref, vl, vsuf)
            ours = inverted_s(a.chain, L, os.path.join("/tmp", a.ref, "layers", "step1")) if q == "apost_x" \
                else chain_vec(a.chain, L, pats, n)
            if ref is None or ours is None:
                continue
            d = rel(ours[:ref.size], ref)
            ratio = d / band if band > 0 else float("inf")
            if band == 0.0:
                verdict = "bitexact" if d == 0.0 else f"BITEXACT-LAYER off by {d:.2e}"
            else:
                verdict = "inside" if ratio <= a.slack else "OUTSIDE"
            out.append((L, q, band, d, ratio, verdict))
            print(f"{L:<6d}{q:8s} {band:>10.3e} {d:>10.3e} {ratio:>7.2f}  {verdict}")

    if not out:
        print("nothing compared")
        return
    bad = [r for r in out if r[5] not in ("inside", "bitexact")]
    ins = [r for r in out if r[5] == "inside"]
    print(f"\n  inside the band: {len(ins)}/{len(out)}   worst ratio {max(r[4] for r in out):.2f} "
          f"(layer {max(out, key=lambda r: r[4])[0]}, {max(out, key=lambda r: r[4])[1]})")
    be = [r for r in out if r[2] == 0.0]
    if be:
        print(f"  bit-exact layers: {len(be)}  " + ("all matched" if all(r[3] == 0.0 for r in be)
                                                 else f"off by up to {max(r[3] for r in be):.2e}"))
    if bad:
        print(f"  outside: {len(bad)}  e.g. " + "; ".join(f"L{r[0]} {r[1]} ratio {r[4]:.2f}" for r in bad[:6]))


if __name__ == "__main__":
    main()
