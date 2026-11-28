"""Judge a chain's logits against the reference's OWN run-to-run spread (FINDINGS 12.40.5).

Why this and not a fixed tolerance: SGLang is not bit-repeatable past L1, so its own logits move run to run. The band is
the relRMS between two of its own launches (dec_io2 and dec_io3, same hook same flags); ours is inside when
relRMS(ours, ref) <= relRMS(ref, ref') -- the ratio below is 1.0 at the band edge.

    python cmp_logits_band.py --ours /tmp/e2e_cert/logits.pt \
        --ref /tmp/dec_io2/layers/step1/logits.f32 --ref2 /tmp/dec_io3/layers/step1/logits.f32

Also prints the argmax and the top-5 ids of each: the strongest end-to-end statement available on this stack is
"same argmax, same top-5 set, no further from a reference than the reference is from itself".
"""
import argparse

import numpy as np
import torch


def relrms(a, b):
    """The chain's relRMS: ||a - b|| / ||b||, fp64 accumulation (as cmp_band.py)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))


def top5(x):
    ids = np.argsort(-x)[:5]
    return [int(i) for i in ids], [float(x[i]) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="e2e_decode DSV41_E2E_DUMP_LOGITS file (torch.save of a float tensor)")
    ap.add_argument("--ref", default="/tmp/dec_io2/layers/step1/logits.f32")
    ap.add_argument("--ref2", default="/tmp/dec_io3/layers/step1/logits.f32")
    a = ap.parse_args()

    got = torch.load(a.ours, map_location="cpu")
    ours = (got["logits"] if isinstance(got, dict) else got).float().numpy().reshape(-1)
    ref = np.fromfile(a.ref, dtype=np.float32).reshape(-1)
    ref2 = np.fromfile(a.ref2, dtype=np.float32).reshape(-1)
    assert ours.size == ref.size == ref2.size, f"sizes {ours.size} {ref.size} {ref2.size}"

    band = relrms(ref2, ref)
    d1, d2 = relrms(ours, ref), relrms(ours, ref2)
    print(f"reference vs reference   relRMS {band:.4e}  max|d| {np.abs(ref2 - ref).max():.3f}")
    print(f"ours vs {a.ref.split('/')[2]:10s}  relRMS {d1:.4e}  max|d| {np.abs(ours - ref).max():.3f}   -> {d1 / band:.2f}x of band")
    print(f"ours vs {a.ref2.split('/')[2]:10s}  relRMS {d2:.4e}  max|d| {np.abs(ours - ref2).max():.3f}   -> {d2 / band:.2f}x of band")
    print(f"-> {'INSIDE' if max(d1, d2) <= band else 'OUTSIDE'} the reference's own band "
          f"(worst {max(d1, d2) / band:.2f}x)")

    oi, ov = top5(ours)
    ri, _ = top5(ref)
    r2i, _ = top5(ref2)
    print(f"\nours   top5 {oi}")
    print(f"ref    top5 {ri}")
    print(f"ref2   top5 {r2i}")
    print(f"argmax {'MATCH' if oi[0] == ri[0] else 'DIFF'} (ours {oi[0]}, ref {ri[0]}, ref2 {r2i[0]})")
    print(f"top-5 set vs ref  {'SAME' if set(oi) == set(ri) else 'DIFF: ' + str(sorted(set(oi) ^ set(ri)))}")
    print(f"top-5 set vs ref2 {'SAME' if set(oi) == set(r2i) else 'DIFF: ' + str(sorted(set(oi) ^ set(r2i)))}")


if __name__ == "__main__":
    main()
