"""Teacher-forced per-sublayer fidelity: the direct form of the acceptance criterion.

The free chain accumulates, so a residual comparison at L30 mostly measures how well a chaotic trajectory was carried.
With `DSV41_TEACH=1` every sublayer instead READS SGLang's own residual and writes only its own product:

    attn{L} reads SGLang L{L}_mpost  and writes R' that should be  L{L}_apost
    moe{L}  reads SGLang L{L}_apost  and writes R' that should be  L{L+1}_mpost

so scoring `attn{L}_teach_out` against `L{LL}_apost` measures that one kernel (attention + hc_post_combine), with no
carried-over error at all. `_teach_out.f32` is the fp32 row-0 capture of what the launch actually wrote (not the injected
input -- checked: attn0 against L00_apost is 0.00265, whereas the injected residual would be ~0.5 off).

The `floor` column is SGLang against ITSELF on the same quantity: the run that supplied the inputs (`dec_coef8`) against
a different launch (`dec_coef7`). Nothing can be closer to a reference than the reference is to itself, so any chain
number at or below the floor is at the limit.

NOT used: `_teach_x.f32`. It is the attention core's `o`, not the post-mixed `y` (`L{LL}_apost_x`), so comparing the two
is meaningless (attn0: 0.85, cos 0.66).

    DSV41_TEACH=1 full_decode.py ... --inputs-dir /tmp/dec_coef8 --dump-dir /tmp/chain_teach
    python teacher_table.py --chain /tmp/chain_teach
"""
import argparse
import os

import numpy as np

H, HC = 5120, 4
GOLD = "/tmp/dec_coef8/layers/step1"     # the dump the teacher run was fed from
OTHER = "/tmp/dec_coef7/layers/step1"    # a different launch, for the self-consistency floor


def rr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def golden(d, name):
    p = os.path.join(d, f"{name}.f32")
    if not os.path.exists(p):
        return None
    v = np.fromfile(p, np.float32)
    return v.reshape(HC, H).astype(np.float64) if v.size == HC * H else v.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="/tmp/chain_teach")
    ap.add_argument("--layers", default="0-39")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.layers.split("-"))

    print("teacher-forced: every sublayer reads SGLang's residual, so this scores that one KERNEL, not the trajectory")
    print(f"{'sub':8s} {'vs golden':>10s} {'cos':>11s} {'SGLang-vs-itself':>17s} {'verdict':>10s}")
    over = []
    for L in range(lo, min(hi, 39) + 1):
        for sub, key, LL in (("attn", "apost", L), ("moe", "mpost", L + 1)):
            name = f"{sub}{L}"
            p = os.path.join(a.chain, f"{name}_teach_out.f32")
            g = golden(GOLD, f"L{LL:02d}_{key}")
            if not os.path.exists(p) or g is None:
                continue
            v = np.fromfile(p, np.float32)
            if v.size != g.size:
                continue
            v = v.reshape(HC, H).astype(np.float64)
            o = golden(OTHER, f"L{LL:02d}_{key}")
            floor = rr(o, g) if o is not None and o.size == g.size else float("nan")
            d = rr(v, g)
            if not np.isnan(floor):
                over.append(d / max(floor, 1e-30))
            ver = ("at floor" if np.isnan(floor) else
                   ("AT/BELOW FLOOR" if d <= max(floor, 1e-12) * 1.35 else ("above floor" if d < 0.02 else "ABOVE")))
            fl = "0 (deterministic)" if (not np.isnan(floor) and floor < 1e-9) else f"{floor:.5f}"
            print(f"{name:8s} {d:>10.5f} {cos(v, g):>+11.6f} {fl:>17s} {ver:>10s}")
    if over:
        o = np.array(over)
        print(f"\nchain / reference-self-consistency ratio over {len(o)} sublayers: "
              f"mean {o.mean():.3f}  median {np.median(o):.3f}  worst {o.max():.3f}")


if __name__ == "__main__":
    main()
