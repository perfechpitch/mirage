"""Is an attention sublayer's output `y` ours or the reference's? One number, with its own error bar.

The boundary is decidable because mHC's post is *invertible* on the dump:

    R'[i] = bf16( post[i] * y + sum_j comb[j][i] * R[j] )

`R` (the layer's 4 input streams, `L{LL}_in`), `post`/`comb` (`L{LL}_ct_*`) and `R'` (`L{LL}_apost`) are all dumped,
and `y` (`L{LL}_apost_x`) is dumped too -- so the identity can be *checked* before it is used to invert. With the
reference's own numbers it reproduces `L{LL}_apost` bit for bit, which is what makes the inversion legitimate:

    y = (R'[i] - sum_j comb[j][i] R[j]) / post[i]

Only stream 3 is well conditioned (post = 0.1362 at L0; the others are 7e-4, 4.8e-4, 2.5e-9, so their y is 1e3-1e6
times noisier). The inversion is solved on the bf16 grid, so it reports how many elements it closed exactly -- if that
is not 100%, the numbers below are not trustworthy.

Why this and not the streams_out relRMS: `bf16(post*y + comb@R)` rounds, so a y that differs by *less than one bf16
ulp* still shows up as 1-3 ulp on many elements and as ~4e-3 relRMS on R'. The ulp histogram is the honest scale --
compare it against the "one bf16 ulp" of y (2^-8 = 3.9e-3 relative).

    attn_y_audit.py --ours /tmp/chain_a0 --ref /tmp/dec_coef8 --layers 0,1
"""
import argparse
import os
import sys

import numpy as np

HID, HC = 5120, 4


def f32(p):
    return np.fromfile(p, np.float32).astype(np.float64)


def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16).view(np.float32).astype(np.float64)


def ours_bf16(p, width):
    a = np.fromfile(p, np.int16).view(np.uint16)
    assert a.size >= width, f"{p}: {a.size} < {width}"
    return (a[:width].astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def rr(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))


def cos(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def shift(v, k):
    u = v.view(np.int32).astype(np.int64) + k
    return u.astype(np.int32).view(np.float32).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="/tmp/chain_a0")
    ap.add_argument("--ref", default="/tmp/dec_coef8")
    ap.add_argument("--layers", default="0,1")
    a = ap.parse_args()
    R = os.path.join(a.ref, "layers", "step1")

    print(f"\n  chain {a.ours}   vs SGLang {a.ref} step 1")
    for L in [int(x) for x in a.layers.split(",")]:
        need = [f"L{L:02d}_in.f32", f"L{L:02d}_apost.f32", f"L{L:02d}_apost_x.f32",
                f"L{L:02d}_ct_post.f32", f"L{L:02d}_ct_comb.f32"]
        if not all(os.path.exists(os.path.join(R, f)) for f in need):
            print(f"\n  L{L}: the reference dump is missing {[f for f in need if not os.path.exists(os.path.join(R, f))]}")
            continue
        Rv = f32(os.path.join(R, f"L{L:02d}_in.f32")).reshape(HC, HID)
        ap_ref = f32(os.path.join(R, f"L{L:02d}_apost.f32")).reshape(HC, HID)
        y_ref = f32(os.path.join(R, f"L{L:02d}_apost_x.f32"))
        post = f32(os.path.join(R, f"L{L:02d}_ct_post.f32"))
        comb = f32(os.path.join(R, f"L{L:02d}_ct_comb.f32")).reshape(HC, HC)
        o_ap = ours_bf16(os.path.join(a.ours, f"attn{L}_streams_out.bin"), HC * HID).reshape(HC, HID)

        def mix(y):
            return np.stack([bf16(post[i] * y + comb[:, i] @ Rv) for i in range(HC)])

        print(f"\n  L{L}   post {np.array2string(post, precision=6)}")
        print(f"        y_ref: bf16-valued {np.mean(bf16(y_ref) == y_ref) * 100:.1f}%  rms {np.sqrt((y_ref ** 2).mean()):.6f}")
        ok = rr(mix(y_ref).ravel(), ap_ref.ravel())
        print(f"        identity check (ref coef + ref y -> R'): relRMS vs L{L:02d}_apost {ok:.3e}"
              + ("   <- invertible" if ok < 1e-12 else "   <- NOT exact, stop"))
        if ok >= 1e-12:
            continue
        print(f"        the boundary: our streams_out vs L{L:02d}_apost  relRMS {rr(o_ap.ravel(), ap_ref.ravel()):.3e}")

        i = int(np.argmax(post))
        base = comb[:, i] @ Rv
        y0 = ((o_ap[i] - base) / post[i]).astype(np.float32)
        sol, done = np.zeros(HID), np.zeros(HID, bool)
        for k in range(-4, 5):
            c = shift(y0, k)
            hit = (bf16(post[i] * c + base) == o_ap[i]) & ~done
            sol, done = np.where(hit, c, sol), done | hit
        print(f"        inversion on stream {i} (post {post[i]:.6f}): closed {done.mean() * 100:.2f}% exactly")
        d = sol - y_ref
        ulp = 2.0 ** np.floor(np.log2(np.abs(y_ref) + 1e-30) - 7)
        hist = " ".join(f"{k}ulp:{int(((np.abs(d) / ulp > k - 0.5) & (np.abs(d) / ulp < k + 0.5)).sum())}"
                        for k in (0, 1, 2, 3))
        print(f"        our y vs y_ref: relRMS {rr(sol, y_ref):.3e}  cos {cos(sol, y_ref):.10f}  "
              f"bit-exact {int((sol == y_ref).sum())}/{HID}")
        print(f"                        norms {np.sqrt((sol ** 2).mean()):.6f} vs {np.sqrt((y_ref ** 2).mean()):.6f}   "
              f"|d| in bf16 ulps: {hist}")
        print(f"        (one bf16 ulp of y is 2^-8 = 3.9e-3 relative; anything at this scale is a last-bit difference)")
    print()


if __name__ == "__main__":
    sys.exit(main())
