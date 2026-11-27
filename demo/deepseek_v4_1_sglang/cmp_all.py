"""Compare a full_decode.py dump with SGLang's decode-step dumps, every layer.

The chain's per-sublayer residuals are the same quantity as the SGLang hook's `_apost`/`_mpost`:
`attn{L}_streams_out` is the residual right after layer L's attention, `moe{L}_streams_out` the one after its
MoE. For layers whose FFN post is fused into the next layer's boundary (0, 1, 14, 20) there is no `_mpost`, and
the FFN's result is the next layer's `_in` instead -- so that is the candidate for `moe{L}`.

Besides the residuals, it inverts the mHC post for the sublayer OUTPUT `s`:
    R'[j] = post[j] * s + sum_i comb[i][j] * R[i]
which is well conditioned only where `post[j]` is large: stream 3 at layer 0 has post 0.1362 while streams 0..2
have ~1e-4, so their matching residuals say nothing about `s`. Reporting `s` from the largest-post stream is what
keeps a match on the near-zero rows from being read as "the sublayer is right".

  python cmp_all.py --chain /tmp/full_dec7 [--sgl /tmp/dec_coef2/layers/step1] [--layers 40] [--tol 1e-2]
"""
import argparse
import os

import numpy as np

HC, HID, T = 4, 5120, 8


def sgl_vec(path, rows=1):
    return np.fromfile(path, dtype=np.float32).astype(np.float64)


def chain_vec(path):
    a = np.fromfile(path, dtype=np.uint16)
    return ((a.astype(np.uint32) << 16).view(np.float32)).astype(np.float64)


def as_streams(v, want_row):
    """[T, HC*HID] (chain) or [HC*HID] (SGLang) -> [HC, HID] of the token's row."""
    if v.size == T * HC * HID:
        v = v.reshape(T, HC, HID)[0]
    return v.reshape(HC, HID)


def cmp_streams(a, b):
    """(cos, relRMS) over the 4 streams, plus each stream's own cosine."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cos = (a * b).sum() / (na * nb).sum()
    return cos, np.linalg.norm(a - b) / np.linalg.norm(b), (a * b).sum(1) / (na * nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, help="full_decode --dump-dir")
    ap.add_argument("--sgl", default="/tmp/dec_coef2/layers/step1")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-2, help="relRMS counted as aligned")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    def c(name):
        p = os.path.join(a.chain, name)
        return chain_vec(p) if os.path.exists(p) else None

    def s(name):
        p = os.path.join(a.sgl, name)
        return sgl_vec(p) if os.path.exists(p) else None

    print(f"{'sublayer':10s} {'ref':12s} {'resid cos':>10s} {'resid relRMS':>13s} {'out cos':>9s} {'out relRMS':>11s}")
    rows = []
    for L in range(a.layers):
        # attention: the residual after it is this layer's _apost
        for kind, refs in (("attn", [f"L{L:02d}_apost"]), ("moe", [f"L{L:02d}_mpost", f"L{L + 1:02d}_in"])):
            v = c(f"{kind}{L}_streams_out.bin")
            if v is None:
                continue
            ref = refn = None
            for rn in refs:
                w = s(rn + ".f32")
                if w is not None and w.size == HC * HID:
                    ref, refn = w, rn
                    break
            if ref is None:
                continue
            A, B = as_streams(v, 0), ref.reshape(HC, HID)
            cos, rel, per = cmp_streams(A, B)
            # the sublayer's own output via the largest-post stream (the others are ill-conditioned)
            out_cos = out_rel = None
            ct = lambda nm: np.fromfile(os.path.join(a.sgl, f"L{L:02d}_ct_{nm}.f32"), dtype=np.float32).astype(np.float64)
            pp, cc = ct("post"), ct("comb").reshape(HC, HC)
            j = int(np.argmax(pp))
            # both sides invert with the SAME (SGLang) coefficients and the SAME input residual, so what differs is s
            Rin = s(f"L{L:02d}_in.f32")
            if pp[j] > 2e-2 and Rin is not None:
                Rin = Rin.reshape(HC, HID)
                sch = (A[j] - sum(cc[i, j] * Rin[i] for i in range(HC))) / pp[j]
                ssg = (B[j] - sum(cc[i, j] * Rin[i] for i in range(HC))) / pp[j]
                out_cos = float(sch @ ssg / (np.linalg.norm(sch) * np.linalg.norm(ssg)))
                out_rel = float(np.linalg.norm(sch - ssg) / np.linalg.norm(ssg))
            ok = "ok " if rel < a.tol else "BAD"
            oc = f"{out_cos:+9.5f}" if out_cos is not None else "        -"
            orr = f"{out_rel:11.4f}" if out_rel is not None else "          -"
            print(f"{kind}{L:<7d} {refn:12s} {cos:+10.5f} {rel:13.4f} {oc} {orr}  {ok}")
            rows.append((kind, L, rel, out_rel))

    bad = [r for r in rows if r[2] >= a.tol]
    print(f"\naligned (<{a.tol}): {len(rows) - len(bad)}/{len(rows)} residuals")
    if bad:
        first = bad[0]
        print(f"first divergence: {first[0]}{first[1]} (residual relRMS {first[2]:.4f})")
    outs = [r[3] for r in rows if r[3] is not None]
    if outs:
        print(f"sublayer outputs checked: {len(outs)}, worst relRMS {max(outs):.4f}, "
              f"median {float(np.median(outs)):.4f}")


if __name__ == "__main__":
    main()
