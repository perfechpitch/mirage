"""Per-sublayer fidelity at the ONLY granularity the kernels expose: their own output y, for all 80 sublayers.

Why y and not R': the residual R' = post*y + comb^T R is dominated by comb^T R (the carried-over mix), so comparing R'
mostly measures how well the *input* was carried, not what the kernel computed. y is the sublayer's actual product, and
SGLang dumps it directly (`L{LL}_apost_x` / `L{LL}_mpost_x`). Recovering it on the chain's side is exact algebra:

    y[j] = (R'[j] - sum_i comb[i][j] R[i]) / post[j]

taken on the stream whose |post| is largest (the least ill-conditioned). Two baselines are printed per sublayer:

  `own`    the chain's y against the chain's own input -- what the kernel really produced, carried-error included
  `gold`   the same y against SGLang's y, but with SGLang's input subtracted -- isolates the kernel from the input, and is
           only meaningful while the carried error is small (it is not, past ~L10; see FINDINGS 12.3)

  python y_table.py --layers 0-39
"""
import argparse
import itertools
import os

import numpy as np
import torch

H, HC = 5120, 4
GOLD = "/tmp/dec_coef8/layers/step1"


def rr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def golden(L, key, rows=HC):
    f = os.path.join(GOLD, f"L{L:02d}_{key}.f32")
    if not os.path.exists(f):
        return None
    v = np.fromfile(f, np.float32)
    return v.reshape(rows, H).astype(np.float64) if v.size == rows * H else v.astype(np.float64)


def chain_streams(d, name):
    f = os.path.join(d, f"{name}_streams_out.bin")
    if not os.path.exists(f):
        return None
    t = torch.frombuffer(bytearray(open(f, "rb").read()), dtype=torch.bfloat16).float()
    return t.reshape(-1, HC, H)[0].numpy().astype(np.float64)


def chain_coef(d, name):
    f = os.path.join(d, f"{name}_coef.bin")
    if not os.path.exists(f):
        return None
    c = np.fromfile(f, np.float32).reshape(-1, 32)[0]
    return c[:4].astype(np.float64), c[4:8].astype(np.float64), c[8:24].reshape(4, 4).astype(np.float64)


def y_of(Rp, R, post, comb):
    """Least-squares y over ALL four streams: for each component i, R'[j][i] = post[j] y_i + (comb^T R)[j][i] gives four
    scalar equations in one unknown, so y_i = sum_j post[j] z[j][i] / sum_j post[j]^2. This weights the well-conditioned
    streams automatically, which is strictly better than picking one. Returns y, the dominant stream and its post."""
    z = Rp - (comb.T @ R)
    w = post
    y = (w @ z) / max(float(w @ w), 1e-30)
    j = int(np.argmax(np.abs(post)))
    return y, j, float(post[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/chain40")
    ap.add_argument("--layers", default="0-39")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.layers.split("-"))

    print(f"{'sub':8s} {'post':>8s} {'y: |chain|':>10s} {'vs golden y (own in)':>20s} "
          f"{'gold in':>17s} {'deconv':>9s} {'ref scatter':>11s}")
    band, own = [], []
    for L in range(lo, min(hi, 39) + 1):
        for sub, tag, key in (("attn", "attn", "apost"), ("moe", "moe", "mpost")):
            name = f"{sub}{L}"
            LL = L if sub == "attn" else L + 1
            yg = golden(LL, f"{key}_x")
            Rp = chain_streams(a.dump, name)
            cf = chain_coef(a.dump, name)
            if yg is None or Rp is None or cf is None:
                continue
            pre, post, comb = cf
            # the input this sublayer reads, from the CHAIN's point of view: its producer's dumped residual
            if sub == "attn":
                prod = "attn0" if L == 0 else f"moe{L - 1}"
                rg = golden(0, "in") if L == 0 else (golden(L, "eg_y") if golden(L, "eg_y") is not None
                                                    else golden(L, "in"))
            else:
                prod = f"attn{L}"
                rg = golden(L, "apost")
            if L == 0 and sub == "attn":      # attn0's input is the embedding x HC, which no sublayer dumps
                prod, rg = None, None
            rc = chain_streams(a.dump, prod) if prod else None
            if rc is None and prod is None:
                rc = golden(0, "in")          # the embedding IS in the golden dump, so both sides share it
            if rc is not None and sub == "attn" and L > 0:
                # an Engram host layer (L1, L14): model.py applies the engram to the RAW residual at the top of the block,
                # so the attention input is  eg_y = eg_x + gate*value, and the chain's previous MoE kernel dumps only the
                # pre-engram R' (= eg_x). Add the delta back. Both endpoints are dumped, so the delta is exact and does not
                # depend on decoding the engram's own embedding layout. (The chain's own gate was checked against the
                # official one: [0.02089 0.08042 1e-5 5e-5] vs [0.02090 0.08031 1e-5 5e-5], see FINDINGS 12.12.)
                ex, ey = golden(L, "eg_x"), golden(L, "eg_y")
                if ex is not None and ey is not None and ex.size == rc.size and ey.size == rc.size:
                    rc = rc + (ey - ex)
            if rc is None:
                continue
            yc, j, pv = y_of(Rp, rc, post, comb)
            yc_g = y_of(Rp, rg, post, comb)[0] if rg is not None else yc
            # How much of the observed gap the INPUT difference alone would produce: synthesise a perfect R' from the
            # golden output, then deconvolve it with the chain's input. That isolates the deconvolution's own
            # amplification (~1/post) from anything the kernel did.
            ind = float("nan")
            if rg is not None:
                syn = comb.T @ rg + post[:, None] * yg[None, :]
                ind = rr(y_of(syn, rc, post, comb)[0], yg)
            # the reference's own scatter on this same y: max pairwise among FOUR launches of the SAME config. dec_coef5
            # is excluded on purpose -- it is bit-identical to none of them even at L0 (L00_apost 1.52% off), i.e. it was
            # a different configuration, not a repeat. With {2,6,7,8} the reference is bit-exact at L0/L1 and only starts
            # to scatter at L2 (0.6%), jumping to 8.5% at L8 -- one layer after the L7 gate tie (FINDINGS 12.13).
            ys = []
            for d in (f"/tmp/dec_coef{n}/layers/step1" for n in (2, 6, 7, 8)):
                f = os.path.join(d, f"L{LL:02d}_{key}_x.f32")
                if os.path.exists(f):
                    t = np.fromfile(f, np.float32)
                    if t.size == yg.size:
                        ys.append(t.astype(np.float64))
            b = max((rr(u, v) for u, v in itertools.combinations(ys, 2)), default=float("nan"))
            if not np.isnan(b):
                band.append(b)
            own.append(rr(yc, yg))
            print(f"{name:8s} {pv:>8.4f} {np.linalg.norm(yc)/np.sqrt(H):>10.5f} "
                  f"{rr(yc, yg):>10.5f}/{cos(yc, yg):+.6f} {rr(yc_g, yg):>17.5f} {ind:>9.5f} {b:>11.5f}")
    if band:
        print(f"\nreference's own scatter on the same y (max pairwise over dec_coef 2/6/7/8): "
              f"mean {np.mean(band):.5f}  median {np.median(band):.5f}  worst {np.max(band):.5f}")
    if own:
        print(f"chain vs golden y, own input: mean {np.mean(own):.5f}  median {np.median(own):.5f}  "
              f"worst {np.max(own):.5f}")
        if band:
            n = min(len(own), len(band))
            close = sum(1 for x, b in zip(own[:n], band[:n]) if x <= b)
            print(f"sublayers where the chain is CLOSER to SGLang than SGLang's other launch is: {close}/{n}")


if __name__ == "__main__":
    main()
