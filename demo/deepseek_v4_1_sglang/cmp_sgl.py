"""Compare a full_decode dump with one of SGLang's dumped decode steps, sublayer by sublayer.

The chain writes `attn{L}_streams_out` / `moe{L}_streams_out` (bf16 raw, [T, 20480] tokens by 4 hc streams); SGLang's
dump hook writes, per layer, `L{LL}_apost` / `L{LL}_mpost` (fp32 rows of the same 20480). Those two are the same
quantity: the mHC residual right after the attention / the MoE update. A whole layer's row (`L{LL}_st`) is the MoE
post of that layer, so the layer rows are comparable with `moe{L}_streams_out` too.

  python cmp_sgl.py --dump /tmp/full_dec1 --sgl /tmp/dec_ref/layers/step1

Use the DECODE step's directory, never the prefill: SGLang's prefill sparse attention reads its own chunk's tokens from
a flat bf16 buffer and only the prefix from the fp8 pool, so a prefill row is not the fp8-only attention a decode chain
runs. `python cmp_sgl.py --which` prints what each side holds.
"""
import argparse
import glob
import os
import re

import numpy as np

HID, HC = 5120, 4
STREAMS = HID * HC


def raw_chain(path, what):
    """The chain's bf16 dump as fp64."""
    if what == "logits":
        return np.fromfile(path, dtype=np.float32).astype(np.float64)
    a = np.fromfile(path, dtype=np.uint16)
    return ((a.astype(np.uint32) << 16).view(np.float32)).astype(np.float64)


def raw_sgl(path):
    """SGLang's hook writes the residual already as fp32 (bf16 values, exactly representable)."""
    return np.fromfile(path, dtype=np.float32).astype(np.float64)


def live_row(v, what):
    """The chain writes [T, N] and only row 0 is live (T = 8, one real token, the rest idle rows)."""
    n = {"streams_out": STREAMS, "x_out": HID}.get(what)
    if n is not None and v.size == 8 * n:
        return v.reshape(8, n)[0]
    return v


def rr(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-30))


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="full_decode --dump-dir")
    ap.add_argument("--sgl", required=True, help="SGLang's DIR/layers/step{N}")
    ap.add_argument("--layers", default=None, help="a,b,c (default: every layer the two sides share)")
    ap.add_argument("--which", action="store_true", help="print what each side holds, then exit")
    a = ap.parse_args()

    chain = {}
    for p in glob.glob(os.path.join(a.dump, "*.bin")):
        b = os.path.basename(p)[:-4]
        m = re.fullmatch(r"(attn|moe)(\d+)_(streams_out|x_out)", b)
        if m:
            chain[(m.group(1), int(m.group(2)), m.group(3))] = p
    sgl = {}
    for p in glob.glob(os.path.join(a.sgl, "*.f32")):
        m = re.fullmatch(r"L(\d+)(_[a-z]+)?", os.path.basename(p)[:-4])
        if m:
            sgl[(int(m.group(1)), m.group(2) or "_out")] = p
    if a.which:
        print(f"chain: {len(chain)} files, sublayers {sorted({(s, L) for s, L, w in chain if w == 'streams_out'})[:4]} ...")
        print(f"  kinds {sorted({w for _, _, w in chain})}")
        print(f"sglang: {len(sgl)} files, layers {min(L for L, _ in sgl)}..{max(L for L, _ in sgl)}")
        print(f"  kinds {sorted({s for _, s in sgl})}")
        return

    layers = sorted({L for _, L, w in chain if w == "streams_out"})
    if a.layers:
        layers = [int(x) for x in a.layers.split(",")]
    print(f"{'L':>3} {'sub':>5} {'sgl rms':>9} {'static':>9} {'relRMS':>7} {'cos':>8}  verdict")
    worst = (0.0, None)
    for L in layers:
        for sub, suf in (("attn", "_apost"), ("moe", "_mpost")):
            ps, pc = sgl.get((L, suf)), chain.get((sub, L, "streams_out"))
            if ps is None or pc is None:
                continue
            s = raw_sgl(ps)
            c = live_row(raw_chain(pc, "streams_out"), "streams_out")
            r, k = rr(c, s), cos(c, s)
            note = "exact" if k > 0.9999 else ("close" if k > 0.99 else ("drift" if k > 0.9 else "WRONG"))
            print(f"{L:3d} {sub:>5} {np.sqrt((s ** 2).mean()):9.5f} {np.sqrt((c ** 2).mean()):9.5f} "
                  f"{r:7.3f} {k:+8.4f}  {note}")
            if k < worst[0] or worst[1] is None:
                worst = (k, f"L{L} {sub}")
    print(f"\nfirst/worst sublayer: {worst[1]} (cos {worst[0]:+.4f})")


if __name__ == "__main__":
    main()
