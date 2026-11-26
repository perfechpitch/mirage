"""Compare full_decode.py's per-sublayer dump against a reference, offline, with no GPU.

full_decode.py --dump-dir DIR writes, one file a sublayer (raw, no header, the dtype is known from the flag):
    DIR/a{L}_x_out.bin        bf16 [T, 5120]    the attention kernel's output rows
    DIR/a{L}_streams_out.bin  bf16 [T, 20480]   its residual streams R'
    DIR/m{L}_x_out.bin        bf16 [T, 5120]
    DIR/m{L}_streams_out.bin  bf16 [T, 20480]
    DIR/logits.bin            fp32 [T, vocab]

Two things this answers, in the order they have to be asked:

  1. does the chain stay finite, and where does it stop changing?  A step reads one row, so a chain that is right
     has x_out changing layer by layer and never NaN. --rms prints this from the run itself; this prints it from
     the files, which is what survives a run whose log was truncated by a crash.

  2. how far is it from the reference?  relRMS = ||got - ref|| / ||ref|| on the same flattening, then a
     classification (dump_layers.classify) that says whether the difference is a scale, a permutation, a few
     elements, or everywhere -- those have nothing in common as bugs.

A reference is another dump in the same layout (name to name). There is no reference for a chain that ran with the
prefill's real KV except another run: SGLang's per-layer outputs are not on disk, so this compares runs with each
other (two builds of the same math must agree bit for bit; a build flag's effect shows up layer by layer).

  python compare_chain.py --dir /tmp/full_run/dump --rms
  python compare_chain.py --dir /tmp/full_run/dump --ref /tmp/full_run2/dump
  python compare_chain.py --dir A --ref B --hist
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HID = 5120
HCS = 4
STREAMS = HCS * HID


def load(path, shape, dtype):
    a = np.fromfile(path, dtype=dtype)
    assert a.size == int(np.prod(shape)) if shape else True, f"{path}: {a.size} elements, want {shape}"
    return a.reshape(shape) if shape else a


def files(d):
    """The dump's sublayers in launch order, as (name, kind, layer, what, path)."""
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.bin"))):
        b = os.path.basename(p)[:-4]
        m = re.fullmatch(r"(attn|moe)(\d+)_(x_out|streams_out)", b)
        if m:
            out.append((b, "a" if m.group(1) == "attn" else "m", int(m.group(2)), m.group(3), p))
        elif b == "logits":
            out.append((b, "h", 10 ** 6, "logits", p))
    out.sort(key=lambda t: (t[2], 0 if t[1] == "a" else 1))
    return out


def shape_of(what):
    return {"x_out": (8, HID), "streams_out": (8, STREAMS), "logits": None}[what]


def dtype_of(what):
    return np.float32 if what == "logits" else np.uint16   # bf16 read as its bits


def to_f32(a, what):
    if what == "logits":
        return a.astype(np.float64)
    return a.view(np.float32) if a.dtype == np.float32 else \
        (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def classify(g, r):
    """dump_layers.classify, on two float64 arrays (same shapes)."""
    if g.shape != r.shape:
        return f"shape {g.shape} vs {r.shape}"
    if np.array_equal(g, r):
        return "equal"
    if not (np.isfinite(g).all() and np.isfinite(r).all()):
        return (f"non-finite got {int((~np.isfinite(g)).sum())}/{g.size}, "
                f"ref {int((~np.isfinite(r)).sum())}/{r.size}")
    if g.size >= 4 and g.argmax() == r.argmax():
        nz = r != 0
        if nz.any():
            ratio = g[nz] / r[nz]
            if np.allclose(ratio, ratio.ravel()[0], rtol=1e-3):
                return f"scale x{ratio.ravel()[0]:.6g}"
    if np.allclose(np.sort(g.ravel()), np.sort(r.ravel()), atol=0):
        return "permuted"
    diff = np.abs(g - r)
    floor = max(1e-6 * float(np.abs(r).max()), 1e-30)
    nbig = int((diff > floor).sum())
    frac = nbig / diff.size
    return (f"sparse {frac * 100:.2f}% ({nbig}/{diff.size}) over {floor:.3g}" if 0 < frac < 0.05
            else "spread")


def relrms(g, r):
    d = np.linalg.norm(g.ravel() - r.ravel())
    n = np.linalg.norm(r.ravel())
    return d / max(n, 1e-30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="a --dump-dir from full_decode.py")
    ap.add_argument("--ref", default=None, help="another dump in the same layout")
    ap.add_argument("--rms", action="store_true", help="each sublayer's rms and non-finite count")
    ap.add_argument("--hist", action="store_true", help="relRMS per sublayer as bars")
    ap.add_argument("--cap", type=float, default=1e-1, help="--hist: the relRMS a full bar is")
    ap.add_argument("--entry", default=None, help="one sublayer's name, e.g. a04_x_out")
    a = ap.parse_args()

    ent = files(a.dir)
    if not ent:
        sys.exit(f"{a.dir}: no *.bin (did full_decode.py run with --dump-dir?)")
    print(f"{a.dir}: {len(ent)} tensors from {len({e[2] for e in ent})} sublayers")

    if a.rms:
        print(f"\n{'sublayer':<16}{'rms':>12}{'non-finite':>12}  head[4]")
        for name, k, L, what, p in ent:
            sh = shape_of(what)
            v = to_f32(load(p, sh, dtype_of(what)), what)
            fin = np.isfinite(v)
            nf = int((~fin).sum())
            head = " ".join(f"{x:+.4f}" for x in v.ravel()[:4])
            print(f"{name:<16}{np.sqrt(np.mean(v[fin] ** 2)) if fin.any() else float('nan'):>12.5g}"
                  f"{nf:>12}  {head}")

    if a.ref:
        ref = {n: (L, w, p) for n, k, L, w, p in files(a.ref)}
        rows = []
        for name, k, L, what, p in ent:
            if name not in ref:
                continue
            rL, rw, rp = ref[name]
            sh = shape_of(what)
            g = to_f32(load(p, sh, dtype_of(what)), what)
            r = to_f32(load(rp, sh, dtype_of(what)), what)
            rows.append((name, relrms(g, r), classify(g, r)))
        print(f"\n{'sublayer':<16}{'relRMS':>12}  what differs")
        for name, v, c in rows:
            print(f"{name:<16}{v:>12.3e}  {c}")
        if a.hist:
            top = max((v for _, v, _ in rows), default=0.0) or 1.0
            print()
            for name, v, c in rows:
                n = max(1, int(40 * min(v / max(a.cap, 1e-30), 1.0)))
                print(f"{name:<16}{'#' * n:<40} {v:.3e} {c}")
        ok = [v for _, v, _ in rows if np.isfinite(v)]
        if ok:
            print(f"\n{len(rows)} compared: max relRMS {max(ok):.3e}, mean {np.mean(ok):.3e}")

    if a.entry:
        pick = [e for e in ent if e[0] == a.entry]
        if not pick:
            sys.exit(f"no {a.entry!r}; names: " + " ".join(n for n, *_ in ent)[:400])
        name, k, L, what, p = pick[0]
        v = to_f32(load(p, shape_of(what), dtype_of(what)), what)
        print(f"\n{name} {v.shape} rms {np.sqrt(np.mean(v ** 2)):.6g}")
        print("row0[:24] " + " ".join(f"{x:+.5f}" for x in v.reshape(8, -1)[0][:24]))


if __name__ == "__main__":
    main()
