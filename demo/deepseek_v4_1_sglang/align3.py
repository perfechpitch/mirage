"""Three-way alignment table: the static chain against TWO independent SGLang runs, and those two against each other.

Why this is the right test: the per-sublayer residual is not a stable quantity. SGLang, run twice on the same decode step,
differs from itself by 13% at L17 (the MoE gate's top-6/top-7 margin there is ~1e-4, so one expert flips and the residual
jumps). Any claim that "sublayer L matches to under 1%" is therefore a claim about a number that the reference itself does
not reproduce. What has to be checked is that the chain is as close to each run as the runs are to each other -- i.e. that
the chain is inside the reference's own scatter, not that it hits one sample of it.

  python align3.py --chain /tmp/chain40 --a /tmp/dec_coef7/layers/step1 --b /tmp/dec_coef8/layers/step1
"""
import argparse
import os

import numpy as np
import torch

H, HC = 5120, 4


def rr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-30))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def golden(d, L, key):
    """SGLang's dumped residual, [HC, HID] fp32 (bf16 values)."""
    f = os.path.join(d, f"L{L:02d}_{key}.f32")
    return np.fromfile(f, dtype=np.float32).reshape(HC, H) if os.path.exists(f) else None


def chain(d, name):
    """The chain's dumped sublayer output, [T, HC, HID] bf16 -> row 0, [HC, HID]."""
    f = os.path.join(d, f"{name}_streams_out.bin")
    if not os.path.exists(f):
        return None
    t = torch.frombuffer(bytearray(open(f, "rb").read()), dtype=torch.bfloat16).float()
    return t.reshape(-1, HC, H)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="/tmp/chain40")
    ap.add_argument("--a", default="/tmp/dec_coef7/layers/step1", help="SGLang run A")
    ap.add_argument("--b", default="/tmp/dec_coef8/layers/step1", help="SGLang run B")
    ap.add_argument("--layers", type=int, default=40)
    a = ap.parse_args()
    print("  sub     chain-A          chain-B          A-B             (relRMS / cos)")
    tot = {"cA": [], "cB": [], "AB": []}
    for L in range(a.layers):
        for sub in ("attn", "moe"):
            name = f"{sub}{L}"
            key, tag = ("apost", L) if sub == "attn" else ("mpost", L + 1)
            ga, gb = golden(a.a, tag, key), golden(a.b, tag, key)
            c = chain(a.chain, name)
            if ga is None or c is None:
                continue
            cells = []
            for tag_, x, y in (("cA", c, ga), ("cB", c, gb), ("AB", ga, gb)):
                if y is None:
                    cells.append("     --          ")
                    continue
                r = rr(x, y)
                tot[tag_].append(r)
                cells.append(f"{r:.4f}/{cos(x, y):+.5f}")
            print(f"  {name:8s} " + " ".join(cells))
    print()
    for tag_, label in (("cA", "chain vs SGLang A"), ("cB", "chain vs SGLang B"), ("AB", "SGLang vs itself")):
        v = tot[tag_]
        if v:
            print(f"  {label:18s} mean {np.mean(v):.4f}   median {np.median(v):.4f}   worst {max(v):.4f}")


if __name__ == "__main__":
    main()
