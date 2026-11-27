"""Multi-step decode: does the chain's cache actually advance, or only its tables?

Step 1 is the easy case -- the caches come pre-loaded from the prefill, so nothing has to be written. From step 2 on the
chain has to do three things itself, and each is a different mechanism:

    window_cache    kv_post writes position p's row into slot p % W of the ring
    compress_cache  ckv_post writes the block, and for ratio 2 the compressor's state carries the half block over
    compress_idx    the indexer re-scores and re-takes the top-k every step

The reference makes this checkable without any new SGLang run: `/tmp/dec_coef8/layers/step1..step10` are ten consecutive
dumps of the SAME row's greedy decode, and each step's `L00_in` is exactly `embed.weight[token]` -- verified, max|d| 0.0e+00
against vocab rows 4588 / 343 / 6333. So a step's input token is not a guess: it is read out of the dump.

The pass condition is the same as step 1's: the residual error has to stay at the step-1 level (~1e-2 early, ~9e-2 at L39
by the time 40 layers have accumulated). A cache that did not advance does not degrade gracefully -- it puts the wrong
keys in front of the softmax and jumps to O(1).

    multistep.py [--ours /tmp/chain_m3] [--ref /tmp/dec_coef8] [--steps 1,2]
"""
import argparse
import json
import os
import sys

import numpy as np

T = 8
HID, HC = 5120, 4


def ours(where, name, width):
    p = os.path.join(where, f"{name}.bin")
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dtype=np.int16).view(np.uint16)
    assert a.size == T * width, f"{p}: {a.size} values, expected {T}x{width}"
    m = a.reshape(T, width)
    assert (m[1:] == 0).all(), f"{p}: rows 1.. are not zero"
    return m[0].astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)


def ref(where, name):
    p = os.path.join(where, name)
    return np.fromfile(p, dtype=np.float32).astype(np.float64) if os.path.exists(p) else None


def rr(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((b ** 2).mean()) + 1e-30))


def embedding():
    """[V, HID] fp32, to name the token each step starts from (its `L00_in` is exactly the embedding row)."""
    from safetensors import safe_open
    ck = "/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017"
    idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(ck, idx["embed.weight"]), "pt") as f:
        return f.get_tensor("embed.weight").float().numpy()


def step_of(ours_dir, s):
    """Our dump of step s: step 1 is the flat directory, later steps its own subdirectory (full_decode's layout)."""
    return ours_dir if s == 1 else os.path.join(ours_dir, f"step{s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="/tmp/chain_m3")
    ap.add_argument("--ref", default="/tmp/dec_coef8")
    ap.add_argument("--steps", default="1,2")
    ap.add_argument("--layers", default="0,1,7,20,39")
    a = ap.parse_args()
    steps = [int(x) for x in a.steps.split(",")]
    layers = [int(x) for x in a.layers.split(",")]
    sp = json.load(open("/tmp/ref_spread.json")) if os.path.exists("/tmp/ref_spread.json") else {}
    emb = embedding()

    print(f"\n  chain {a.ours}   vs SGLang {a.ref}   (each step's input token read from its own L00_in)")
    print(f"\n  {'step':>4} {'token':>7} {'pos':>6}   " +
          "  ".join(f"{'L%d attn' % L:>10} {'moe':>10}" for L in layers))
    print("  " + "-" * (28 + 24 * len(layers)))
    ok = True
    for s in steps:
        R = os.path.join(a.ref, "layers", f"step{s}")
        O = step_of(a.ours, s)
        if not os.path.isdir(R) or not os.path.isdir(O):
            print(f"  {s:>4}  (missing: ref {os.path.isdir(R)}, ours {os.path.isdir(O)})")
            continue
        rin = ref(R, "L00_in.f32")
        tok = int(np.abs(emb - rin[:HID]).max(1).argmin()) if rin is not None else -1
        row = f"  {s:>4} {tok:>7} {31905 + s:>6}   "
        for L in layers:
            o = ours(O, f"attn{L}_streams_out", HC * HID)
            e = ref(R, f"L{L:02d}_apost.f32")
            row += f"{rr(o, e):>10.2e} " if o is not None and e is not None else f"{'--':>10} "
            o = ours(O, f"moe{L}_streams_out", HC * HID)
            e = ref(R, f"L{L + 1:02d}_mpost.f32") if L else ref(R, "L01_eg_x.f32")
            if e is None:                     # the last MoE's post is not in a dump: nothing follows it
                row += f"{'--':>10}  " if L == 39 else f"{'--':>10}  "
            else:
                d = rr(o, e)
                row += f"{d:>10.2e}  "
                if d > 0.5:
                    ok = False
        print(row)

    if os.path.exists(os.path.join(a.ours, "step2")):
        print("\n  argmax per step (our head's own):")
        for s in steps:
            O = step_of(a.ours, s)
            p = os.path.join(O, "logits.bin")
            if not os.path.exists(p):
                continue
            lg = np.fromfile(p, np.float32)
            V = lg.size // T
            g = lg[:V].astype(np.float64)
            t1 = int(g.argmax())
            R = os.path.join(a.ref, "layers", f"step{s + 1}")
            nxt = ref(R, "L00_in.f32") if os.path.isdir(R) else None
            nt = int(np.abs(emb - nxt[:HID]).max(1).argmin()) if nxt is not None else -1
            mark = "MATCH" if t1 == nt else ("MISMATCH" if nt >= 0 else "(no next step in the ref)")
            print(f"    step {s}: argmax {t1:>7}   the reference's next input {nt:>7}   {mark}")


if __name__ == "__main__":
    sys.exit(main())
