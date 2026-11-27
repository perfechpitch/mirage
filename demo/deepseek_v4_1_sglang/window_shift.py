"""Which 128 keys does L0's attention attend? Sweep the window's absolute positions against the kernel's own `o_q`.

The evidence that makes this the right question: at L0 the chain's x matches SGLang's normed embedding to 1.7e-3, every
weight is byte-identical to the checkpoint, `attn_nodes.py` shows x_q/qr_q/oa_q bit-identical and all five GEMM nodes at
the fp32 floor -- and yet the kernel's `o` is 21-30% from ANY replay, while SGLang's own `o` is 0.96% from fp64. With q
and k(right) verified, the only input left that can move `o` that far is WHICH pool rows are read: L0 is `swa`, a
128-slot ring, and slot -> absolute position is a table the kernel and the reference derive independently.

`attn_core.py` already rules out the softmax shape (the official's two-block online softmax and a one-pass max give the
same 0.216 against the kernel), so it is not the reduction order.

k=0 is the reference's own mapping. A k whose replay lands near 1e-3 identifies an off-by-k in the ring/position table.

  window_shift.py --layer 0 --dump /tmp/bit_L0 [--kmax 6] [--pos 31906]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import attn64 as A                                            # noqa: E402

T = 8
W = 128


def kern(dump, name, width):
    a = np.fromfile(os.path.join(dump, f"{name}.bin"), dtype=np.int16).view(np.uint16)
    m = a.reshape(T, -1)
    assert m.shape[1] == width and (m[1:] == 0).all(), f"{name}: {m.shape}"
    return m[0].astype(np.uint32).__lshift__(16).view(np.float32).astype(np.float64)


def pool_rows(L, pos):
    """All W key rows of the window, oldest first, exactly as attn64.pool_row builds them (None = current token)."""
    return A.pool_row(L, pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--pos", type=int, default=31906)
    ap.add_argument("--dump", default="/tmp/bit_L0")
    ap.add_argument("--kmax", type=int, default=6)
    a = ap.parse_args()
    L, pos = a.layer, a.pos
    p = f"layers.{L}.attn."

    x = kern(a.dump, f"attn{L}_x", A.HID)
    o_k = kern(a.dump, f"attn{L}_r0_o_q", 8 * A.D)        # rank 0: 8 of the 64 heads

    # everything weight-side, once
    wqa, wqb, wkv = A.deq(p + "wq_a"), A.deq(p + "wq_b"), A.deq(p + "wkv")
    qn, kn = A.read(p + "q_norm.weight").double().numpy(), A.read(p + "kv_norm.weight").double().numpy()
    sink = A.read(p + "attn_sink").double().numpy()
    print(f"\nL{L}: kernel |o_q| rms {np.sqrt((o_k ** 2).mean()):.5f}   chain |x| rms {np.sqrt((x ** 2).mean()):.5f}")

    qr = A.rms(wqa @ x, qn)
    q = A.rope((wqb @ qr).reshape(A.HEADS, A.D), pos)
    k_now = A.act_quant(A.rope(A.rms(wkv @ x, kn), pos), 32)

    rows = pool_rows(L, pos)
    print(f"     window rows: {len(rows)}, current-token slot at index "
          f"{[i for i, r in enumerate(rows) if r is None][:3]}")
    kv0 = np.stack([k_now if r is None else r for r in rows])

    def replay(kv):
        s = (q @ kv.T) / np.sqrt(A.D)
        m = s.max(1)
        e = np.exp(s - m[:, None])
        o = (e @ kv) / (e.sum(1) + np.exp(sink - m))[:, None]
        return A.rope(o, pos, inverse=True)[:8].reshape(-1)

    print(f"\n  {'k':>4}  {'relRMS vs kernel o_q':>22}")
    res = []
    for k in range(-a.kmax, a.kmax + 1):
        if k == 0:
            kv = kv0
        else:
            idx = np.clip(np.arange(W) + k, 0, W - 1)     # slide the looked-up positions, clamp at the ends
            kv = kv0[idx]
        o = replay(kv)
        r = float(np.sqrt(((o - o_k) ** 2).mean()) / np.sqrt((o_k ** 2).mean()))
        res.append((r, k))
        print(f"  {k:>4}  {r:>22.5f}")
    res.sort()
    print(f"\n  best k = {res[0][1]}, relRMS {res[0][0]:.5f}   (k = 0 is the reference's own mapping)")
    print(f"  k=0 relRMS {dict((kk, rr) for rr, kk in res)[0]:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
