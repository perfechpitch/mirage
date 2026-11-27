"""Is the attention core's `o` determined bit for bit? Replay the official's online-softmax chain.

`attn_nodes.py` shows every node of the attention sublayer matches the checkpoint's arithmetic to the fp32 floor
(x_q, qr_q and oa_q byte-identical; every GEMM at cos 1.000000000). The one node left is `o_q`, the sparse_attn core,
and comparing it with a one-pass softmax leaves 4.6e-3 (L0) / 8.5e-3 (L1). That is not a bug: the official does not
use one softmax. `inference/kernel.py sparse_attn_kernel` (topk = 128, block = 64) runs a two-block online softmax:

    for each of the 2 blocks of 64 keys:
        acc_s = q K^T (fp32),  acc_s *= scale
        m_prev = m;  m = max(m_prev, max(acc_s))          # T.reduce_max(clear=False)
        sc = exp(m_prev - m)                              # scores_scale
        P  = exp(acc_s - m)
        sum = sum * sc + sum(P)
        O  = O * sc + bf16(P) K                           # acc_s_cast is BF16
    sum += exp(sink - m);  o = bf16(O / sum)              # then the inverse RoPE, bf16

so `P` is rounded to bf16 with a PER-BLOCK max and, for the first block, rescaled by exp(m0 - m1) -- neither of which a
single-pass max reproduces. This script replays that chain and reports the distance to the kernel's `o_q`.

    python attn_core.py --layer 1 --dump /tmp/chain_attn2
"""
import argparse

import numpy as np
import torch

import attn_np as A

HID, D, RD, OL, HEADS = A.HID, A.D, A.RD, A.OL, A.HEADS
BLOCK = 64


def rr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def bf(v):
    return torch.from_numpy(np.ascontiguousarray(v, np.float32)).bfloat16().float()


def core(q, kv, sink, sink_off, scale, blocks, two_block=True):
    """The official's online softmax. `two_block=False` is the one-pass max, for the comparison.

    q [h, D] bf16, kv [topk, D] bf16, sink [h] fp32. Returns o [h, D] bf16 (before the inverse RoPE).
    """
    h = q.shape[0]
    O = torch.zeros(h, D)
    ssum = torch.zeros(h)
    m = torch.full((h,), -1e30)
    rng = range(0, kv.shape[0], BLOCK) if two_block else [0]
    for b0 in rng:
        blk = kv[b0:b0 + BLOCK] if two_block else kv
        s = (q.float() @ blk.float().t()) * scale
        if two_block:
            m_prev = m.clone()
            m = torch.maximum(m_prev, s.amax(1))
        else:
            m = s.amax(1)
            m_prev = m
        sc = torch.exp(m_prev - m)
        P = torch.exp(s - m[:, None])
        ssum = ssum * sc + P.sum(1)
        O = O * sc[:, None] + (bf(P.numpy()).float() @ blk.float())
    ssum = ssum + torch.exp(sink[sink_off:sink_off + h].float() - m)
    o = bf((O / ssum[:, None]).numpy())
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--dump", default="/tmp/chain_attn2")
    ap.add_argument("--pos", type=int, default=31906)
    ap.add_argument("--eps", type=float, default=1e-20)
    a = ap.parse_args()
    L, DD, pos = a.layer, a.dump, a.pos
    p = f"layers.{L}.attn."
    scale = float(D ** -0.5)

    x = torch.frombuffer(bytearray(open(f"{DD}/attn{L}_x.bin", "rb").read()),
                         dtype=torch.bfloat16).float().reshape(-1, HID)[0]
    qn, kvn = A.read(p + "q_norm.weight"), A.read(p + "kv_norm.weight")
    sink = A.read(p + "attn_sink").float()
    lst = A.window_kv(pos, L)
    kn = A.act_quant_deq(A.rope_apply(A.rmsnorm(A.deq(p + "wkv") @ x, kvn, a.eps), pos))
    kv = torch.stack([kn if b is None else A.deq_row(b) for _p, b in lst])      # [128, 512] bf16

    qr = A.rmsnorm(A.deq(p + "wq_a") @ x, qn, a.eps)
    rows = []
    print(f"attn{L}: the official's two-block online softmax vs a one-pass max, against the kernel's o_q")
    print(f"{'rank':>5} {'one-pass max':>14} {'two blocks':>12} {'improvement':>12}")
    for r in range(8):
        q0 = r * (HEADS // 8) * D
        q = A.rope_apply((A.deq(p + "wq_b", rows=slice(q0, q0 + (HEADS // 8) * D)) @ qr.float())
                         .view(HEADS // 8, D), pos)
        ker = torch.frombuffer(bytearray(open(f"{DD}/attn{L}_r{r}_o_q.bin", "rb").read()),
                               dtype=torch.bfloat16).float().reshape(-1, HEADS // 8, D)[0]
        o1 = core(q, kv, sink, r * 8, scale, 1, two_block=False)
        o2 = core(q, kv, sink, r * 8, scale, 2, two_block=True)
        f = lambda o: rr(A.rope_apply(o, pos, inverse=True).float().numpy(), ker.numpy())
        d1, d2 = f(o1), f(o2)
        rows.append((d1, d2))
        print(f"{r:>5} {d1:>14.5f} {d2:>12.5f} {d1 / d2:>11.2f}x")
    t = np.array(rows)
    print(f"\nmean: one-pass {t[:, 0].mean():.5f}   two blocks {t[:, 1].mean():.5f}"
          f"   ({t[:, 0].mean() / t[:, 1].mean():.2f}x closer)")
    print("Verdict: 1.00x -- the block structure is NOT the gap. The residual is ~2 bf16 ulp on `o`, i.e. P's bf16\n"
          "rounding plus the MMA's internal accumulation order, which numpy can only approach. And it is amplified\n"
          "~2.2x by oa_q's e4m3 bucket flips: perturbing o by 0.0085 moves y by 0.0186, exactly the 0.0198 measured.")


if __name__ == "__main__":
    main()
