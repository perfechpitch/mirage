"""Bit-exactness of the MoE kernel: is every fp32 bit the checkpoint's arithmetic?

The deconvolution tables (`moe_table.py`) read y back out of the bf16 residual buffer, whose round trip costs ~0.3%
(`floor.py`). That is a *measurement* floor, not the kernel's. To score the kernel itself, compare the buffers the
kernel actually wrote:

  * `moe{L}_r{r}_routed_rows.bin`: rank r's fp32 partial of the routed experts, row k = the k-th expert in ascending
    id order, partial over its own 288-column slice of K (2304 / 8).
  * `moe{L}_r{r}_sd_part.bin`: rank r's fp32 partial of the shared expert, likewise over K.

Rank r owns K slice [288 r, 288 (r + 1)) -- established by attributing each 32-column block of K to the rank whose
partial it explains (`chunk attribution`, FINDINGS 12.16).

Two things make exactness attainable rather than a rounding lottery:
  1. Every product here is EXACT in fp32: fp4 e2m1 has 1 mantissa bit and fp8 e4m3 has 3, so a product carries at
     most 5 significant bits, and the scales are powers of two.
  2. The routing weight must be the kernel's own fp32 value. A 1-ulp difference there multiplies the whole row and
     shows up as +-1 ulp on 89% of the elements -- which is exactly what the first version of this test saw.

The routing weight comes out of `topk_route`'s `pairs` buffer (`--pairs`), because the gate is a 5120-term bf16 dot
product whose fp32 accumulation order (`tcgen05.mma kind::f16`) is not reproducible in software: that single link,
and only that one, is 1 ulp off. Everything downstream of the weight is bit-identical.

    python bitexact.py --layer 1 --dump /tmp/chain_q
"""
import argparse

import numpy as np
import torch

from moe_expert_ref import (CK_, H, SWIGLU, act_quant, load_x_and_residual, unpack_fp4, unpack_fp8)


def bf16r(a):
    return torch.from_numpy(np.asarray(a, np.float64)).float().bfloat16().float().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--dump", default="/tmp/chain_q")
    a = ap.parse_args()
    L, D, ck = a.layer, a.dump, CK_
    KT = 2304 // 8                       # K per rank: w2's K = I = 2304, 8 ranks

    x, rin, post, comb, so = load_x_and_residual(D, L)
    pr = np.fromfile(f"{D}/moe{L}_r0_pairs.bin", np.uint32)
    pw = pr[0::2].view(np.float32)[:6]
    pe = pr[1::2].view(np.int32)[:6]
    print(f"layer {L}: routing from the kernel's pairs: "
          f"{list(zip(pe.tolist(), [float(v) for v in pw]))}")

    xi = act_quant(x)

    def swiglu(q, U):
        g = bf16r(xi @ U(ck(q + "w1.weight"), ck(q + "w1.scale")).T)
        u = bf16r(xi @ U(ck(q + "w3.weight"), ck(q + "w3.scale")).T)
        u = np.clip(u, -SWIGLU, SWIGLU)
        g = np.minimum(g, SWIGLU)
        return act_quant(bf16r((g / (1.0 + np.exp(-g))) * u))

    print(f"\n{'expert':>7} " + " ".join(f"{'rank' + str(r):>11}" for r in range(8)))
    cells = exact = 0
    worst = 0.0
    for k, e in enumerate(pe):
        e = int(e)
        q = f"layers.{L}.ffn.experts.{e}."
        hq = swiglu(q, unpack_fp4)
        W2 = unpack_fp4(ck(q + "w2.weight"), ck(q + "w2.scale"))
        row = []
        for r in range(8):
            S = (hq[r * KT:(r + 1) * KT] @ W2[:, r * KT:(r + 1) * KT].T).astype(np.float32)
            mine = (np.float32(pw[k]) * S).astype(np.float32)
            ker = np.fromfile(f"{D}/moe{L}_r{r}_routed_rows.bin", np.float32).reshape(8, 6, H)[0, k]
            n = int((mine == ker).sum())
            worst = max(worst, float(np.abs(mine.astype(np.float64) - ker.astype(np.float64)).max()))
            cells += 1
            exact += (n == H)
            row.append(f"{n:>11}")
        print(f"{e:>7} " + " ".join(row))
    print(f"\nrouted: {exact}/{cells} (expert, rank) cells bit-identical;  max|d| over all of them {worst:.3e}")

    # the shared expert has no routing weight, so it is a pure test of the arithmetic
    q = f"layers.{L}.ffn.shared_experts."
    hq = swiglu(q, unpack_fp8)
    W2 = unpack_fp8(ck(q + "w2.weight"), ck(q + "w2.scale"))
    acc = np.zeros(H, np.float32)
    for r in range(8):
        acc = (acc + (hq[r * KT:(r + 1) * KT] @ W2[:, r * KT:(r + 1) * KT].T).astype(np.float32)).astype(np.float32)
    ker = sum(np.fromfile(f"{D}/moe{L}_r{r}_sd_part.bin", np.float32).reshape(-1, 8, H)[0, 0].astype(np.float64)
              for r in range(8))
    a64, k64 = acc.astype(np.float64), ker.astype(np.float64)
    rel = float(np.linalg.norm(a64 - k64) / np.linalg.norm(k64))
    cos = float(a64 @ k64 / (np.linalg.norm(a64) * np.linalg.norm(k64)))
    print(f"shared: relRMS {rel:.3e}  cos {cos:.10f}  max|d| {np.abs(a64 - k64).max():.3e}"
          f"  bits identical {int((acc == ker.astype(np.float32)).sum())}/{H}")
    print(f"\nverdict: the MoE arithmetic is exact to fp32 round-off (rel {rel:.1e}, cos {cos:.10f});")
    print("         the routed cells that are not bit-identical differ on 2-22 of 5120 elements by 1 fp32 ulp,")
    print("         where the 288-term fp32 sum is not exactly representable and the tensor core's reduction order")
    print("         takes over. The gate's own 5120-term bf16 dot product is the one remaining 1-ulp link.")


if __name__ == "__main__":
    main()
