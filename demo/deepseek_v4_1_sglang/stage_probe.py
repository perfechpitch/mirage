"""Locate the MoE kernel's remaining error stage by stage, byte-exactly.

The kernel's MoE is four stages, and every one of them leaves a buffer behind:

  1. act_quant      x -> x_q            (tasks/act_quant.cuh, ue8m0 per 32 columns)      dump: {L}_r0_x_q.bin
  2. topk_route     x -> pairs          (the 6 (weight, expert) pairs)                   dump: {L}_r0_pairs.bin
  3. gemm_fp8_tile  x_q x w13 -> sgu_part, then act_quant_swiglu -> h_q                 dump: {L}_r0_swiglu_moe_experts.h_q[.bin|_scales.bin]
  4. gemm_fp8_tile  h_q x w2 -> routed_rows                                            dump: {L}_r0_routed_rows.bin

Stages 1 and 2 are checked byte-for-byte; stage 3 is too (h_q is fp8, so an exact implementation reproduces the bytes).
Whatever survives is stage 4. This is what tells a wrong dequant from a rounding-order difference instead of guessing.

    python stage_probe.py --layer 1 --dump /tmp/chain_q
"""
import argparse

import numpy as np
import torch

from moe_expert_ref import (CK_, H, INTER, SWIGLU, _expand, _rne_e4m3, act_quant, gate_of,
                            load_x_and_residual, unpack_fp4)

SLOTS, IRP = 8, INTER           # static_megakernel.py: NSLOT, IRP = I padded to whole 128 tiles (2304 already)


def e4m3_to_f(b):
    b = np.asarray(b).astype(np.int64)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 0x7
    return s * np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7))


def f32_to_e4m3_bytes(v):
    """The inverse of e4m3_to_f for values already representable as e4m3 (what act_quant produces).

    The mantissa carry matters: 252 has ef = 7 but rounds to m = 8, which is the next exponent (256), not m = 7 (240).
    """
    v0 = np.asarray(v, np.float64)
    a = np.abs(v0)
    b = np.zeros(a.shape, np.uint8)
    nz = a > 0
    if not nz.any():
        return b
    ef = np.zeros(a.shape, np.int64)
    m = np.zeros(a.shape, np.int64)
    e = np.zeros(a.shape, np.int64)
    sub = nz & (a < 2.0 ** -6)          # e4m3 subnormals: step 2^-9, e = 0
    nrm = nz & ~sub
    if nrm.any():
        ef[nrm] = np.floor(np.log2(a[nrm])).astype(np.int64)
        m8 = np.rint((a[nrm] / np.exp2(ef[nrm]) - 1.0) * 8.0).astype(np.int64)
        carry = m8 >= 8                 # rounds up into the next binade
        e[nrm] = ef[nrm] + 7 + carry
        m[nrm] = np.where(carry, 0, m8)
    if sub.any():
        e[sub] = 0
        m[sub] = np.rint(a[sub] / 2.0 ** -9).astype(np.int64)
    sat = e > 15                        # satfinite: clamp to 448 (e = 15, m = 6)
    e = np.where(sat, 15, e)
    m = np.where(sat, 6, m)
    m = np.clip(m, 0, 7)
    b = ((v0 < 0).astype(np.uint8) << 7) | ((e.astype(np.uint8) & 0xF) << 3) | m.astype(np.uint8)
    b[~nz] = 0
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--dump", default="/tmp/chain_q")
    ap.add_argument("--chain", default=None, help="where moe{L}_x.bin lives (default: --dump)")
    a = ap.parse_args()
    L, D = a.layer, a.dump
    ch = a.chain or a.dump
    ck = CK_

    x, rin, post, comb, so = load_x_and_residual(ch, L)
    gw = ck(f"layers.{L}.ffn.gate.weight").float().numpy().astype(np.float64)
    gb = ck(f"layers.{L}.ffn.gate.bias").float().numpy().astype(np.float64)
    eid, wgt = gate_of(x, gw, gb)

    print(f"layer {L}: my routing {eid.tolist()} {np.round(wgt, 5).tolist()}")
    pr = np.fromfile(f"{D}/moe{L}_r0_pairs.bin", np.uint32)
    kp = np.stack([pr[0::2].view(np.float32), pr[1::2].view(np.int32)], 1)[:6]
    print(f"           kernel pairs {kp[:, 1].astype(int).tolist()} {np.round(kp[:, 0], 5).tolist()}")
    print(f"  stage 2 topk_route: sets equal = {sorted(eid.tolist()) == sorted(kp[:, 1].astype(int).tolist())}"
          f"   weights equal = {np.allclose(np.sort(wgt), np.sort(kp[:, 0]), atol=2e-7)}")

    # --- stage 1: x_q bytes
    xq = np.fromfile(f"{D}/moe{L}_r0_x_q.bin", np.uint8)
    q = xq[:8 * H].reshape(8, H)
    v = np.asarray(x, np.float64).reshape(-1, 32)
    ex = np.ceil(np.log2(np.maximum(np.abs(v).max(1), 1e-4) / 448.0))[:, None]
    mine = f32_to_e4m3_bytes((v / np.exp2(ex)).ravel())
    print(f"  stage 1 act_quant:  byte-identical = {np.array_equal(mine, q[0])}")

    # --- stage 3: w13 GEMM + swiglu + act_quant -> h_q (fp8, so exact means equal bytes)
    hq = np.fromfile(f"{D}/moe{L}_r0_swiglu_moe_experts.h_q.bin", np.uint8).reshape(SLOTS, 8, IRP)
    hsf = np.fromfile(f"{D}/moe{L}_r0_swiglu_moe_experts.h_q_scales.bin", np.uint8).reshape(SLOTS, IRP // 128, 512)

    def swiglu_fp8(e):
        qp = f"layers.{L}.ffn.experts.{e}."
        w1, w3 = ck(qp + "w1.weight"), ck(qp + "w3.weight")
        xq_d = (e4m3_to_f(f32_to_e4m3_bytes(act_quant(x))))
        # dequantized fp8 activations x the fp4 weights, exactly as the kernel's MMA sees them
        g = xq_d @ unpack_fp4(w1, ck(qp + "w1.scale")).T
        u = xq_d @ unpack_fp4(w3, ck(qp + "w3.scale")).T
        g = torch.from_numpy(g).float().bfloat16().float().numpy().astype(np.float64)
        u = torch.from_numpy(u).float().bfloat16().float().numpy().astype(np.float64)
        u = np.clip(u, -SWIGLU, SWIGLU)
        g = np.minimum(g, SWIGLU)
        h = (g / (1.0 + np.exp(-g))) * u
        pad = np.zeros(IRP)
        pad[:INTER] = h
        return f32_to_e4m3_bytes(act_quant(pad))

    print(f"  stage 3 w13+swiglu+quant: matching each expert to a slot by bytes")
    used = set()
    for e in eid:
        ref = swiglu_fp8(int(e))
        best = None
        for s in range(SLOTS):
            if s in used:
                continue
            same = int((ref == hq[s][0]).sum())
            if best is None or same > best[1]:
                best = (s, same)
        s, same = best
        used.add(s)
        print(f"    expert {int(e):>3} -> slot {s}: {same}/{IRP} bytes equal ({100.0 * same / IRP:.2f}%)")
        if same < IRP:
            d = np.abs(e4m3_to_f(ref) - e4m3_to_f(hq[s][0]))
            print(f"      max |d| {d.max():.3e}  mean |d| {d.mean():.3e}  ref rms {np.sqrt((e4m3_to_f(ref) ** 2).mean()):.5f}"
                  f"  kernel rms {np.sqrt((e4m3_to_f(hq[s][0]) ** 2).mean()):.5f}")


if __name__ == "__main__":
    main()
