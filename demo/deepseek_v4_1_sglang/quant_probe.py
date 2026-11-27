"""Which quantization accounts for the MoE kernel's 3% distance from fp64 checkpoint math?

The kernel builds its experts through `act_quant` twice (see the launch plan's `act_quant#9` / `act_quant#15`):
  * x -> fp8 e4m3, one power-of-two scale per 32 columns (tasks/act_quant.cuh, ue8m0, satfinite at +-448)
  * gate/up rounded to bf16, silu(gate)*clamp(up) computed, then the product -> fp8 e4m3 the same way, into w2
So a fp64 reference that skips them is measuring the quantization, not the kernel. This script tries the four
combinations and reports which reproduces the kernel's own y.

    python quant_probe.py --layer 1 --chain /tmp/chain_fresh
"""
import argparse

import numpy as np
import torch

from moe_expert_ref import CK_, H, expert_raw, gate_of, kernel_y_of, load_x_and_residual


def _rne_e4m3(v):
    """Round float32 to e4m3 (1 sign, 4 exp, 3 mantissa), RNE, satfinite at +-448, as cvt_e4m3_nv."""
    a = np.abs(v).astype(np.float32)
    s = np.sign(v)
    out = np.zeros_like(a, dtype=np.float64)
    # normal range: 2^-6 <= |v| < 2^9 (448)
    live = a > 0
    if live.any():
        e = np.floor(np.log2(np.maximum(a, 1e-30))).astype(np.int64)
        # clamp into the representable exponent band (subnormals below 2^-6)
        e = np.clip(e, -6, 8)
        scale = np.exp2(e).astype(np.float64)
        m = (a / scale).astype(np.float64)          # in [1, 2)
        # round the 3-bit fraction, ties to even
        q = np.rint(m * 8.0) / 8.0
        up = q >= 2.0
        q = np.where(up, q / 2.0, q)
        scale = np.where(up, scale * 2.0, scale)
        out = s.astype(np.float64) * q * scale
    # subnormal handling: anything under 2^-6 snaps to a multiple of 2^-9
    sub = live & (a < 2.0 ** -6)
    if sub.any():
        out[sub] = s[sub].astype(np.float64) * np.rint(a[sub] / 2.0 ** -9) * 2.0 ** -9
    return np.clip(out, -448.0, 448.0)


def act_quant(x):
    """Per-32-column power-of-two scale, then e4m3: the reference's round trip (quantise and dequantise)."""
    v = np.asarray(x, np.float64).reshape(-1, 32)
    amax = np.maximum(np.abs(v).max(1), 1e-4)
    ex = np.ceil(np.log2(amax / 448.0))
    scaled = v / np.exp2(ex)[:, None]
    return (_rne_e4m3(scaled) * np.exp2(ex)[:, None]).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--chain", default="/tmp/chain_fresh")
    a = ap.parse_args()
    L = a.layer
    ck = CK_

    x, rin, post, comb, so = load_x_and_residual(a.chain, L)
    kernel_y = kernel_y_of(so, rin, post, comb)
    gw = ck(f"layers.{L}.ffn.gate.weight").float().numpy().astype(np.float64)
    gb = ck(f"layers.{L}.ffn.gate.bias").float().numpy().astype(np.float64)
    eid, wgt = gate_of(x, gw, gb)
    print(f"layer {L}: experts {eid.tolist()}")

    def rr(u, v):
        u, v = np.ravel(u), np.ravel(v)
        return float(np.linalg.norm(u - v) / max(np.linalg.norm(v), 1e-30))

    def cos(u, v):
        u, v = np.ravel(u), np.ravel(v)
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))

    def build(qx, qa):
        routed = np.zeros(H)
        for e, wv in zip(eid, wgt):
            routed += wv * expert_raw(ck, L, int(e), x, quant_x=qx, quant_act=qa)
        sh = expert_raw(ck, L, None, x, quant_x=qx, quant_act=qa)
        return routed + sh

    print(f"{'x quant':>8s} {'act quant':>10s} {'relRMS vs kernel y':>20s} {'cos':>11s}")
    for qx in (False, True):
        for qa in (False, True):
            out = build(qx, qa)
            print(f"{str(qx):>8s} {str(qa):>10s} {rr(out, kernel_y):>20.5f} {cos(out, kernel_y):>+11.7f}")


if __name__ == "__main__":
    main()
