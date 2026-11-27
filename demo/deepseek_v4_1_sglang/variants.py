"""Which of the plausible implementation variants reproduces the kernel's y? (the last few % of the MoE)

With the two act_quant stages reproduced, the kernel y at layer 1 lands 1.95% from the checkpoint math -- and the
deconvolution instrument's own floor is only 0.12% (floor.py), so that 1.95% is real arithmetic, not measurement.
The candidates that can move a MoE by ~1%:

  * where the routing weight is applied: w * quantize(h)  vs  quantize(w * h)   (before or after w2)
  * whether the GEMM output is rounded to bf16 before the SwiGLU (the kernel's act_quant_swiglu calls bf16r)
  * clamp-before-round vs round-before-clamp (the kernel rounds first: model.py's w1 returns bf16)
  * whether w2's output is rounded to bf16 (sum_store's input is fp32, so it should not be)

    python variants.py --layer 1 --chain /tmp/chain_q
"""
import argparse

import numpy as np
import torch

from moe_expert_ref import (CK_, H, SWIGLU, _expand, act_quant, cos, gate_of, kernel_y_of,
                            load_x_and_residual, rr, unpack_fp4)


def expert_v(ck, L, e, x, *, quant_x=True, bf16_io=True, round_first=True, q_after_w=False, w=1.0,
             bf16_w2_out=False):
    q = f"layers.{L}.ffn." + ("shared_experts." if e is None else f"experts.{e}.")
    w1, w3, w2 = ck(q + "w1.weight"), ck(q + "w3.weight"), ck(q + "w2.weight")
    U = unpack_fp4 if w1.dtype == torch.int8 else (lambda t, s: t.double().numpy() * _expand(s, *t.shape))
    xi = act_quant(x) if quant_x else np.asarray(x, np.float64)
    g = xi @ U(w1, ck(q + "w1.scale")).T
    u = xi @ U(w3, ck(q + "w3.scale")).T
    if bf16_io:
        g = torch.from_numpy(g).float().bfloat16().float().numpy().astype(np.float64)
        u = torch.from_numpy(u).float().bfloat16().float().numpy().astype(np.float64)
    if round_first:                      # tasks/act_quant.cuh: bf16r(gate) then fminf(., lim)
        g = np.minimum(g, SWIGLU)
        u = np.clip(u, -SWIGLU, SWIGLU)
    else:                                # model.py Expert.forward: clamp the fp32, no bf16 round
        u = np.clip(u, -SWIGLU, SWIGLU)
        g = np.minimum(g, SWIGLU)
    h = (g / (1.0 + np.exp(-g))) * u
    if q_after_w:                        # the weight folded in before the fp8 quantization
        h = act_quant(w * h)
    else:
        h = act_quant(h)
    out = h @ U(w2, ck(q + "w2.scale")).T
    if bf16_w2_out:
        out = torch.from_numpy(out).float().bfloat16().float().numpy().astype(np.float64)
    return out if q_after_w else w * out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--chain", default="/tmp/chain_q")
    ap.add_argument("--all-layers", action="store_true")
    a = ap.parse_args()
    ck = CK_
    layers = range(40) if a.all_layers else [a.layer]

    VARIANTS = {
        "base (w*quant(h), bf16, round-first)": dict(),
        "quantize(w*h) (weight inside fp8)": dict(q_after_w=True),
        "no bf16 round of g/u": dict(bf16_io=False),
        "clamp before bf16 round": dict(round_first=False),
        "bf16 round w2's output": dict(bf16_w2_out=True),
    }
    tot = {k: [] for k in VARIANTS}
    for L in layers:
        try:
            x, rin, post, comb, so = load_x_and_residual(a.chain, L)
        except FileNotFoundError:
            continue
        gw = ck(f"layers.{L}.ffn.gate.weight").float().numpy().astype(np.float64)
        gb = ck(f"layers.{L}.ffn.gate.bias").float().numpy().astype(np.float64)
        eid, wgt = gate_of(x, gw, gb)
        y = kernel_y_of(so, rin, post, comb)
        line = f"L{L:>2}: "
        for name, kw in VARIANTS.items():
            ref = np.zeros(H)
            for e, w in zip(eid, wgt):
                ref += expert_v(ck, L, int(e), x, w=w, **kw)
            ref += expert_v(ck, L, None, x, **kw)
            r = rr(y, ref)
            tot[name].append((r, cos(y, ref)))
            line += f"{r:.5f} "
        print(line + "| " + " | ".join(n.split(" ")[0] for n in VARIANTS))
    print("\n" + "=" * 100)
    for name in VARIANTS:
        if tot[name]:
            rs = np.array([t[0] for t in tot[name]])
            cs = np.array([t[1] for t in tot[name]])
            print(f"{name:<40} relRMS mean {rs.mean():.5f} median {np.median(rs):.5f} max {rs.max():.5f}"
                  f"   cos min {cs.min():+.7f}  >=.9999 on {int((cs >= 0.9999).sum())}/{len(cs)}")


if __name__ == "__main__":
    main()
