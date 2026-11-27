"""The MoE fidelity table over all 40 layers: kernel y vs the checkpoint's own math, with and without the act_quant stages.

`--from-chain` deconvolves each kernel's y out of its `streams_out`. The instrument's floor is not a guess: build a
known y, form R' = post*y + comb^T R, round it to bf16 (what the residual buffer stores), deconvolve it back, and the
recovered y is off by ~0.0012 (mean) -- `floor.py`. So a sublayer within ~0.001-0.005 of this table is at the instrument,
and anything larger is the kernel's own arithmetic.

    python moe_table.py --chain /tmp/chain_fresh [--quant]
"""
import argparse

import numpy as np

from moe_expert_ref import CK_, H, SWIGLU, act_quant, cos, expert_raw, gate_of, kernel_y_of, load_x_and_residual, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="/tmp/chain_fresh")
    ap.add_argument("--layers", default=",".join(str(i) for i in range(40)))
    ap.add_argument("--quant", action="store_true", help="reproduce the kernel's two act_quant stages")
    ap.add_argument("--kbits", action="store_true", help="act_quant exponent exactly as the kernel's bits")
    a = ap.parse_args()
    ck = CK_

    print(f"MoE kernel y vs checkpoint math on the kernel's own x and routing"
          f"{' (act_quant reproduced)' if a.quant else ' (no act_quant: NOT like-for-like)'}")
    print(f"{'layer':>5} {'|y| rms':>9} {'relRMS':>8} {'cos':>11} {'cos>=.9999?':>12}")
    vals = []
    for L in [int(t) for t in a.layers.split(",")]:
        try:
            x, rin, post, comb, so = load_x_and_residual(a.chain, L)
        except FileNotFoundError:
            print(f"{L:>5}   (no dump)")
            continue
        gw = ck(f"layers.{L}.ffn.gate.weight").float().numpy().astype(np.float64)
        gb = ck(f"layers.{L}.ffn.gate.bias").float().numpy().astype(np.float64)
        eid, wgt = gate_of(x, gw, gb)
        y = kernel_y_of(so, rin, post, comb)
        ref = np.zeros(H)
        for e, w in zip(eid, wgt):
            ref += w * expert_raw(ck, L, int(e), x, quant_x=a.quant, quant_act=a.quant, kbits=a.kbits)
        ref += expert_raw(ck, L, None, x, quant_x=a.quant, quant_act=a.quant, kbits=a.kbits)
        r, c = rr(y, ref), cos(y, ref)
        vals.append((L, r, c))
        print(f"{L:>5} {np.sqrt((y ** 2).mean()):>9.5f} {r:>8.5f} {c:>+11.7f} {'YES' if c >= 0.9999 else '':>12}")
    if vals:
        rs = np.array([v[1] for v in vals])
        cs = np.array([v[2] for v in vals])
        print(f"\n{len(vals)} layers: relRMS mean {rs.mean():.5f} median {np.median(rs):.5f} "
              f"min {rs.min():.5f} max {rs.max():.5f}")
        print(f"           cos   mean {cs.mean():+.7f} min {cs.min():+.7f};  >=0.9999 on {int((cs >= 0.9999).sum())}/{len(cs)}")


if __name__ == "__main__":
    main()
