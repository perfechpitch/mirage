"""The official MoE, on the host, so a divergence can be attributed to the gate, the routed experts or the shared one.

Everything here is transcribed from the checkpoint's own reference implementation, not fitted to a dump:

  inference/model.py  Gate.forward      scores = linear(x.float(), W.float()) / gate_temp(1.0)
                                        scores = sqrt(softplus(scores))          (config: score_func=sqrtsoftplus)
                                        indices = (scores + bias).topk(6).indices
                                        weights = scores.gather(indices); /= sum+1e-20 only if norm_topk_prob
                                        weights *= route_scale(1.5)
                      Expert.forward    gate = w1(x).float(); up = w3(x).float()
                                        up = clamp(up, -10, 10); gate = clamp(gate, max=10)   (swiglu_limit=10)
                                        silu(gate) * up -> w2(...)
  inference/convert.py cast_e2m1fn_to_e4m3fn
                                        a byte holds (low nibble = even column, high nibble = odd column);
                                        e2m1 values are [0, .5, 1, 1.5, 2, 3, 4, 6] + sign;
                                        one e8m0 scale per 32 columns (scale[out, col // 32]);
                                        the shared expert is e4m3 with a 32x32 block scale (scale[i//32, j//32]).

`--x` is the sublayer's input row (5120 floats), which for layer L's FFN is what `coef_ref.py` builds:
norm(sum_j pre_j R[j]) * ffn_norm.weight on SGLang's own post-attention residual.

  python moe_ref.py --checkpoint CK --layer 0 --x ffn0_x.f32
  python moe_ref.py ... --pairs chain_pairs.bin --routed chain_routed.bin --shared chain_sd.bin
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

H, I, E, K = 5120, 2304, 384, 6
GATE_TEMP, ROUTE_SCALE, LIMIT = 1.0, 1.5, 10.0
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def _load(ck, keys):
    idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for k in keys:
        with safe_open(os.path.join(ck, idx[k]), "pt") as sf:
            out[k] = sf.get_tensor(k)
    return out


def e8m0(s):
    """F8_E8M0 is a biased power-of-two exponent, no mantissa (convert.py treats it as one)."""
    return torch.pow(2.0, s.view(torch.uint8).float() - 127.0)


def dequant_fp4(w, sc):
    """w: int8 [O, K/2] packed e2m1; sc: e8m0 [O, K/32] -> fp32 [O, K]."""
    u = w.view(torch.uint8)
    lo, hi = u & 0x0F, (u >> 4) & 0x0F
    x = torch.stack([FP4[lo.long()], FP4[hi.long()]], dim=-1).flatten(1)      # low = even col, high = odd col
    x = x * e8m0(sc).repeat_interleave(32, dim=1)
    return x.bfloat16()


def dequant_fp8_block(w, sc):
    """w: e4m3 [O, K]; sc: e8m0 [O/32, K/32] -> fp32 [O, K], one scale per 32x32 tile."""
    return (w.float() * e8m0(sc).repeat_interleave(32, 0).repeat_interleave(32, 1)).bfloat16()


def expert(x, w1, w3, w2):
    """x: [n, H] bf16 -> [n, H] fp32 accumulation, model.py Expert.forward."""
    g = (x @ w1.t()).float()
    u = (x @ w3.t()).float()
    u = torch.clamp(u, -LIMIT, LIMIT)
    g = torch.clamp(g, max=LIMIT)
    h = torch.nn.functional.silu(g) * u
    return h.bfloat16() @ w2.t()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--x", required=True, help="the FFN input row ([5120] f32)")
    ap.add_argument("--pairs", default=None, help="the chain's pairs buffer (top-k indices) to compare")
    ap.add_argument("--routed", default=None, help="the chain's routed rows, summed over the top-k")
    ap.add_argument("--shared", default=None, help="the chain's shared expert output")
    ap.add_argument("--save", default=None, help="write routed+shared (the sublayer's y, fp32 [5120]) here")
    a = ap.parse_args()

    x = torch.from_numpy(np.fromfile(a.x, dtype=np.float32).reshape(-1)[:H]).float()
    # `expert()` is written for a batch ([n, H] -> [n, H]); a 1-D row makes it return a bare [H] and the `[0]` below then
    # takes element 0 instead of row 0 -- silently broadcasting one scalar over the whole `routed` vector. Hence xb 2-D.
    xb = x.bfloat16().reshape(1, H)
    p = f"layers.{a.layer}.ffn."
    w = _load(a.checkpoint, [p + "gate.weight", p + "gate.bias"])
    logits = (x.bfloat16().float() @ w[p + "gate.weight"].float().t()) / GATE_TEMP
    scores = torch.nn.functional.softplus(logits).sqrt()
    idx = (scores + w[p + "gate.bias"]).topk(K).indices
    weights = scores.gather(0, idx)
    # model.py:826 -- ModelArgs.norm_topk_prob defaults to True and this config does not override it, so the top-k weights
    # ARE normalised (to sum 1, then x route_scale). Skipping this made |routed| 3.3x too large.
    weights = weights / (weights.sum() + 1e-20)
    weights *= ROUTE_SCALE
    print(f"gate: x norm {x.norm():.4f}  logits |max| {logits.abs().max():.4f}")
    print(f"  experts (by id) {sorted(idx.tolist())}")
    print(f"  scores          {np.round(scores[idx].numpy(), 5).tolist()}")
    print(f"  weights x1.5    {np.round(weights.numpy(), 5).tolist()}")

    if a.pairs:
        c = np.fromfile(a.pairs, dtype=np.int64).reshape(-1, K)[0]
        same = set(c.tolist()) == set(idx.tolist())
        print(f"  vs chain pairs {sorted(c.tolist())}  {'SAME SET' if same else 'DIFFERENT'}"
              f"  (overlap {len(set(c.tolist()) & set(idx.tolist()))}/{K})")
        for e_new, e_old in ((set(idx.tolist()) - set(c.tolist())), (set(c.tolist()) - set(idx.tolist()))):
            print(f"    only official {sorted(e_new)}   only chain {sorted(e_old)}")

    keys = []
    for e in sorted(set(idx.tolist())):
        keys += [p + f"experts.{e}.w1.weight", p + f"experts.{e}.w1.scale",
                 p + f"experts.{e}.w3.weight", p + f"experts.{e}.w3.scale",
                 p + f"experts.{e}.w2.weight", p + f"experts.{e}.w2.scale"]
    for n in ("w1", "w3", "w2"):
        keys += [p + f"shared_experts.{n}.weight", p + f"shared_experts.{n}.scale"]
    ck = _load(a.checkpoint, keys)

    routed = torch.zeros(H, dtype=torch.float32)
    for j, e in enumerate(sorted(set(idx.tolist()))):
        g = torch.where(idx == e)[0]
        if g.numel() == 0:
            continue
        w1 = dequant_fp4(ck[p + f"experts.{e}.w1.weight"], ck[p + f"experts.{e}.w1.scale"])
        w3 = dequant_fp4(ck[p + f"experts.{e}.w3.weight"], ck[p + f"experts.{e}.w3.scale"])
        w2 = dequant_fp4(ck[p + f"experts.{e}.w2.weight"], ck[p + f"experts.{e}.w2.scale"])
        y = expert(xb, w1, w3, w2)[0]
        routed += weights[g[0]] * y

    sh = {n: dequant_fp8_block(ck[p + f"shared_experts.{n}.weight"], ck[p + f"shared_experts.{n}.scale"])
          for n in ("w1", "w3", "w2")}
    shared = expert(xb, sh["w1"], sh["w3"], sh["w2"])[0]

    if a.save:
        np.array((routed + shared).numpy(), dtype=np.float32).tofile(a.save)
    print(f"\nofficial: |routed| {routed.norm():.4f}  |shared| {shared.norm():.4f}  |sum| {(routed + shared).norm():.4f}")
    for tag, path, ref in (("routed", a.routed, routed), ("shared", a.shared, shared)):
        if not path:
            continue
        v = np.fromfile(path, dtype=np.float32)
        if v.size != H:
            v = v.reshape(-1, H).sum(0)          # a per-expert / per-rank layout: the sum is the comparable quantity
        v = torch.from_numpy(v.astype(np.float32))
        cos = float(v @ ref / (v.norm() * ref.norm() + 1e-30))
        rel = float((v - ref).norm() / (ref.norm() + 1e-30))
        print(f"  {tag:<7s} chain |{v.norm():.4f}|  cos {cos:+.5f}  relRMS {rel:.4f}"
              f"  ratio {float(v.norm() / (ref.norm() + 1e-30)):.4f}")


if __name__ == "__main__":
    main()
