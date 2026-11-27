"""Can a *perfect* last layer do better?  The logits' error floor, measured instead of argued.

Teacher forcing (§12.25) showed the end-to-end logits error is dominated by the LAST layer's own rounding: with every
residual forced, our logits are 0.3377 off golden (mean |dlogprob|), while the free chain is 0.3851 -- so 40 layers of
accumulation add only 0.047.  The last MoE's `post` is large ([0.84, 0.03, 1.20, 0.65]), so its own bf16 rounding goes
straight into the head.

That raises the only question worth asking: is our layer 39 leaving something on the table, or is 0.34 what ANY
implementation of this layer would give?

This script recomputes layer 39 from the reference's own residual in fp64 (the mHC post stays bf16, as model.py's
`hc_post` does), feeds the result to the head, and reports the same metric.  Three variants:

  fp64, no quant     the checkpoint's math with no quantization at all          -- the "ideal" the kernel is aiming at
  fp64, quant        + the two act_quant stages the kernel really runs          -- a perfect kernel, still quantized
  (the kernel)       0.3377 / 0.3851 from the teach / free runs                 -- for scale

  last_layer_ceiling.py [--layer 39]

If `fp64, quant` lands well below 0.3377 there is real headroom in our layer 39; if it lands at ~0.34 the gap is what
the bf16/fp8 rounding of this layer costs ANY implementation, and the free chain is at its floor.
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

import moe_expert_ref as M

CK = "/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017"
GOLD = "/tmp/dec_coef8/layers/step1"
H, HC = 5120, 4
EPS = 1e-20


def bf16r(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).bfloat16().float().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--golden-jsonl", default="/tmp/topk_1.jsonl")
    ap.add_argument("--ours", default="/tmp/teach_dump_logits.f32", help="the teach run's logits (all residual forced)")
    a = ap.parse_args()
    L = a.layer

    idx = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]

    def w(n):
        with safe_open(os.path.join(CK, idx[n]), "pt") as sf:
            return sf.get_tensor(n)

    f32 = lambda n: np.fromfile(os.path.join(GOLD, n), np.float32)
    pre = f32(f"L{L:02d}_ct_pre.f32").astype(np.float64)
    post = f32(f"L{L:02d}_ct_post.f32").astype(np.float64)
    comb = f32(f"L{L:02d}_ct_comb.f32").astype(np.float64).reshape(HC, HC)
    rin = f32(f"L{L:02d}_apost.f32").astype(np.float64).reshape(HC, H)

    # the FFN's input, exactly as model.py Block.forward: hc_pre (fp32 sum) -> to(bf16) -> norm -> bf16
    y = (pre[:, None] * rin).sum(0)
    g = w(f"layers.{L}.ffn_norm.weight").double().numpy()
    x = y / np.sqrt((y ** 2).mean() + EPS) * g
    x = bf16r(x)
    print(f"layer {L}: |x| rms {np.sqrt((x ** 2).mean()):.5f}")

    gw = w(f"layers.{L}.ffn.gate.weight").float().numpy().astype(np.float64)
    gb = w(f"layers.{L}.ffn.gate.bias").float().numpy().astype(np.float64)
    eid, wgt = M.gate_of(x, gw, gb)
    print(f"routing: {eid.tolist()}  weights {np.round(wgt, 4).tolist()}")

    def moe(quant):
        r = np.zeros(H)
        for e, wv in zip(eid, wgt):
            r += wv * M.expert_raw(M.CK_, L, int(e), x, quant_x=quant, quant_act=quant)
        return r + M.expert_raw(M.CK_, L, None, x, quant_x=quant, quant_act=quant)

    # the head's weight and gamma, and the golden logprobs
    Wh = w("head.weight").float().numpy().astype(np.float64)
    gn = w("norm.weight").double().numpy()
    row = json.loads(open(a.golden_jsonl).read().splitlines()[0])
    A = {int(p[1]): float(p[0]) for p in row["topk"][1]}

    def logprob_of(head_in):
        lg = Wh @ head_in
        m = lg.max()
        return lg, (lg - m - np.log(np.exp(lg - m).sum()))

    def report(tag, head_in):
        lg, lp = logprob_of(head_in)
        d = np.array([abs(lp[t] - A[t]) for t in A])
        ours = set(np.argsort(-lg)[:32].tolist())
        print(f"  {tag:<34} mean|dlogprob| {d.mean():.4f}  max {d.max():.4f}  top1 {int(np.argmax(lg)):>7}  "
              f"overlap {len(ours & set(A))}/32")

    print(f"\n{'':<34} {'mean |dlogprob| vs golden'}")
    for quant in (False, True):
        yy = moe(quant)
        # hc_post, bf16 as model.py's y.type_as(x)
        so = np.empty((HC, H))
        for j in range(HC):
            so[j] = post[j] * yy + sum(comb[i][j] * rin[i] for i in range(HC))
        so = bf16r(so)
        # the head's input: hc_pre with the SAME (ffn) pre, then norm
        hh = (pre[:, None] * so).sum(0)
        head_in = bf16r(hh / np.sqrt((hh ** 2).mean() + EPS) * gn)
        report(f"fp64 MoE, quant={quant}", head_in)

    # scale: what the kernel itself produced with the identical residual forced in
    tx = np.fromfile("/tmp/teach_dump/moe39_teach_x.f32", np.float32).astype(np.float64)
    report("our kernel's own layer 39 (teach)", tx)
    if os.path.exists(a.ours):
        lg = np.fromfile(a.ours, np.float32)
        m = lg.max()
        lp = (lg - m - np.log(np.exp((lg - m).astype(np.float64)).sum())).astype(np.float64)
        d = np.array([abs(lp[t] - A[t]) for t in A])
        print(f"  {'teach run logits (whole chain forced)':<34} mean|dlogprob| {d.mean():.4f}")


if __name__ == "__main__":
    main()
