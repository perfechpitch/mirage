"""Are e2e_decode's layer tensors the same ones full_decode builds? One script, both constructions, a diff.

e2e_decode goes through dsv41_static_layer.StaticAttn/StaticMoE (which add the serving-only pieces and compile), so
its weight loading is one indirection away from full_decode's inline loop. This compares the two *construction*
steps only -- layer_tensors + state_tensors + the mHC tensors -- with the arguments each side actually passes:

    e2e         max_pos = pos + 8   ->  rows = (max_pos + 2 + 255) // 256 * 256, cap from max_pos + 1
    full_decode pos                 ->  rows = (pos + 2 + 255) // 256 * 256,    cap = pool_cap(ratio)

and the mHC tensors (mhc_tensors vs mhc_into), where the one visible difference is gamma_next's cast.

python cmp_weights.py --checkpoint DIR [--pos 31906] [--layers 0,1,2,3]
"""
import argparse
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from deepseek_v4_1_csa2 import demo as attn_demo            # noqa: E402
from deepseek_v4_1_csa2 import weights as attn_weights      # noqa: E402
import dsv41_static_layer as Lm                             # noqa: E402

A = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
A.add_argument("--checkpoint", required=True)
A.add_argument("--pos", type=int, default=31906)
A.add_argument("--layers", default="0,1,2,3")
A.add_argument("--tp", type=int, default=8)
A.add_argument("--rank", type=int, default=0)
a = A.parse_args()

T = Lm.T
HC, HID = Lm.HC, Lm.HID
dev = torch.device("cpu")          # loading only: no GPU needed (and the GPUs are busy)
cfg = Lm.attn_weights.model_config(a.checkpoint)
read = Lm.reader(a.checkpoint)
pos = a.pos


def digest(t):
    b = t.detach().cpu().contiguous()
    h = hashlib.sha1(b.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
    if t.is_floating_point() and t.numel():
        return f"{h} mean {b.float().mean():+.6e} rms {b.float().pow(2).mean().sqrt():.6e}"
    return h


def e2e_side(L, ratio, csrc):
    """StaticAttn.__init__'s construction, verbatim (minus the build)."""
    variant = attn_weights.variant_of(cfg, L)
    max_pos = pos + 8                                        # e2e_decode passes pos + 8
    rows = (max_pos + 2 + 255) // 256 * 256
    cap = max(((max_pos + 1) // max(ratio, 1) + 255) // 256 * 256, attn_demo.NC)
    lt = attn_weights.layer_tensors(read, cfg, L, a.tp, a.rank, rows, dev)
    st = attn_demo.state_tensors(variant, ratio, csrc, cap, dev)
    mh = Lm.mhc_tensors(read, L, "attn", f"layers.{L}.ffn_norm.weight", dev)
    return variant, rows, cap, {**lt, **st, **mh}


def full_side(L, ratio, csrc):
    """full_decode's loop, verbatim (its lines 411-431 + mhc_into)."""
    variant = attn_weights.variant_of(cfg, L)
    rows = (pos + 2 + 255) // 256 * 256
    cap = max(-(-((pos + 1) // max(ratio, 1) + 1) // 256) * 256, attn_demo.NC)   # pool_cap
    lt = attn_weights.layer_tensors(read, cfg, L, a.tp, a.rank, rows, dev)
    st = attn_demo.state_tensors(variant, ratio, csrc, cap, dev)
    # mhc_into(wg, d, L, "attn", read(f"layers.{L}.ffn_norm.weight"))
    p = f"layers.{L}."
    fn, base, scale = (read(p + f"hc_attn_{n}").float() for n in ("fn", "base", "scale"))
    hi = fn.bfloat16()
    fn_w = torch.cat([hi, (fn - hi.float()).bfloat16(),
                      torch.zeros(128 - 2 * fn.shape[0], fn.shape[1], dtype=torch.bfloat16)])
    gamma = read(f"layers.{L}.ffn_norm.weight")
    mh = dict(hc_fn_w=fn_w.to(dev).contiguous(),
              streams=torch.zeros(T, HC * HID, dtype=torch.bfloat16, device=dev),
              streams_out=torch.zeros(T, HC * HID, dtype=torch.bfloat16, device=dev),
              hc_fn=fn.to(dev).contiguous(), hc_small=torch.cat([base, scale]).to(dev).contiguous(),
              h_pre=torch.zeros(T, HC, device=dev), x_out=torch.zeros(T, HID, dtype=torch.bfloat16, device=dev),
              gamma_next=gamma.to(dev).contiguous())
    return variant, rows, cap, {**lt, **st, **mh}


print(f"T={T} tp={a.tp} rank={a.rank} pos={pos}   (e2e: max_pos = pos + 8)")
bad_total = 0
for L in [int(v) for v in a.layers.split(",")]:
    ratio = cfg["compress_ratios"][L]
    csrc = L == cfg["candidate_source_layer"]
    v1, r1, c1, w1 = e2e_side(L, ratio, csrc)
    v2, r2, c2, w2 = full_side(L, ratio, csrc)
    print(f"\n===== L{L} {v1} (ratio {ratio})")
    print(f"  variant {v1} vs {v2} {'OK' if v1 == v2 else 'DIFF'};  rows {r1} vs {r2} {'OK' if r1 == r2 else 'DIFF'};"
          f"  cap {c1} vs {c2} {'OK' if c1 == c2 else 'DIFF'}")
    bad = []
    only1, only2 = set(w1) - set(w2), set(w2) - set(w1)
    if only1:
        bad.append(f"only e2e: {sorted(only1)}")
    if only2:
        bad.append(f"only full_decode: {sorted(only2)}")
    for k in sorted(set(w1) & set(w2)):
        t1, t2 = w1[k], w2[k]
        if tuple(t1.shape) != tuple(t2.shape):
            bad.append(f"{k}: shape {tuple(t1.shape)} vs {tuple(t2.shape)}")
            continue
        if t1.dtype != t2.dtype:
            bad.append(f"{k}: dtype {t1.dtype} vs {t2.dtype}")
            continue
        if digest(t1) != digest(t2):
            bad.append(f"{k}: VALUES differ\n      e2e  {digest(t1)}\n      full {digest(t2)}")
    same = len(set(w1) & set(w2)) - len([b for b in bad if b.split(':')[0] in w1])
    print(f"  {len(set(w1) & set(w2))} tensors compared, {len(bad)} differ")
    for b in bad:
        print("   *", b)
    bad_total += len(bad)

print(f"\n{'ALL ALIGNED' if bad_total == 0 else str(bad_total) + ' DIFFERENCES'}")

# ---- the MoE side, and the Engram tensors (the other four layers' worth of loading) --------------------------------
from deepseek_v4_1_moe import demo as moe_demo              # noqa: E402
from deepseek_v4_1_moe import weights as moe_weights        # noqa: E402
import head_demo                                            # noqa: E402

EG_HOST = {g - 1: g for g in (int(v) for v in cfg.get("engram_layer_ids", []))}
L40 = 40


def moe_e2e(L, gamma_name):
    """StaticMoE.__init__'s construction, verbatim (minus the build)."""
    w = moe_weights.layer_tensors(read, L, a.tp, a.rank, dev)
    w.update(moe_weights.routed_tensors(read, L, a.tp, [a.rank], dev)[0])     # <-- one rank, take [0]
    w["x"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
    w["y"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
    norm_next = L not in EG_HOST                    # (the Engram host: norm_next off; see e2e_decode)
    w.update(Lm.mhc_tensors(read, L, "ffn", gamma_name if norm_next else None, dev))
    if L in EG_HOST:
        eg = Lm.engram_tensors(read, L + 1, a.tp, a.rank, dev)
        w.update({k: eg[k] for k in ("eg_qk", "eg_wkv", "eg_wkv_scales", "eg_emb", "streams_eg")})
    return w


def moe_full(L, gamma_name):
    """full_decode's loop, verbatim (its lines 473-489 + mhc_into + engram_into)."""
    w = moe_weights.layer_tensors(read, L, a.tp, a.rank, dev)
    w.update(moe_weights.routed_tensors(read, L, a.tp, list(range(a.tp)), dev)[a.rank])   # <-- all ranks, take [r]
    p = f"layers.{L}."
    fn, base, scale = (read(p + f"hc_ffn_{n}").float() for n in ("fn", "base", "scale"))
    hi = fn.bfloat16()
    w.update(hc_fn_w=torch.cat([hi, (fn - hi.float()).bfloat16(),
                                torch.zeros(128 - 2 * fn.shape[0], fn.shape[1], dtype=torch.bfloat16)]).to(dev).contiguous(),
             streams=torch.zeros(T, HC * HID, dtype=torch.bfloat16, device=dev),
             streams_out=torch.zeros(T, HC * HID, dtype=torch.bfloat16, device=dev),
             hc_small=torch.cat([base, scale]).to(dev).contiguous(),
             h_pre=torch.zeros(T, HC, device=dev), x_out=torch.zeros(T, HID, dtype=torch.bfloat16, device=dev),
             gamma_next=read(gamma_name).to(dev).contiguous())
    if L in EG_HOST:
        from mirage.mpk.static_megakernel import w8_scale_chunks
        q = f"layers.{L + 1}.engram."
        wkv, sc = read(q + "wkv.weight"), read(q + "wkv.scale")
        n = wkv.shape[0] // a.tp
        rows = slice(a.rank * n, a.rank * n + n)
        w.update(eg_wkv=wkv[rows].contiguous().view(torch.uint8).to(dev),
                 eg_wkv_scales=w8_scale_chunks(sc[a.rank * n // 32:(a.rank * n + n) // 32].contiguous()).to(dev),
                 eg_qk=(read(q + "q_weight").float() * read(q + "k_weight").float()).to(dev).contiguous(),
                 eg_emb=torch.zeros(T, wkv.shape[1], dtype=torch.bfloat16, device=dev),
                 streams_eg=torch.zeros(T, HC * HID, dtype=torch.bfloat16, device=dev))
    return w


print("\n############ MoE")
for L in [int(v) for v in a.layers.split(",")]:
    last = L == L40 - 1
    gamma_name = "norm.weight" if last else f"layers.{L + 1}.attn_norm.weight"
    w1, w2 = moe_e2e(L, gamma_name), moe_full(L, gamma_name)
    head = last
    print(f"\n===== MoE L{L}" + (f"  (Engram host of layer {EG_HOST[L]})" if L in EG_HOST else "") + (" + head" if head else ""))
    bad = []
    only1, only2 = set(w1) - set(w2), set(w2) - set(w1)
    if only1:
        bad.append(f"only e2e: {sorted(only1)}")
    if only2:
        bad.append(f"only full_decode: {sorted(only2)}")
    for k in sorted(set(w1) & set(w2)):
        t1, t2 = w1[k], w2[k]
        if tuple(t1.shape) != tuple(t2.shape):
            bad.append(f"{k}: shape {tuple(t1.shape)} vs {tuple(t2.shape)}")
        elif t1.dtype != t2.dtype:
            bad.append(f"{k}: dtype {t1.dtype} vs {t2.dtype}")
        elif digest(t1) != digest(t2):
            bad.append(f"{k}: VALUES differ\n      e2e  {digest(t1)}\n      full {digest(t2)}")
    print(f"  {len(set(w1) & set(w2))} tensors compared, {len(bad)} differ")
    for b in bad:
        print("   *", b)
    bad_total += len(bad)
    if head:   # the head reads "norm.weight"; both sides call head_tensors with the same keys
        hw = head_demo.head_tensors(read, a.tp, a.rank, dev)
        print(f"  head tensors: {[(k, tuple(hw[k].shape), str(hw[k].dtype)) for k in ('head_w', 'logits', 'head_ids') if k in hw]}")

print(f"\n{'ALL ALIGNED' if bad_total == 0 else str(bad_total) + ' DIFFERENCES'}")
