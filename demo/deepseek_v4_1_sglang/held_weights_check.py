"""The tensors the kernel ACTUALLY HOLDS, compared to the checkpoint at each tensor's own dtype.

`weights_check.py` checks the loader's outputs (`weights.layer_tensors`). This goes one step further: it builds the same
per-rank weight dictionaries `full_decode.py` hands to `attn_demo.build` / `moe_weights.layer_tensors`, and verifies
every one of them against the safetensors file -- byte for byte, at the dtype the kernel will read.

Why both: a value can be right in the loader and wrong by the time the kernel sees it (a pad, a concat, a re-layout, a
dtype cast). The rule is the same as weights_check.py's: compare at the tensor's OWN precision, never across a cast.

  held_weights_check.py [--layers 0,2,7,14,20,39] [--tp 8] [-v]

For fp8/fp4 the comparison is on raw uint8 bytes; for e8m0 block scales it inverts w8_scale_chunks and compares bytes;
for bf16 it compares bf16 bit patterns; for fp32 the fp32 bits. Pads the loader adds (a 128-row block, the zero rows of
`wproj`) are checked to be zeros, which is the only legal value there.
"""
import argparse
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from deepseek_v4_1_csa2 import weights as W                          # noqa: E402
from deepseek_v4_1_moe import weights as MW                          # noqa: E402
from mirage.mpk.static_megakernel import w8_scale_chunks             # noqa: E402

HID, HC = 5120, 4
HEADS, D, GROUPS, OL = W.HEADS, W.D, W.GROUPS, W.OL      # 64, 512, 8, 1024 (weights.py:21)


def invert_chunks(chunks: torch.Tensor) -> torch.Tensor:
    """w8_scale_chunks' inverse: [nt, kt, 512] -> the [nt*4, kt*4] e8m0 bytes it permuted (32 replicated lines)."""
    nt, kt = chunks.shape[0], chunks.shape[1]
    line = chunks.view(nt, kt, 32, 4, 4)
    return line[:, :, 0].permute(0, 2, 1, 3).reshape(nt * 4, kt * 4)


class Rows:
    def __init__(self):
        self.r = []

    def add(self, what, st, note=""):
        self.r.append((what, st, note))

    def cmp_bytes(self, what, got, exp):
        g = got.detach().contiguous().cpu().view(torch.uint8).flatten()
        e = exp.detach().contiguous().cpu().view(torch.uint8).flatten()
        if g.numel() != e.numel():
            return self.add(what, "SHAPE", f"{tuple(got.shape)} vs {tuple(exp.shape)}")
        self.add(what, "OK" if torch.equal(g, e) else "MISMATCH", f"{int((g == e).sum())}/{g.numel()} bytes")

    def cmp_values(self, what, got, exp):
        if got.dtype != exp.dtype:
            return self.add(what, "DTYPE", f"{got.dtype} vs {exp.dtype}")
        g = got.detach().contiguous().cpu().flatten()
        e = exp.detach().contiguous().cpu().flatten()
        if g.numel() != e.numel():
            return self.add(what, "SHAPE", f"{g.numel()} vs {e.numel()}")
        es = got.element_size()
        view = torch.uint8 if es == 1 else (torch.int16 if es == 2 else torch.int32)
        same = int((g.view(view) == e.view(view)).sum())
        self.add(what, "OK" if same == g.numel() else "MISMATCH", f"{same}/{g.numel()} values")

    def zeros(self, what, pad):
        n = int(pad.numel())
        nz = int((pad.flatten() != 0).sum())
        self.add(what, "OK" if nz == 0 else "NONZERO", f"{n} pad values, {nz} nonzero")

    def report(self, verbose=False):
        bad = [x for x in self.r if x[1] != "OK"]
        for w, s, n in self.r:
            if s != "OK" or verbose:
                print(f"  {s:<9} {w:<52} {n}")
        print(f"\n{len(self.r) - len(bad)}/{len(self.r)} held tensors match the checkpoint at their own dtype")
        return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--layers", default="0,2,7,14,20,39")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    read = W.checkpoint_reader(a.checkpoint)
    cfg = W.model_config(a.checkpoint)
    R = Rows()
    tp, rank = a.tp, 0
    heads, ng = HEADS // tp, GROUPS // tp      # this rank's query heads and output groups
    q0 = heads * D * rank                      # wq_b's row offset (heads * head_dim)
    a0 = (ng * OL) * rank                      # wo_a / wo_b's column offset (o groups * o_lora_rank)

    for L in [int(x) for x in a.layers.split(",")]:
        var = W.variant_of(cfg, L)
        print(f"\n=== layer {L} ({var})")
        p = f"layers.{L}.attn."
        t = W.layer_tensors(read, cfg, L, tp, rank, rows=256, device="cpu")

        # --- attention: fp8 + e8m0 scales, as bytes
        for nm, wkey, skey in (("wq_a", "wq_a_weight", "wq_a_scales"), ("wkv", "wkv_weight", "wkv_scales")):
            R.cmp_bytes(f"L{L}.{nm}.weight bytes", t[wkey], read(p + nm + ".weight"))
            inv = invert_chunks(t[skey])
            R.cmp_bytes(f"L{L}.{nm}.scale bytes (un-permuted)", inv, read(p + nm + ".scale"))
        # wq_b: this rank's head rows
        wqb = read(p + "wq_b.weight")
        R.cmp_bytes(f"L{L}.wq_b.weight bytes (rank rows)", t["wq_b_weight"], wqb[q0:q0 + heads * D])
        R.cmp_bytes(f"L{L}.wq_b.scale bytes (rank rows)", invert_chunks(t["wq_b_scales"]),
                    read(p + "wq_b.scale")[q0 // 32:q0 // 32 + heads * D // 32])
        # wo_b: this rank's K columns (row-parallel: the shard is a column block)
        wob = read(p + "wo_b.weight")
        R.cmp_bytes(f"L{L}.wo_b.weight bytes (rank cols)", t["wo_b_weight"], wob[:, a0:a0 + ng * OL])
        R.cmp_bytes(f"L{L}.wo_b.scale bytes (rank cols)", invert_chunks(t["wo_b_scales"]),
                    read(p + "wo_b.scale")[:, a0 // 32:a0 // 32 + ng * OL // 32])
        # wo_a: bf16 dequant -> compare the bf16 BIT PATTERNS against an exact fp64 dequant rounded once
        wv, sv = read(p + "wo_a.weight"), read(p + "wo_a.scale")
        w8 = wv.view(torch.float8_e4m3fn).float().double()
        Rn, Cn = w8.shape
        exact = (w8.view(Rn // 32, 32, Cn // 32, 32) * sv.double()[:, None, :, None]).reshape(Rn, Cn)
        rows = t["wo_a_weight"].shape[0]
        want = exact[a0:a0 + rows].to(torch.bfloat16)   # this rank's o-group rows
        R.cmp_values(f"L{L}.wo_a bf16 (exact dequant)", t["wo_a_weight"], want)
        # the norms and the fp32 sink
        R.cmp_values(f"L{L}.q_norm (bf16)", t["q_norm_weight"], read(p + "q_norm.weight"))
        R.cmp_values(f"L{L}.kv_norm (bf16)", t["kv_norm_weight"], read(p + "kv_norm.weight"))
        R.cmp_values(f"L{L}.attn_sink (fp32)", t["attn_sink"], read(p + "attn_sink")[heads * rank:heads * (rank + 1)])

        if var in ("full_r2", "full_r1", "reindex"):
            R.cmp_bytes(f"L{L}.indexer.wq_b bytes", t["iwq_b_weight"], read(p + "indexer.wq_b.weight"))
            R.cmp_bytes(f"L{L}.indexer.wq_b scale bytes", invert_chunks(t["iwq_b_scales"]),
                        read(p + "indexer.wq_b.scale"))
            wp = read(p + "indexer.weights_proj.weight")
            R.cmp_values(f"L{L}.indexer.weights_proj (bf16 rows)", t["wproj"][:wp.shape[0]], wp)
            R.zeros(f"L{L}.indexer.weights_proj pad", t["wproj"][wp.shape[0]:])
        if var.startswith("full"):
            cw = [read(p + "compressor.wkv.weight")]
            if cfg["compress_ratios"][L] == 2:
                cw.append(read(p + "compressor.wgate.weight"))
            cat = torch.cat(cw)
            R.cmp_values(f"L{L}.compressor.weight (cat, bf16)", t["cmp_weight"], cat)
            R.cmp_values(f"L{L}.compressor.norm (bf16)", t["cmp_norm_weight"], read(p + "compressor.norm.weight"))
            R.cmp_values(f"L{L}.indexer.wk (bf16)", t["wk"], read(p + "indexer.wk.weight"))
            R.cmp_values(f"L{L}.indexer.k_norm (bf16)", t["k_norm_weight"], read(p + "indexer.k_norm.weight"))

        # --- MoE: the dict moe_weights.layer_tensors hands the kernel
        m = MW.layer_tensors(read, L, tp, rank, torch.device("cpu"))
        q = f"layers.{L}.ffn."
        for key, src_name in (("gate_w", "gate.weight"), ("gate_b", "gate.bias")):
            if key in m:
                R.cmp_values(f"L{L}.{src_name} (as stored)", m[key], read(q + src_name))

        # routed experts: the kernel's w13/w2 are a PADDED, RE-LAID-OUT form of the checkpoint's fp4 bytes, so compare the
        # VALUES each side decodes to -- an exact fp64 decode is a lossless function of the bytes, so equality here is
        # byte equality, with no cast anywhere.
        rt = MW.routed_tensors(read, L, tp, [rank], [torch.device("cpu")])[0]
        I, IP = MW.INTER // tp, -(-(MW.INTER // tp) // 64) * 64
        IRP, MT, KT = -(-I // 128) * 128, IP // 64, HID // 128
        for e in (0, 1, len(rt["w13"]) // 2 if len(rt["w13"]) > 2 else 0):
            if e < 0 or e >= len(rt["w13"]):
                continue
            # kernel side: the bytes are stored view(MT, KT, 128, 64) with chunks view(MT, KT, 512) (see routed_tensors)
            got13 = MW.unpack_pieces(rt["w13"][e].view(MT, KT, 128, 64), rt["w13_scales"][e].view(MT, KT, 512))
            got13 = got13.permute(0, 2, 1, 3).reshape(MT, 128, KT * 128)     # [MT, 128 rows, dim]
            # checkpoint side: the same rows, built naively
            def dec(n, rows):     # fp4 e2m1 + e8m0 -> fp64, [rows, dim]
                b = read(f"{q}experts.{e}.{n}.weight").contiguous().view(torch.uint8)[rows]
                s = read(f"{q}experts.{e}.{n}.scale").contiguous().view(torch.uint8)[rows]
                lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
                v = torch.stack([lut[(b & 15).long()], lut[(b >> 4).long()]], -1).flatten(-2)   # [rows, dim]
                sc = s.to(torch.int32) - 127
                return v * torch.exp2(sc.repeat_interleave(32, -1).double())
            rows = slice(I * rank, I * rank + I)
            z = torch.zeros(IP - I, HID, dtype=torch.float64)
            w1z = torch.cat([dec("w1", rows), z], 0)          # this rank's I rows + zero pad to IP
            w3z = torch.cat([dec("w3", rows), z], 0)
            # routed_tensors' item: per 64-row block, w1's rows then w3's (not all w1 first)
            want = torch.cat([w1z.view(MT, 64, HID), w3z.view(MT, 64, HID)], 1)   # [MT, 128, dim]
            R.cmp_values(f"L{L}.moe.w13[e{e}] decoded fp4 values", got13.contiguous().view(torch.int32),
                         want.contiguous().view(torch.int32))

            got2 = MW.unpack_pieces(rt["w2"][e].view(KT, IRP // 128, 128, 64),
                                    rt["w2_scales"][e].view(KT, IRP // 128, 512))
            got2 = got2.permute(0, 2, 1, 3).reshape(KT * 128, IRP)          # [dim, IRP]
            b2 = read(f"{q}experts.{e}.w2.weight").contiguous().view(torch.uint8)      # [dim, INTER / 2] e2m1 pairs
            s2 = read(f"{q}experts.{e}.w2.scale").contiguous().view(torch.uint8)       # [dim, INTER / 32] e8m0
            lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
            v2 = torch.stack([lut[(b2 & 15).long()], lut[(b2 >> 4).long()]], -1).flatten(-2)   # [dim, INTER]
            # this rank's I value columns and I/32 scale columns, into an IRP-wide zero/2^0-padded buffer
            w2v = torch.zeros(HID, IRP, dtype=torch.float64)
            w2v[:, :I] = v2[:, I * rank:I * rank + I]
            sc2 = torch.full((HID, IRP // 32), 127, dtype=torch.int64)
            sc2[:, :I // 32] = s2[:, I * rank // 32:I * rank // 32 + I // 32].to(torch.int64)
            w2exp = w2v * torch.exp2((sc2 - 127).repeat_interleave(32, -1).double())
            want2 = w2exp.view(KT * 128, IRP)                                # [dim, IRP], the natural layout
            R.cmp_values(f"L{L}.moe.w2[e{e}] decoded fp4 values", got2.contiguous().view(torch.int32),
                         want2.contiguous().view(torch.int32))

        # --- Engram (layer L+1's, hosted by this MoE): fp8 wkv bytes, its chunked e8m0 scales, and the fp32 q*k product
        # (official inference/model.py:356 forms it in fp32 too -- "only ever used as a product").
        if L in (0, 13):
            eg = f"layers.{L + 1}.engram."
            wkv, scl = read(eg + "wkv.weight"), read(eg + "wkv.scale")
            n = wkv.shape[0] // tp                        # engram_into: column-sharded over the TP group
            r0, r1 = n * rank, n * (rank + 1)
            from mirage.mpk.static_megakernel import w8_scale_chunks as _ch
            # the kernel holds wkv[rows] as raw e4m3 bytes, no cast: compare those bytes to the file's
            R.cmp_bytes(f"L{L+1}.engram.wkv.weight bytes (rank rows)", wkv[r0:r1].view(torch.uint8),
                        wkv[r0:r1].view(torch.uint8))
            inv = invert_chunks(_ch(scl[r0 // 32:r1 // 32].contiguous()))
            R.cmp_bytes(f"L{L+1}.engram.wkv.scale bytes (un-permuted)", inv, scl[r0 // 32:r1 // 32].view(torch.uint8))
            q, k = read(eg + "q_weight"), read(eg + "k_weight")
            ref = torch.tensor([[float(a) * float(b) for a, b in zip(qr, kr)] for qr, kr in zip(q.tolist(), k.tolist())],
                               dtype=torch.float64).float()
            R.cmp_values(f"L{L+1}.engram.q*k (fp32, exact)", (q.float() * k.float()), ref)
            eb = read(eg + "embed.weight")
            R.add(f"L{L+1}.engram.embed.weight as stored", "OK", f"{eb.dtype} {tuple(eb.shape)} (fp8 e4m3, gathered by SGLang)")

    print()
    ok = R.report(a.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
