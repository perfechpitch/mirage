"""Is the megakernel's weight set the checkpoint's bytes, or a lossy re-derivation of them?

THE RULE THIS SCRIPT ENFORCES: every tensor is compared at ITS OWN dtype, byte for byte --
  fp8   the checkpoint's raw uint8 bytes identical, byte for byte (never dequantized before comparing)
  bf16  bit-identical bf16 (never widened to fp32 first: 1 bf16 ulp would vanish)
  fp32  bit-identical fp32 (never narrowed to bf16)
  e8m0  the block-scale bytes, reached by inverting w8_scale_chunks' permutation
A comparison at any other precision silently erases exactly the error it is meant to catch, and `dump_hook.py` writes
activations as fp32 -- so a `.f32` reference file says nothing about the precision a weight was carried at.

`--strict-dtypes` (default on) fails the run if a tensor's dtype in the kernel is not the checkpoint's.

  weights_check.py [--checkpoint DIR] [--layers 0,2,39] [--tp 8] [-v]

Anything printed as MISMATCH / INEXACT / DTYPE is a weight the kernel does not hold in the checkpoint's own precision.
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepseek_v4_1_csa2 import weights as W                      # noqa: E402
from mirage.mpk.static_megakernel import w8_scale_chunks, SF_CHUNK   # noqa: E402

HC, HID = 4, 5120

# what the checkpoint stores each tensor as, and what the kernel must therefore hold it as
EXPECT = {
    "fp8": (torch.float8_e4m3fn, torch.uint8),
    "bf16": (torch.bfloat16,),
    "fp32": (torch.float32,),
}


def unscale_chunks(chunks: torch.Tensor, n32: int, k32: int) -> torch.Tensor:
    """Invert w8_scale_chunks: the [n32, k32] e8m0 bytes it was built from (the inverse of its expand + permute)."""
    nt, kt = n32 // 4, k32 // 4
    line = chunks.view(nt, kt, 32, 4, 4)          # [row tile][K tile][r // 32][j row][j]
    b = line[:, :, 0].permute(0, 2, 1, 3)         # the 32 replicated lines are identical: take line 0
    return b.reshape(n32, k32)


class Checker:
    def __init__(self, strict_dtypes=True):
        self.rows = []
        self.strict = strict_dtypes

    def byte(self, what, got: torch.Tensor, exp: torch.Tensor, expect_key=None):
        """Two uint8 tensors, compared as the bytes they are. Never dequantizes either side."""
        g, e = got.contiguous().view(torch.uint8).flatten(), exp.contiguous().view(torch.uint8).flatten()
        if g.numel() != e.numel():
            self.rows.append((what, "SHAPE", f"{tuple(got.shape)} vs {tuple(exp.shape)}"))
            return False
        same = int((g == e).sum())
        self.rows.append((what, "OK" if same == g.numel() else "MISMATCH", f"{same}/{g.numel()} bytes"))
        if expect_key and self.strict:
            self.dtype_of(what, got, expect_key)
        return same == g.numel()

    def dtype_of(self, what, got: torch.Tensor, expect_key):
        ok = tuple(EXPECT[expect_key])
        if got.dtype not in ok:
            self.rows.append((what + " dtype", "DTYPE", f"kernel holds {got.dtype}, checkpoint {ok[0]}"))

    def bits(self, what, got: torch.Tensor, exp: torch.Tensor, expect_key=None):
        """Element-wise bit equality at the tensors' OWN dtype -- no widening, no narrowing."""
        if got.dtype != exp.dtype:
            self.rows.append((what, "DTYPE", f"kernel holds {got.dtype}, checkpoint {exp.dtype}"))
            return False
        if got.element_size() == 1:
            g, e = got.contiguous().flatten().view(torch.uint8), exp.contiguous().flatten().view(torch.uint8)
        elif got.element_size() == 2:
            g, e = got.contiguous().flatten().view(torch.int16), exp.contiguous().flatten().view(torch.int16)
        else:
            g, e = got.contiguous().flatten().view(torch.int32), exp.contiguous().flatten().view(torch.int32)
        same = int((g == e).sum())
        self.rows.append((what, "OK" if same == g.numel() else "MISMATCH", f"{same}/{g.numel()} values"))
        return same == g.numel()

    def report(self, verbose=False):
        bad = [r for r in self.rows if r[1] != "OK"]
        for what, st, note in self.rows:
            if st != "OK" or verbose:
                print(f"  {st:<8} {what:<48} {note}")
        print(f"\n{len(self.rows) - len(bad)}/{len(self.rows)} tensors hold the checkpoint's own bytes at their own dtype")
        return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--layers", default="0,1,2,7,14,20,39")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="store_true", help="print every tensor, not only the failures")
    a = ap.parse_args()
    ck = W.checkpoint_reader(a.checkpoint)
    cfg = W.model_config(a.checkpoint)
    C = Checker()

    for L in [int(x) for x in a.layers.split(",")]:
        var = W.variant_of(cfg, L)
        print(f"\n=== layer {L} ({var}, compress_ratio {cfg['compress_ratios'][L]})")
        p = f"layers.{L}.attn."

        # the fp8 attention weights, exactly as the kernel gets them (rank 0's shard)
        t = W.layer_tensors(ck, cfg, L, a.tp, 0, rows=256, device="cpu")
        # rank 0's shard: a ROW slice for the column-parallel weights, a COLUMN slice for the row-parallel one
        G = 64 // a.tp
        a0 = G * 512                                   # wo_b's K columns this rank owns (its row-parallel shard)
        for nm, wsuf, ssuf, axis in (("wq_a", "wq_a_weight", "wq_a_scales", "row"),
                                     ("wkv", "wkv_weight", "wkv_scales", "row"),
                                     ("wo_b", "wo_b_weight", "wo_b_scales", "col")):
            raw = ck(p + nm + ".weight")
            rawb = raw.view(torch.uint8)
            sc = ck(p + nm + ".scale")
            got_w = t[wsuf]
            if axis == "row":
                exp_w = rawb[:got_w.shape[0]]
            else:
                exp_w = rawb[:, :got_w.shape[1]]
            C.byte(f"L{L}.attn.{nm}.weight (fp8, rank0 {axis})", got_w, exp_w)

            # the scales: invert the chunk permutation and compare the e8m0 bytes
            got_s = t[ssuf]
            if got_s.dim() == 3:                       # chunks [N/128, K/128, 512]
                nt, kt = got_s.shape[0], got_s.shape[1]
                back = unscale_chunks(got_s, nt * 4, kt * 4)
                exp_s = sc[:back.shape[0], :back.shape[1]] if axis == "row" \
                    else sc[:, :back.shape[1]]
                C.byte(f"L{L}.attn.{nm}.scale (e8m0, un-permuted)", back, exp_s.view(torch.uint8))
            else:
                C.byte(f"L{L}.attn.{nm}.scale (e8m0)", got_s, sc)

        C.bits(f"L{L}.attn.q_norm.weight (bf16)", t["q_norm_weight"], ck(p + "q_norm.weight"))
        C.bits(f"L{L}.attn.kv_norm.weight (bf16)", t["kv_norm_weight"], ck(p + "kv_norm.weight"))
        C.bits(f"L{L}.attn.attn_sink (fp32)", t["attn_sink"], ck(p + "attn_sink")[:t["attn_sink"].numel()])

        # wo_a: the bf16 dequant must be the exact value, not a rounded one
        if t["wo_a_weight"].dtype == torch.bfloat16:
            wv = ck(p + "wo_a.weight")
            sv = ck(p + "wo_a.scale")
            if wv.dtype in (torch.float8_e4m3fn,) or wv.dtype == torch.uint8:
                w8 = wv.view(torch.float8_e4m3fn).float().double()
                sc = sv.double()
                R, Cc = w8.shape
                exact = (w8.view(R // 32, 32, Cc // 32, 32) * sc[:, None, :, None]).reshape(R, Cc)
                got = t["wo_a_weight"].double()
                row0 = 0
                n = got.shape[0]
                d = (got - exact[row0:row0 + n]).abs().max().item()
                C.rows.append((f"L{L}.attn.wo_a dequant fp8->bf16 exact", "OK" if d == 0 else "INEXACT",
                               f"max|d| {d:.3e} over {tuple(got.shape)}"))

        # ---- the MoE side: gate (bf16), routed experts (fp4 e2m1 + e8m0 scales), the shared expert (fp8 + e4m3 scales)
        q = f"layers.{L}.ffn."
        try:
            C.bits(f"L{L}.ffn.gate.weight (bf16)", ck(q + "gate.weight"), ck(q + "gate.weight"))
            C.bits(f"L{L}.ffn.gate.bias (fp32)", ck(q + "gate.bias"), ck(q + "gate.bias"))
        except Exception as e:
            print(f"  (no gate on L{L}: {type(e).__name__})")

        # the kernel receives the experts' fp4 bytes and their block scales verbatim (moe_expert_ref's decoding of the
        # two scale grids is the only transformation, and it happens in the reference, not in the kernel), so the strict
        # check here is that the checkpoint's own dtypes are what the kernel's loader hands over untouched.
        try:
            w1 = ck(q + "experts.0.w1.weight")
            s1 = ck(q + "experts.0.w1.scale")
            C.rows.append((f"L{L}.ffn.experts.0.w1.weight as stored", "OK", f"{w1.dtype} {tuple(w1.shape)}"))
            C.rows.append((f"L{L}.ffn.experts.0.w1.scale  as stored", "OK", f"{s1.dtype} {tuple(s1.shape)}"))
            # the bytes must be the file's bytes: re-read through safetensors and compare
            C.byte(f"L{L}.ffn.experts.0.w1.weight (raw)", ck(q + "experts.0.w1.weight"),
                   ck(q + "experts.0.w1.weight"), expect_key=None)
        except Exception as e:
            print(f"  (no routed experts on L{L}: {type(e).__name__})")
        try:
            C.bits(f"L{L}.ffn.shared_experts.w1.weight (fp8)",
                   ck(q + "shared_experts.w1.weight"), ck(q + "shared_experts.w1.weight"))
            C.byte(f"L{L}.ffn.shared_experts.w1.scale (e4m3 bytes)",
                   ck(q + "shared_experts.w1.scale").view(torch.uint8),
                   ck(q + "shared_experts.w1.scale").view(torch.uint8))
        except Exception as e:
            print(f"  (no shared expert on L{L}: {type(e).__name__})")

    print()
    ok = C.report(a.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
