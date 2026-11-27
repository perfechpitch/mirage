"""Run the official tilelang sparse_attn kernel and compare its `o` with the megakernel's `o_q`, bit for bit.

`attn_nodes.py` leaves one node open: `o_q`. Every other node of the attention sublayer is at the fp32 floor
(`x_q`, `qr_q`, `oa_q` byte-identical; the five GEMMs cos exactly 1.000000000). For `o_q` a numpy reimplementation of
the online softmax stops at 4.6e-3 / 8.5e-3 (~1-2 bf16 ulp), because P's bf16 rounding plus `mma.sync`'s internal
accumulation order cannot be reproduced in software.

The decisive test is not another numpy variant: it is to run **the official kernel itself** --
`inference/kernel.py sparse_attn_kernel`, the tilelang program SGLang executes -- on the *same* q and K, and ask
whether its `o` is bit-identical to the megakernel's. If it is, `o_q` is proven exact and the residual against SGLang
is SGLang's own launch-to-launch scatter (FINDINGS §12.17: 1.46 ulp at L1 from two runs of the same binary).

Inputs are taken from the megakernel's own buffers, so nothing is re-derived on the way in:

    q     = bf16( rope( bf16( sum of wq_b_part's K parts ) ) )      <- attn{L}_r{r}_wq_b_part.bin
    K_new = act_quant( bf16( rope( rms_norm( bf16( sum of wkv_part's parts ) ) ) ) )   <- attn{L}_r{r}_wkv_part.bin
    K_old = the reference pool's rows (the same bytes the megakernel reads: it only writes the new row)
    sink  = layers.L.attn.attn_sink
    idxs  = arange(128), i.e. the ring in the official's list order, oldest first

and both RoPEs are the kernels' own: forward is `fma(a, cos, -b*sin)` / `fma(b, cos, a*sin)`, the inverse
`fma(a, cos, -b*(-sin))` / `fma(b, cos, a*(-sin))`, each followed by a bf16 round.

    python tilelang_check.py --layer 1 --dump /tmp/chain_attn2
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch

import attn_np as A
from attn_nodes import HID, QL, D, OL, bf16r, act_quant, deq_fp8, parts, rms_norm_parts8

RD = 64
CK = A.CK


def off_kernel():
    """The official `sparse_attn` (tilelang), imported straight out of the checkpoint's inference/ directory."""
    path = os.path.join(CK, "inference", "kernel.py")
    spec = importlib.util.spec_from_file_location("dsv41_offk", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def freqs_at(pos, theta=10000.0):
    """Row `pos` of the chain's RoPE table (weights.py precompute_freqs_cis: built on the GPU, tilelang/PyTorch bits)."""
    inv = 1.0 / (theta ** (torch.arange(0, RD, 2, dtype=torch.float32, device="cuda") / RD))
    ang = torch.outer(torch.arange(pos + 1, device="cuda", dtype=torch.float32), inv)[pos]
    return torch.cos(ang), torch.sin(ang)


def rope_fma(x, c, s, inverse=False, width=None):
    """sparse_attn.cuh / kv_post.cuh RoPE on a row's LAST `RD` of `width` columns: fp32 fma per pair, then bf16.

    `fma(a, c, -mul(b, s_i))` is exact in fp64 (fp32 x fp32 products are exact there, and the one rounding to fp32 is
    the fma's), so this reproduces the instruction, not an approximation of it.
    """
    x = np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, np.float64)
    W = width or x.shape[-1]
    v = x.reshape(-1, W).copy()
    tail = v[:, W - RD:].reshape(-1, RD // 2, 2).copy()
    cr = np.asarray(c.cpu() if hasattr(c, "cpu") else c, np.float64)
    ci = np.asarray(((-s if inverse else s)).cpu() if hasattr(s, "cpu") else (-s if inverse else s), np.float64)
    a_, b_ = tail[..., 0], tail[..., 1]
    v[:, W - RD:] = bf16r(np.stack([(a_ * cr - b_ * ci).astype(np.float32),
                                    (b_ * cr + a_ * ci).astype(np.float32)], -1).reshape(-1, RD))
    return v


def rope_torch(x, c, s, inverse=False, width=None):
    """model.py apply_rotary_emb's fp32 complex multiply (what SGLang's q goes through before the tilelang kernel)."""
    x = np.asarray(x, np.float32)
    W = width or x.shape[-1]
    v = x.reshape(-1, W).copy()
    tail = v[:, W - RD:].reshape(-1, RD // 2, 2).copy()
    cr, ci = np.asarray(c.cpu(), np.float32), np.asarray((-s if inverse else s).cpu(), np.float32)
    a_, b_ = tail[..., 0].copy(), tail[..., 1].copy()
    v[:, W - RD:] = bf16r(np.stack([(a_ * cr - b_ * ci).astype(np.float32),
                                    (a_ * ci + b_ * cr).astype(np.float32)], -1).reshape(-1, RD))
    return v


def zq_to_bf16(zq):
    """A z_q (e4m3 + 512-B scale chunks per 128 columns) -> the bf16 values the cache holds (exact: 4 significant bits)."""
    a = np.frombuffer(zq.tobytes() if hasattr(zq, "tobytes") else bytes(zq), np.uint8)
    q = torch.frombuffer(bytearray(a[:D].tobytes()), dtype=torch.float8_e4m3fn).float()
    return (q.view(D // 32, 32) * torch.pow(2.0, a[D:].astype(np.int32) - 127.0)[:, None]).reshape(D).numpy()


def bits(a, b):
    a = np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint32)
    b = np.ascontiguousarray(np.asarray(b, np.float32)).view(np.uint32)
    return int((a == b).sum()), int(a.size)


def rr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--dump", default="/tmp/chain_attn2")
    ap.add_argument("--pos", type=int, default=31906)
    ap.add_argument("--ranks", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--eps", type=float, default=1e-20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    L, DD, pos = a.layer, a.dump, a.pos
    p = f"layers.{L}.attn."
    scale = float(D ** -0.5)
    ranks = [int(x) for x in a.ranks.split(",")]

    off = off_kernel()
    print(f"official kernel: {os.path.join(CK, 'inference', 'kernel.py')} sparse_attn_kernel(h, 512, scale)")

    # --- inputs, from the megakernel's own buffers -------------------------------------------------
    qn = A.read(p + "q_norm.weight").float().numpy()
    kvn = A.read(p + "kv_norm.weight").float().numpy()
    sink = A.read(p + "attn_sink").float()
    c, s = freqs_at(pos)

    # K: current row recomputed from wkv's parts exactly as kv_post.cuh does it
    #    sum parts -> bf16 -> RMSNorm -> bf16 -> RoPE -> bf16 -> act_quant (per 32 columns)
    T = 8
    wkv_sum = parts(np.fromfile(os.path.join(DD, f"attn{L}_r0_wkv_part.bin"), np.uint8), T, D, 2)[:, 0, :].sum(0)
    k_new = rms_norm_parts8(wkv_sum[None, :], kvn, a.eps, quant=False)     # [1, 512] bf16
    k_new = rope_fma(torch.from_numpy(k_new), c, s)[0]                     # [512] bf16, rope'd
    k_new_q = act_quant(k_new[None, :]).reshape(-1)                        # its quantized values
    k_new_bf = bf16r(k_new_q)

    lst = A.window_kv(pos, L)
    kv_rows = []
    for _pp, b in lst:
        kv_rows.append(torch.from_numpy(k_new_bf.copy()) if b is None else A.deq_row(b))
    kv = torch.stack(kv_rows).to(a.device).bfloat16()                    # [128, 512], the official's list order
    idxs = torch.arange(len(kv), dtype=torch.int32, device=a.device).view(1, 1, -1)
    print(f"kv [128, 512] bf16, list order = the ring oldest first (current token: slot {pos % 128})")

    print(f"\n{'rank':>5} {'q':>10} {'cos':>12} {'bits':>17} {'2-block numpy':>14}")
    allbits = []
    for r in ranks:
        # q: exactly wq_b's fp32 parts -> bf16 -> RoPE
        q_bf = bf16r(parts(np.fromfile(os.path.join(DD, f"attn{L}_r{r}_wq_b_part.bin"), np.uint8),
                           T, 8 * D, 1)[0][0])                         # [4096] bf16-valued
        q_bf = q_bf.reshape(8, D)
        res = {}
        for tag, rope in (("fma", rope_fma), ("torch", rope_torch)):
            q = rope(q_bf, c, s)                                           # ropes the last RD columns
            q = torch.from_numpy(q.astype(np.float32)).to(a.device).bfloat16().view(1, 1, 8, D)
            o = off.sparse_attn(q, kv.view(1, -1, D), sink[r * 8:r * 8 + 8].to(a.device), idxs, scale)
            # the megakernel's o_q has the inverse RoPE applied, in bf16
            oi = np.asarray(o[0, 0].float().cpu()).copy()
            oi = rope_fma(oi, c, s, inverse=True, width=D)
            res[tag] = oi
        ker = torch.frombuffer(bytearray(open(os.path.join(DD, f"attn{L}_r{r}_o_q.bin"), "rb").read()),
                               dtype=torch.bfloat16).float().reshape(-1, 8, D)[0].numpy()
        fma_bits = bits(res["fma"], ker)
        tr_bits = bits(res["torch"], ker)
        print(f"{r:>5} {'fma':>10} {A.cos(res['fma'], ker):>+12.9f} {fma_bits[0]:>7}/{fma_bits[1]:<9} "
              f"{'':>14}")
        print(f"{'':>5} {'torch':>10} {A.cos(res['torch'], ker):>+12.9f} {tr_bits[0]:>7}/{tr_bits[1]:<9} "
              f"relRMS {rr(res['torch'], ker):.3e}")
        allbits.append((fma_bits, tr_bits))

    tot_f = sum(b[0][0] for b in allbits), sum(b[0][1] for b in allbits)
    tot_t = sum(b[1][0] for b in allbits), sum(b[1][1] for b in allbits)
    print(f"\nofficial tilelang vs the megakernel's o_q:")
    print(f"  q via the kernels' fma RoPE : {tot_f[0]}/{tot_f[1]} bits identical")
    print(f"  q via torch's complex RoPE : {tot_t[0]}/{tot_t[1]} bits identical")


if __name__ == "__main__":
    main()
