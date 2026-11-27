"""Where does the fp8 GEMM's fp32 1-ulp difference come from: the arithmetic, or the K-part split?

At the checkpoint's own precision every node is settled except the four fp8 GEMMs, whose outputs are fp32:
wq_a 411/1280, wq_b 2562/4096, wkv 149/512, wo_b 4226/5120 bits identical against the official tilelang
`fp8_gemm_kernel`. The megakernel and the reference are on different tensor cores --

    megakernel  tcgen05.mma kind::mxf8f6f4 block_scale  (scale applied per 32-K block inside the MMA), K in P parts
                of K/P, each with its own accumulator, added by the consumer in part order
    reference   mma.sync m16n8k32, C_acc += C_local * sa * sb per 32-K block, one accumulator over all of K

-- so three things can differ: (1) the 32-term dot each engine rounds, (2) how the 32 blocks are added, (3) the
K-part split. This tool separates them. If the reference, run on the *same* K parts and added in the same order,
reproduces the megakernel bit for bit, then the difference is the decomposition alone -- a framework choice, not
arithmetic -- and it is fixable on either side.

    python gemm_reduction.py --layer 1 --dump /tmp/chain_attn2 --rank 0
"""
import argparse
import os

import numpy as np
import torch

import attn_np as A
import tilelang_check as TL
from tilelang_gemm import e8m0, zq_split
from attn_nodes import HID, QL, D, OL, parts

CK = A.CK
E4 = None


def e4_table():
    """e4m3 byte -> its exact value (4 significant bits; a power-of-two scale keeps it exact everywhere)."""
    global E4
    if E4 is None:
        E4 = np.zeros(256, np.float64)
        for b in range(256):
            sg = -1.0 if b >> 7 else 1.0
            ex_, mn_ = (b >> 3) & 0xF, b & 0x7
            if ex_ == 0:
                v = (mn_ / 8.0) * 2.0 ** -6
            elif ex_ == 0xF and mn_ == 0x7:
                v = np.nan
            else:
                v = (1.0 + mn_ / 8.0) * 2.0 ** (ex_ - 7)
            E4[b] = sg * v
    return E4


def f32(x):
    return np.asarray(x, np.float64).astype(np.float32)


def run_off(off, Aq, As, Bq, Bs, N, K):
    kern = off.fp8_gemm_kernel(N, K, block_size=32, out_dtype=off.FP32, accum_dtype=off.FP32,
                               scale_dtype=off.FE8M0)
    c = torch.empty(1, N, dtype=torch.float32, device="cuda")
    kern(Aq.to("cuda"), Bq.to("cuda"), c, As, Bs.to("cuda"))
    return c[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--dump", default="/tmp/chain_attn2")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--node", default="wq_a")
    a = ap.parse_args()
    L, DD, r = a.layer, a.dump, a.rank
    T = 8
    p = f"layers.{L}.attn."
    off = TL.off_kernel()
    e4 = e4_table()

    # ---- the node's operands, as the bytes the checkpoint and the quantizer actually hold ----------------------
    if a.node == "wq_a":
        key, N, K, wname = "wq_a_part", QL, HID, p + "wq_a"
        Aq, As = zq_split(np.fromfile(os.path.join(DD, f"attn{L}_r{r}_x_q.bin"), np.uint8), T, HID)
        Bq_t = A.read(wname + ".weight").contiguous()
        Bs_t = A.read(wname + ".scale").contiguous()
    elif a.node == "wq_b":
        key, N, K, wname = "wq_b_part", 8 * D, QL, p + "wq_b"
        Aq, As = zq_split(np.fromfile(os.path.join(DD, f"attn{L}_r{r}_qr_q.bin"), np.uint8), T, QL)
        Bq_t = A.read(wname + ".weight")[r * N:(r + 1) * N].contiguous()
        Bs_t = A.read(wname + ".scale")[r * N // 32:(r + 1) * N // 32].contiguous()
    elif a.node == "wkv":
        key, N, K, wname = "wkv_part", D, HID, p + "wkv"
        Aq, As = zq_split(np.fromfile(os.path.join(DD, f"attn{L}_r{r}_x_q.bin"), np.uint8), T, HID)
        Bq_t = A.read(wname + ".weight").contiguous()
        Bs_t = A.read(wname + ".scale").contiguous()
    else:
        raise SystemExit("wq_a, wq_b, wkv only")

    ker_buf = np.fromfile(os.path.join(DD, f"attn{L}_r{r}_{key}.bin"), np.uint8)
    npart = ker_buf.size // 4 // (T * N)
    mega = parts(ker_buf, T, N, npart).sum(0)[0]          # the consumer's fp32 sum of the parts
    print(f"{a.node}: K {K} N {N}  megakernel splits K into P={npart} parts of {K // npart}")
    print(f"operands from the checkpoint/quantizer: A e4m3 {Aq.shape}, B e4m3 {tuple(Bq_t.shape)}\n")

    Aq = Aq[0:1]
    As = As[0:1]
    Af = torch.from_numpy(Aq).view(torch.float8_e4m3fn)
    Asf = e8m0(As)

    # ---- the reference, whole K and split exactly like the megakernel -----------------------------------------
    full = run_off(off, Af, Asf, Bq_t, Bs_t, N, K)
    kp = K // npart
    split = np.zeros(N, np.float32)
    for q in range(npart):
        c = run_off(off, Af[:, q * kp:(q + 1) * kp].contiguous(), e8m0(As[:, q * kp // 32:(q + 1) * kp // 32]),
                    Bq_t[:, q * kp:(q + 1) * kp].contiguous(), Bs_t[:, q * kp // 32:(q + 1) * kp // 32], N, kp)
        split = (split + c).astype(np.float32)

    # ---- software models of the arithmetic, in fp64 where an engine would not round ----------------------------
    # per-32-block dot: every product e4m3 x e4m3 is exact (8 significant bits) and 32 of them need ~44 bits, so the
    # fp64 sum IS the exactly-rounded 32-term dot; the scale is a power of two, so scaling is exact too
    A64 = (e4[Aq[0]].reshape(K // 32, 32) * 2.0 ** (As[0].astype(np.int32) - 127)[:, None]).reshape(K)
    Braw = Bq_t.cpu().view(torch.uint8).numpy()
    bs = Bs_t.float().double().cpu().numpy()              # [N/32][K/32], already 2^(byte - 127)
    B64 = e4[Braw].reshape(N, K // 32, 32) * bs[np.arange(N) // 32][:, :, None]   # [N][K/32][32]
    block = np.einsum("bk,nbk->nb", A64.reshape(-1, 32), B64)   # [N][K/32], exact, scale already inside
    exact = block.sum(1)                                  # the correctly-rounded value
    terms = A64.reshape(-1, 32)[None, :, :] * B64          # [N][K/32][32], every product exact in fp64

    def dot32(t, how):
        """One 32-term dot the way a tensor core might do it. Each product is exact, so the fp64 sum of any two of
        them is exact and its cast to fp32 is that pair's correct rounding -- the tree is modelled faithfully."""
        if how == "exact":
            return t.sum(-1)
        if how == "seq":
            acc = t[..., 0].astype(np.float32)
            for j in range(1, t.shape[-1]):
                acc = (acc + t[..., j]).astype(np.float32)
            return acc
        x = t
        while x.shape[-1] > 1:                             # pairwise butterfly, fp32 at every node
            x = (x[..., 0::2] + x[..., 1::2]).astype(np.float32)
        return x[..., 0]

    def dot32_aligned(t, mode):
        """Blackwell's fp8/fp4 tensor cores sum the 32 products as block-floating point: every product is aligned to
        the largest exponent in the block, truncated (or rounded) to the accumulator's significand, summed as an
        integer, then scaled back. Not IEEE addition -- and not reproducible by any ordering of fp32 adds.

        `t` holds the 32 exact products; returns their block-FP sum."""
        mx = np.abs(t).max(-1, keepdims=True)
        with np.errstate(divide="ignore"):
            _, e = np.frexp(mx)                                    # mx = m * 2^e, m in [0.5, 1)
        e = np.where(mx > 0, e, 1)                                 # an all-zero block: any exponent works
        q = t * np.ldexp(1.0, 23 - e)                              # |q| <= 2^23: the accumulator's significand
        q = np.trunc(q) if mode == "trunc" else np.rint(q)
        return np.ldexp(q.sum(-1), e[..., 0] - 23)

    nb = K // 32
    per_part = nb // npart
    models = {}
    for how in ("exact", "seq", "tree", "trunc", "round"):
        blk = dot32_aligned(terms, how) if how in ("trunc", "round") else dot32(terms, how)
        acc = np.zeros(N, np.float32)
        for i in range(nb):
            acc = (acc + f32(blk[:, i])).astype(np.float32)
        models[f"32-dot {how:<5} + one fp32 accumulator over all K"] = acc
        # same dot, but K split into P accumulators and the parts added in order (the megakernel's layout)
        accp = []
        for q in range(npart):
            ac = np.zeros(N, np.float32)
            for i in range(q * per_part, (q + 1) * per_part):
                ac = (ac + f32(blk[:, i])).astype(np.float32)
            accp.append(ac)
        tot = accp[0]
        for q in range(1, npart):
            tot = (tot + accp[q]).astype(np.float32)
        models[f"32-dot {how:<5} + {npart} K-part accumulators"] = tot
        # ... and the parts interleaved (every P-th 32-block), in case that is the layout
        accp = []
        for q in range(npart):
            ac = np.zeros(N, np.float32)
            for i in range(q, nb, npart):
                ac = (ac + f32(blk[:, i])).astype(np.float32)
            accp.append(ac)
        tot = accp[0]
        for q in range(1, npart):
            tot = (tot + accp[q]).astype(np.float32)
        models[f"32-dot {how:<5} + {npart} interleaved accumulators"] = tot

    def score(tag, x):
        b = TL.bits(x, mega)
        bo = TL.bits(x, full)
        print(f"  {tag:<58} vs megakernel {b[0]:>5}/{N}   vs reference {bo[0]:>5}/{N}")

    print("the two engines:")
    score("reference, one accumulator over all K", full)
    score(f"reference, same K split (P={npart}) added in order", split)
    print(f"\n  reference(whole K) itself vs reference(split): {TL.bits(full, split)[0]}/{N}")
    print("\nsoftware models of the reduction:")
    for tag, x in models.items():
        score(tag, x)
    print(f"\n  correctly-rounded value (fp64) vs megakernel: {TL.bits(f32(exact), mega)[0]}/{N}"
          f"   vs reference: {TL.bits(f32(exact), full)[0]}/{N}")


if __name__ == "__main__":
    main()
