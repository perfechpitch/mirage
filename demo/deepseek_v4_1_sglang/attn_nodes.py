"""Attention, node by node: does each kernel output equal the checkpoint's arithmetic bit for bit?

The MoE got to `cos` exactly 1.0 (`bitexact.py`) by comparing the fp32 buffers the kernel itself wrote, instead of
inverting the bf16 residual. The attention graph exposes the same: `--dump-all attn{L}` writes every node's buffer, and
the fp32 GEMM partials are the natural place to compare.

The graph at TP8 (one swa layer, `demo/deepseek_v4_1_csa2/demo.py build_graph`), with the byte layout each buffer has:

    x_q          zq_bytes(T, 5120)        e4m3 + per-32 power-of-two scales        = act_quant(x)
    wq_a_part    T * 1280 * 4 * 4         fp32, 4 K parts                          = x_q @ wq_a^T
    qr_q         zq_bytes(T, 1280)        e4m3                                        = act_quant(rmsnorm(wq_a, q_norm))
    wq_b_part    T * 4096 * 4             fp32, 1 part (this rank's 8 heads)         = qr_q @ wq_b^T
    wkv_part     T * 512 * 4 * 2          fp32, 2 K parts                           = x_q @ wkv^T
    o_q          T * 4096 * 2             bf16 (wo_a is bf16 at load -> no act_quant) = sparse_attn(...) then inv rope
    wo_a_part    T * 1024 * 4 * 2         fp32, 2 K parts                           = o @ wo_a^T
    oa_q         zq_bytes(T, 1024)        e4m3                                        = act_quant(wo_a_part)
    wo_b_part    T * 5120 * 4 * 2         fp32, 2 K parts                           = oa_q @ wo_b^T

Every product in these GEMMs is exact in fp32 (e4m3 has 4 significant bits, times a power of two), and at most 5120 of
them are added, so the sum is exact too -- exactly the property that let `bitexact.py` reach 5120/5120 on the MoE. Any
residual here is a real difference, not round-off.

    python attn_nodes.py --layer 1 --dump /tmp/chain_attn2
"""
import argparse
import json
import os

import numpy as np
import torch

HID, QL, D, OL = 5120, 1280, 512, 1024
T = int(os.environ.get("STATIC_TOKENS", "8"))
SF_CHUNK = 512                # bytes a 128-column scale chunk takes (32 token slots x 4 blocks)
CK = "/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017"
_IDX = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]


def rr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def bits_equal(a, b):
    a = np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint32)
    b = np.ascontiguousarray(np.asarray(b, np.float32)).view(np.uint32)
    return int((a == b).sum()), int(a.size)


def read(name):
    from safetensors import safe_open
    with safe_open(os.path.join(CK, _IDX[name]), "pt") as sf:
        return sf.get_tensor(name)


def deq_fp8(name, rows=slice(None), cols=slice(None)):
    """An e4m3 weight with 32 x 32 e8m0 block scales, as fp64 (exact: a 4-bit mantissa x 2^k).

    `.float()` on a float8_e8m0fnu tensor already yields the multiplier 2^(byte - 127) -- do NOT apply pow(2, s - 127)
    to it a second time (that gives 2^-126 and every weight comes out ~1e-37).
    """
    w = read(name + ".weight")[rows, cols].float().double()
    r0, c0 = rows.start or 0, cols.start or 0
    R, C = w.shape
    s = read(name + ".scale")[r0 // 32:r0 // 32 + R // 32, c0 // 32:c0 // 32 + C // 32].float().double()
    return (w.view(R // 32, 32, C // 32, 32) * s[:, None, :, None]).reshape(R, C).numpy()


def rne_e4m3(v):
    """fp64 -> the nearest e4m3 value (RNE, satfinite at +-448), still as fp64.

    Uses torch's float8_e4m3fn cast, which is the hardware's `cvt.rn.satfinite.e4m3x2.f32`. A hand-rolled version must
    NOT normalize with frexp: frexp returns a mantissa in [0.5, 1), i.e. a binade one too high, so rounding the mantissa
    to 1/8 quantizes on a 1/4-of-a-binade grid (8 instead of 4 in [32, 64)) and misses half the representable values.
    """
    t = torch.from_numpy(np.ascontiguousarray(v, np.float32)).to(torch.float8_e4m3fn)
    return t.float().double().numpy()


def act_quant(x):
    """tasks/act_quant.cuh: e4m3 per 32-column block with a power-of-two (e8m0) scale, RNE -> dequantized fp64."""
    x = np.asarray(x, np.float64)
    shape = x.shape
    v = x.reshape(-1, 32)
    amax = np.maximum(np.abs(v).max(1), 1e-4)
    ex = np.ceil(np.log2(amax / 448.0))
    return (rne_e4m3(v / (2.0 ** ex)[:, None]) * (2.0 ** ex)[:, None]).reshape(shape)


_TBL = None


def e4m3_table(bytes_):
    """e4m3 (1-4-3: no inf, 0x7F/0xFF = NaN) -> fp64, through a 256-entry table."""
    global _TBL
    if _TBL is None:
        _TBL = np.zeros(256, np.float64)
        for b in range(256):
            s = -1.0 if (b >> 7) else 1.0
            e, m = (b >> 3) & 0xF, b & 0x7
            if e == 0:
                val = (m / 8.0) * 2.0 ** -6
            elif e == 0xF and m == 0x7:
                val = np.nan
            else:
                val = (1.0 + m / 8.0) * 2.0 ** (e - 7)
            _TBL[b] = s * val
    return _TBL[np.asarray(bytes_, np.uint8)]


def decode_zq(buf, tokens, width):
    """A z_q buffer ([tokens][width] e4m3, then one 512-B scale chunk per 128 columns) -> fp64 [tokens][width]."""
    raw = buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)
    a = np.frombuffer(raw, np.uint8)
    assert a.size == tokens * width + (width // 128) * SF_CHUNK, (a.size, tokens * width)
    qf = e4m3_table(a[:tokens * width]).reshape(tokens, width)
    sc = a[tokens * width:].reshape(width // 128, SF_CHUNK)
    out = np.empty((tokens, width), np.float64)
    for c in range(width // 128):
        for t in range(tokens):
            for kb in range(4):
                b = sc[c, (t % 32) * 16 + 4 * (t // 32) + kb]
                out[t, c * 128 + kb * 32:c * 128 + (kb + 1) * 32] = \
                    qf[t, c * 128 + kb * 32:c * 128 + (kb + 1) * 32] * 2.0 ** (float(b) - 127.0)
    return out


def parts(buf, tokens, width, nparts):
    """A fp32 GEMM node: the output buffer's [nparts][tokens][width] K parts -> fp64."""
    raw = buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)
    a = np.frombuffer(raw, np.float32)
    assert a.size == nparts * tokens * width, (a.size, nparts * tokens * width)
    return a.reshape(nparts, tokens, width).astype(np.float64)


def bf16r(v):
    """Round fp32/fp64 to bf16 (RNE) and widen back."""
    return torch.from_numpy(np.ascontiguousarray(v, np.float32)).bfloat16().float().double().numpy()


def rms_norm_parts8(part_sum, gamma, eps, quant=True):
    """tasks/q_norm.cuh rms_norm_parts8, emulated including its summation order.

    The K parts are added in part order, then rounded to bf16 (that rounding is the GEMM's bf16 output). The mean of
    squares is NOT torch's: each square is rounded in fp32 and they are added in a fixed order -- a thread's 8 columns
    in order, then a 32-lane butterfly, then the warps in order (5 warps at K = 1280). `y = bf16((x * r) * gamma)`, and
    the two multiplies are separate roundings. `quant=False` stops at the bf16 `y` (kv_post.cuh ropes it before the
    act_quant, so it cannot use the quantized form)."""
    x = np.asarray(part_sum, np.float64)                  # [T][K] fp32 sums of the parts, in part order
    K = x.shape[1]
    x = bf16r(x)
    T_ = x.shape[0]
    NW = K // 256
    y = np.empty_like(x)
    for t in range(T_):
        ss = np.zeros(256, np.float32)
        for th in range(256):                              # thread th owns columns 8 th .. 8 th + 7
            c = 8 * th
            if c >= K:
                continue
            for k in range(8):
                xk = np.float32(x[t, c + k])
                ss[th] = np.float32(ss[th] + np.float32(xk * xk))
        red = np.zeros(NW, np.float32)
        for w in range(NW):
            lane = ss[w * 32:(w + 1) * 32].copy()
            for d in (16, 8, 4, 2, 1):                     # __shfl_xor_sync butterfly
                lane = (lane + lane[np.arange(32) ^ d]).astype(np.float32)
            red[w] = lane[0]
        tot = red[0]
        for i in range(1, NW):
            tot = np.float32(tot + red[i])
        r = np.float32(np.float32(1.0) / np.sqrt(np.float32(np.float32(tot) * np.float32(1.0 / K) + np.float32(eps))))
        y[t] = bf16r((x[t] * r) * np.asarray(gamma, np.float64))
    return act_quant(y) if quant else y


def bf16(buf, tokens, width):
    raw = buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)
    t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
    return t.numpy().reshape(tokens, width).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--dump", default="/tmp/chain_attn2")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-20)
    ap.add_argument("--only", default=None, help="a comma list of nodes to run (default: all)")
    a = ap.parse_args()
    L, DD, r = a.layer, a.dump, a.rank
    p = f"layers.{L}.attn."
    only = set(a.only.split(",")) if a.only else None

    def want(k):
        return only is None or k in only

    def node(key):
        f = os.path.join(DD, f"attn{L}_r{r}_{key}.bin")
        return np.fromfile(f, np.uint8) if os.path.exists(f) else None

    x = bf16(open(os.path.join(DD, f"attn{L}_x.bin"), "rb").read(), T, HID)
    print(f"attn{L}, rank {r}: x [T={T}, {HID}]  |x| rms {np.sqrt((x ** 2).mean()):.6f}\n")

    done = []

    def rep(tag, mine, ker):
        if ker is None:
            print(f"  {tag:<14} (no dump)")
            return False
        m, k = np.asarray(mine, np.float64), np.asarray(ker, np.float64)
        if m.shape != k.shape:
            print(f"  {tag:<14} SHAPE {m.shape} vs {k.shape}")
            return False
        b, n = bits_equal(m, k)
        done.append((tag, rr(m, k), cos(m, k), b, n))
        print(f"  {tag:<14} relRMS {rr(m, k):>9.3e}  cos {cos(m, k):+.9f}  bits {b:>6}/{n}")
        return True

    # 1. x_q = act_quant(x)   (x is already bf16, so act_quant's bf16 round is a no-op)
    xq = act_quant(x)
    if want("x_q"):
        rep("x_q", xq, decode_zq(node("x_q"), T, HID) if node("x_q") is not None else None)

    # 2. wq_a_part: a fp32 GEMM node whose buffer holds the plan's P K parts ([P][T][K]). Which K slice each part
    # covers is a scheduling choice, so the semantic quantity is their sum -- that is what q_norm reads.
    wq_a = deq_fp8(p + "wq_a")
    wq_a_sum = None
    if want("wq_a"):
        nk = node("wq_a_part")
        if nk is not None:
            wq_a_sum = parts(nk, T, QL, 4).sum(0)
            rep("wq_a sum", xq @ wq_a.T, wq_a_sum)

    # 3. qr_q = act_quant(bf16((x * r) * gamma)), x = bf16(sum of the parts)
    qr = None
    if want("qr_q"):
        if wq_a_sum is None:
            nk = node("wq_a_part")
            wq_a_sum = parts(nk, T, QL, 4).sum(0) if nk is not None else None
        if wq_a_sum is not None:
            qr = rms_norm_parts8(wq_a_sum, read(p + "q_norm.weight").float().numpy(), a.eps)
            rep("qr_q", qr, decode_zq(node("qr_q"), T, QL) if node("qr_q") is not None else None)

    # 4. wq_b_part: this rank's 8 heads, 1 part
    if want("wq_b"):
        nk = node("wq_b_part")
        if nk is not None:
            if qr is None:
                qr = decode_zq(node("qr_q"), T, QL)
            wq_b = deq_fp8(p + "wq_b", rows=slice(r * 8 * D, (r + 1) * 8 * D))
            rep("wq_b", qr @ wq_b.T, parts(nk, T, 8 * D, 1)[0])

    # 5. wkv_part: 2 K parts
    if want("wkv"):
        nk = node("wkv_part")
        if nk is not None:
            wkv = deq_fp8(p + "wkv")
            rep("wkv sum", xq @ wkv.T, parts(nk, T, D, 2).sum(0))

    o = None
    if want("o_q"):
        nk = node("o_q")
        if nk is not None:
            o = bf16(nk.tobytes(), T, 8 * D)
            print(f"  {'o_q':<14} rms {np.sqrt((o ** 2).mean()):.6f}  (the core: attn_core.py replays the official's"
                  f" two-block softmax; ~2 bf16 ulp is the floor)")

    # 7. wo_a_part: 2 K parts, o (bf16) @ wo_a^T -- SGLang's bf16 fast path, so no act_quant on o
    wa_sum = None
    if want("wo_a"):
        nk = node("wo_a_part")
        if nk is not None:
            if o is None:
                o = bf16(node("o_q").tobytes(), T, 8 * D)
            wo_a = deq_fp8(p + "wo_a", rows=slice(r * OL, (r + 1) * OL))
            wa_sum = parts(nk, T, OL, 2).sum(0)
            rep("wo_a sum", o @ wo_a.T, wa_sum)

    # 8. oa_q = act_quant(bf16(wo_a_sum)): act_quant.cuh rounds the GEMM's fp32 output to bf16 before the formula
    oa = None
    if want("oa_q"):
        nk = node("oa_q")
        if nk is not None:
            if wa_sum is None:
                wa_sum = parts(node("wo_a_part"), T, OL, 2).sum(0)
            oa = act_quant(bf16r(wa_sum))
            rep("oa_q", oa, decode_zq(nk, T, OL))

    # 9. wo_b_part: 2 K parts
    if want("wo_b"):
        nk = node("wo_b_part")
        if nk is not None:
            if oa is None:
                oa = decode_zq(node("oa_q"), T, OL)
            wo_b = deq_fp8(p + "wo_b", cols=slice(r * OL, (r + 1) * OL))
            rep("wo_b sum", oa @ wo_b.T, parts(nk, T, HID, 2).sum(0))

    if done:
        print(f"\n{len(done)} nodes: worst relRMS {max(d[1] for d in done):.3e}"
              f"  lowest cos {min(d[2] for d in done):+.9f}"
              f"  fully bit-exact {sum(1 for d in done if d[3] == d[4])}/{len(done)}")


if __name__ == "__main__":
    main()
