"""The five GEMM nodes, against the official fp8 GEMM: bit for bit, or only reassociation?

`tilelang_check.py` closed `o_q`: the official tilelang `sparse_attn` reproduces the megakernel's `o_q` 32768/32768
bits. The remaining nodes are the GEMMs, at `cos` exactly 1.000000000 but 4e-8 relRMS -- 6% of the elements one fp32
ulp off, which looks like reassociation. The official `inference/kernel.py fp8_gemm_kernel` settles it: it is the
reference's own arithmetic, and its shape is

    per 32-column K block:  C_local = A_fp8 @ B_fp8^T   (fp8 MMA, fp32 accumulate, 32 terms -> exact)
                            C_acc  += C_local * scale_a[i,k] * scale_b[j,k]     (scales are powers of two)
    over the K blocks in order, one fp32 accumulator

which is the same recipe the megakernel's `gemm_fp8_tile` follows. What can still differ is (a) the tensor core's
32-term reduction and (b) the megakernel's P K parts, which are summed by the consumer (q_norm's `add_parts8`) instead
of by the running accumulator. So the comparison reports both: against the parts' sum, and against each part.

    python tilelang_gemm.py --layer 1 --dump /tmp/chain_attn2 --rank 0
"""
import argparse
import os

import numpy as np
import torch

import attn_np as A
import tilelang_check as TL
from attn_nodes import HID, QL, D, OL, deq_fp8, parts

CK = A.CK


def zq_split(zq, tokens, width):
    """A z_q buffer -> (e4m3 bytes [tokens][width], the per-32-column scale bytes [tokens][width / 32])."""
    a = np.frombuffer(zq.tobytes() if hasattr(zq, "tobytes") else bytes(zq), np.uint8)
    q = a[:tokens * width].reshape(tokens, width)
    sc = a[tokens * width:].reshape(width // 128, 512)
    out = np.empty((tokens, width // 32), np.uint8)
    for c in range(width // 128):
        for t in range(tokens):
            for kb in range(4):
                out[t, c * 4 + kb] = sc[c, (t % 32) * 16 + 4 * (t // 32) + kb]
    return q, out


def e8m0(u8, device="cuda"):
    """The raw 8-bit scale bytes as a float8_e8m0fnu tensor (casting it to fp32 yields 2^(byte - 127))."""
    return torch.from_numpy(np.ascontiguousarray(u8)).view(torch.float8_e8m0fnu).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--dump", default="/tmp/chain_attn2")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--nodes", default="wq_a,wq_b,wkv,wo_b")
    a = ap.parse_args()
    L, DD, r = a.layer, a.dump, a.rank
    T = 8
    p = f"layers.{L}.attn."
    off = TL.off_kernel()
    print(f"official: {os.path.join(CK, 'inference', 'kernel.py')} fp8_gemm_kernel(block_size=32, out_dtype=fp32)\n")

    def rd(key):
        f = os.path.join(DD, f"attn{L}_r{r}_{key}.bin")
        return np.fromfile(f, np.uint8) if os.path.exists(f) else None

    def run_gemm(Aq, As, Bq, Bs, N, K, nparts):
        """The official kernel on the megakernel's own bytes, then compare with the sum of the megakernel's K parts."""
        kern = off.fp8_gemm_kernel(N, K, block_size=32, out_dtype=off.FP32, accum_dtype=off.FP32,
                                   scale_dtype=off.FE8M0)
        c = torch.empty(T, N, dtype=torch.float32, device="cuda")
        kern(Aq.to("cuda"), Bq.to("cuda"), c, As, Bs.to("cuda"))
        return c.cpu().numpy()

    for node in a.nodes.split(","):
        if node == "wq_a":
            key, width, N = "wq_a_part", QL, QL
            Aq, As = zq_split(rd("x_q"), T, HID)
            wname, rows = p + "wq_a", slice(None)
            K = HID
        elif node == "wq_b":
            key, width, N = "wq_b_part", QL, 8 * D
            Aq, As = zq_split(rd("qr_q"), T, QL)
            wname, rows = p + "wq_b", slice(r * 8 * D, (r + 1) * 8 * D)
            K = QL
        elif node == "wkv":
            key, width, N = "wkv_part", D, D
            Aq, As = zq_split(rd("x_q"), T, HID)
            wname, rows = p + "wkv", slice(None)
            K = HID
        elif node == "wo_b":
            key, width, N = "wo_b_part", OL, HID
            Aq, As = zq_split(rd("oa_q"), T, OL)
            wname, rows = p + "wo_b", slice(None)
            K = OL
        else:
            continue

        ker = rd(key)
        if ker is None:
            print(f"{node}: no dump for {key}")
            continue
        Bw = A.read(wname + ".weight")[rows].contiguous()
        Bs = A.read(wname + ".scale")
        b0 = rows.start or 0
        Bs = Bs[b0 // 32:(b0 + N) // 32].contiguous()
        if node == "wo_b":                      # wo_b's K is the group axis, so its scale columns shard, not rows
            Bw = A.read(wname + ".weight")[:, r * OL:(r + 1) * OL].contiguous()
            Bs = A.read(wname + ".scale")[:, r * OL // 32:(r + 1) * OL // 32].contiguous()

        mine_official = run_gemm(torch.from_numpy(Aq).view(torch.float8_e4m3fn), e8m0(As), Bw, Bs, N, K, 0)

        # the megakernel's buffer: [P][T][N] fp32
        npart = ker.size // 4 // (T * N)
        prt = parts(ker, T, N, npart)
        s_part = prt.sum(0)
        # only token 0 carries data (the chain runs one token; rows 1..7 are zeros), so score token 0 -- otherwise
        # the 7 empty rows inflate every count to exactly 7/8
        s0, o0 = s_part[0], mine_official[0]
        b_sum = TL.bits(s0.astype(np.float32), o0)
        per = [TL.bits(prt[i][0].astype(np.float32), o0)[0] for i in range(npart)]

        # the exactly-rounded value: every product e4m3 x e4m3 is exact in fp64, the scales are powers of two, and
        # 5120 of them fit in its 53 bits -- so this is the ground truth both engines are approximating
        e4 = np.zeros(256, np.float64)
        for b in range(256):
            sg = -1.0 if b >> 7 else 1.0
            ex_, mn_ = (b >> 3) & 0xF, b & 0x7
            if ex_ == 0:
                v = (mn_ / 8.0) * 2.0 ** -6
            elif ex_ == 0xF and mn_ == 0x7:
                v = np.nan
            else:
                v = (1.0 + mn_ / 8.0) * 2.0 ** (ex_ - 7)
            e4[b] = sg * v
        A64 = e4[Aq] * np.repeat(2.0 ** (As.astype(np.int32) - 127), 32, axis=1)   # [T, K]
        Braw = Bw.cpu().contiguous().view(torch.uint8).numpy()
        bs = Bs.float().double().cpu().numpy()                                  # already 2^(byte - 127)
        B64 = (e4[Braw].reshape(N // 32, 32, K // 32, 32) * bs[:, None, :, None]).reshape(N, K)
        exact = A64 @ B64.T

        def ulps(x):
            """|x - exact| in units of an fp32 ulp at the tensor's largest element (robust near cancellations, where a
            per-element ulp would be meaningless)."""
            x = np.asarray(x, np.float64)
            return np.abs(x - exact[0]) / (np.abs(exact[0]).max() * 2.0 ** -24)

        um, uo = ulps(s0), ulps(o0)
        # The interface value, not a softened test: every one of these nodes' consumers starts by rounding the fp32
        # sum to bf16 -- q_norm.cuh `bf16r(x[k]);  // the GEMM's output`, kv_post.cuh `-> bf16 (wkv's bf16 output)`,
        # sparse_attn.cuh `x[e] = bf16r(x[e]);  // wq_b's output`, and wo_b's sum_store writes y_sent as bfloat16.
        # So bf16(sum) IS the number that crosses the boundary; the fp32 1-ulp difference below it is unobservable
        # downstream. That does NOT make the GEMMs bit-exact -- the line above says so.
        bfw_m = torch.from_numpy(s0.astype(np.float32)).bfloat16().float()
        bfw_o = torch.from_numpy(o0.astype(np.float32)).bfloat16().float()
        bb = TL.bits(bfw_m.numpy(), bfw_o.numpy())
        detail = ""
        flip = np.flatnonzero(bfw_m.numpy() != bfw_o.numpy())
        if flip.size:
            n_ = int(flip[0])
            mid = (float(bfw_m[n_]) + float(bfw_o[n_])) / 2.0
            sp = float(np.spacing(np.float32(s0[n_])))
            detail = (f"\n       the flip: n{n_}, megakernel {s0[n_]:.9e} vs official {o0[n_]:.9e}; the two bf16"
                      f" neighbours' midpoint is {mid:.9e}, and they sit {abs(s0[n_] - mid) / sp:.0f} /"
                      f" {abs(o0[n_] - mid) / sp:.0f} fp32 ulp either side of it")
            detail += (f"\n       cancellation: |result| {abs(exact[0, n_]):.3e} against terms of scale"
                       f" {np.abs(A64[0]).max() * np.abs(B64[n_]).max():.3e}")
        print(f"{node:<6} K {K:>5} N {N:>5}  parts P={npart}   (token 0 only: rows 1..7 are zeros)")
        print(f"       fp32, the buffer's own dtype  : {b_sum[0]:>6}/{b_sum[1]} bits identical"
              f"   relRMS {TL.rr(s0, o0):.3e}      <- the honest score; NOT bit-exact")
        print(f"       the interface value (bf16)    : {bb[0]:>6}/{bb[1]} bits identical" + detail)
        print(f"       official vs each part    : " + " ".join(f"{v}/{N}" for v in per)
              + "   (a part is a K slice, so it should NOT match)")
        print(f"       ulps from the exact value : megakernel mean {um.mean():.3f}"
              f"   | official mean {uo.mean():.3f}")


if __name__ == "__main__":
    main()
