"""One swa attention sublayer of DeepSeek-V4.1, recomputed from the reference dump and the checkpoint weights.

Why: DSV41_TEACH's table says every attention sublayer from L1 on is 5-22% off while L0 is 0.26%, with the input
residual exact. This tool rebuilds the sublayer's own numbers outside the kernel, so the first wrong QUANTITY can be
named instead of guessed: q (after wq_a | q_norm | wq_b | rope), the K rows (the reference pool's, plus the current
token's own k), the sink softmax, o, and the inverse rope.

Everything comes from the reference side (the pool, the weights, the position), except `--x`, which is the row the
sublayer reads (the chain's own dump: attn{L}_x.bin, verified to equal SGLang's input for the teacher run).

  python attn_np.py --layer 0          # expect o to match the chain's o (attn0_r0_o_q.bin) to ~1e-3
  python attn_np.py --layer 1
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

HID, D, RD, HEADS, GROUPS, OL = 5120, 512, 64, 64, 8, 1024
W = 128                    # window_size
HERE = os.path.dirname(os.path.abspath(__file__))
CK = "/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017"
PRE = "/tmp/dec_coef8"


def rr(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-30))


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


_IDX = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]


def read(name):
    with safe_open(os.path.join(CK, _IDX[name]), "pt") as sf:
        return sf.get_tensor(name)


def deq(name, rows=slice(None), cols=slice(None)):
    """An e4m3 weight (32 x 32 e8m0 block scales) as fp32: exact (an e4m3 value times 2^k has 4 significant bits).
    `rows`/`cols` must be whole 32-blocks (the caller's slices are: 8 heads x 512 rows)."""
    w = read(name + ".weight")[rows, cols].float()
    r0, c0 = rows.start or 0, cols.start or 0
    R, C = w.shape
    s = read(name + ".scale")[r0 // 32:r0 // 32 + R // 32, c0 // 32:c0 // 32 + C // 32].float()
    return (w.view(R // 32, 32, C // 32, 32) * s[:, None, :, None]).reshape(R, C)


def rmsnorm(x, g, eps):
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * g.float()).bfloat16()


def rope_apply(x, pos, theta=10000.0, inverse=False):
    """model.py apply_rotary_emb on the last RD dims: adjacent pairs as complex numbers, plain RoPE."""
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, RD, 2, dtype=torch.float32) / RD))
    ang = pos * inv_freq
    if inverse:
        ang = -ang
    c, s = torch.cos(ang), torch.sin(ang)
    y = x.clone()
    v = y[..., d - RD:].float().view(-1, RD // 2, 2)
    re, im = v[..., 0].clone(), v[..., 1].clone()
    v[..., 0] = re * c - im * s
    v[..., 1] = re * s + im * c
    y[..., d - RD:] = v.reshape(-1, RD).bfloat16()
    return y


def act_quant_deq(x, block=32):
    """kernel.py act_quant(inplace=True): e4m3, `block`-column blocks, power-of-two (e8m0) scales -> bf16."""
    x = x.float()
    v = x.view(-1, block)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    ex = torch.ceil(torch.log2(amax / 448.0))
    q = (v / torch.pow(2.0, ex)).clamp(-448, 448).to(torch.float8_e4m3fn).float() * torch.pow(2.0, ex)
    return q.reshape(x.shape).bfloat16()


def window_kv(pos, L):
    """The 128 K rows of layer L's ring in the official's list order (oldest first): [(position, pool row or None)].

    A row is 528 bytes: 512 e4m3 + 16 e8m0 block scales, one token's entry in a 256-token pool page
    (`swa_bytes_per_token`, `swa_page`); the entry for cache loc c sits at page c // 256, slot c % 256. The ring slot of
    position q is q % W, so the list position for slot s is (s - pos) % W. `None` is the current token (pos itself): the
    decode step writes its K this launch, so it is not in the dump -- the caller computes it.

    The pool is PER LAYER (meta's L{L}.window_cache): reading another layer's rows is the same numbers at L0 and
    anything at L>=1 (verified 11-27: with L0's file, L1's o came out at cos -0.13)."""
    meta = json.load(open(os.path.join(PRE, "meta_rank00.json")))["buffers"]
    addr = json.load(open(os.path.join(PRE, "meta_addr_rank00.json")))
    per_page = 135168 // 528
    pool = np.fromfile(os.path.join(PRE, meta[f"L{L}.window_cache"]["file"]), dtype=np.uint8)
    pool = pool.reshape(-1, per_page, 528)
    loc = {int(p_): int(c) for p_, c in zip(addr["positions"], addr["swa_out_cache_loc"])}
    rows = {}
    for p_ in range(pos - W + 1, pos):
        if p_ in loc:
            c = loc[p_]
            rows[p_ % W] = pool[c // per_page, c % per_page]
    order = [(pos % W + 1 + i) % W for i in range(W)]          # the official's get_window_topk_idxs
    out = []
    for s in order:
        q = pos - ((pos - s) % W)                              # the newest position in that ring slot
        if q == pos:
            out.append((pos, None))
        else:
            assert q in loc, f"window slot {s} (position {q}) is missing from the pool"
            out.append((q, rows[s]))
    return out


def deq_row(b):
    """A pool row's 512 values: e4m3 bytes times 2^(e8m0 byte - 127), as bf16 (SGLang's cache holds bf16)."""
    q = torch.frombuffer(bytearray(b[:512].tobytes()), dtype=torch.float8_e4m3fn).float()
    sc = torch.frombuffer(bytearray(b[512:528].tobytes()), dtype=torch.uint8).int() - 127
    return (q.view(16, 32) * torch.pow(2.0, sc.float())[:, None]).reshape(512).bfloat16()


def gemm_ref(o_all_paths, L, pos, quant=128):
    """o (all 8 ranks' [8, 512] rows, 64 heads) -> s [5120], as the official's wo_a/wo_b with SGLang's bf16 wo_a:
        per group g (8 heads = 4096 columns): a_g = o_g . wo_a_g^T   [1024]
        act_quant(a_g, block 128 -- the chain's oa_q is 8 x 1024 + 8 x 512 bytes)
        s += wo_b[:, g * 1024 : (g + 1) * 1024] . a_g^T
    wo_a is block-diagonal over the groups, so each group only sees its own 4096 columns."""
    p = f"layers.{L}.attn."
    s = torch.zeros(HID)
    for g in range(GROUPS):
        # one GPU per group (TP8 = 8 groups): rank g's 8 heads ARE group g's 4096 columns
        o = torch.frombuffer(bytearray(open(o_all_paths[g], "rb").read()), dtype=torch.bfloat16).float()
        o = o.reshape(-1, HEADS // 8, D)[0].reshape(-1)
        # wo_a is [GROUPS * OL, 8 * D] (each group's own 8 heads = all 4096 columns, not a block-diagonal of 32768)
        wa = deq(p + "wo_a", rows=slice(g * OL, (g + 1) * OL))
        aq = act_quant_deq(wa @ o.float(), block=quant)
        wb = deq(p + "wo_b", cols=slice(g * OL, (g + 1) * OL))
        s += wb @ aq.float()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pos", type=int, default=31906)
    ap.add_argument("--x", default=None, help="the row the sublayer reads, bf16 [T, 5120] (default: chain_f2's dump)")
    ap.add_argument("--chain", default="/tmp/chain_f2", help="the chain's dump dir, for attn{L}_r0_o_q.bin (the o)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-20)
    ap.add_argument("--gemm", action="store_true",
                    help="also run the o -> wo_a -> act_quant -> wo_b reference on the CHAIN's o (all 8 ranks)")
    a = ap.parse_args()
    L, pos, r = a.layer, a.pos, a.rank

    xf = a.x or os.path.join(a.chain, f"attn{L}_x.bin")
    x = torch.from_numpy(np.frombuffer(open(xf, "rb").read(), dtype=np.uint16)).view(torch.bfloat16).float()
    x = x.reshape(-1, HID)[0].float()

    p = f"layers.{L}.attn."
    q_norm_w = read(p + "q_norm.weight")
    kv_norm_w = read(p + "kv_norm.weight")
    sink = read(p + "attn_sink").float()

    q0 = r * (HEADS // 8) * D                       # this rank's 8 heads' wq_b rows
    qr = rmsnorm(deq(p + "wq_a") @ x, q_norm_w, a.eps)
    q = (deq(p + "wq_b", rows=slice(q0, q0 + HEADS // 8 * D)) @ qr.float())     # [8*512]
    q = q.view(HEADS // 8, D)
    q = rope_apply(q, pos)                                                     # [8, 512], bf16

    # the current token's own K: kv_norm(wkv(x)) -> rope -> act_quant (the prefill's rows are already in the pool)
    k_now = rmsnorm(deq(p + "wkv") @ x, kv_norm_w, a.eps)
    k_now = act_quant_deq(rope_apply(k_now, pos))                              # [512]
    lst = window_kv(pos, L)
    kv = torch.stack([k_now if b is None else deq_row(b) for _p, b in lst])     # [128, 512] bf16
    assert lst[-1][1] is None, "the current token is the last list position (newest)"

    s = (q.float() @ kv.float().t()) * (D ** -0.5)                              # [8, 128]
    m = s.amax(1)
    pex = torch.exp(s - m[:, None])
    den = pex.sum(1) + torch.exp(sink[r * 8:r * 8 + 8].float() - m)
    o = (pex.bfloat16().float() @ kv.float()) / den[:, None]                    # [8, 512]
    o = rope_apply(o.bfloat16(), pos, inverse=True)                             # [8, 512] bf16

    co = os.path.join(a.chain, f"attn{L}_r{r}_o_q.bin")
    if os.path.exists(co):
        raw = open(co, "rb").read()
        c = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
        c = c.reshape(-1, HEADS // 8, D)[0]                                     # [8, 512]
        on, cn = o.float().numpy(), c.numpy()
        print(f"L{L} o: chain rms {cn.std():.5f}  numpy rms {on.std():.5f}"
              f"  relRMS {rr(on, cn):.4f}  cos {cos(on, cn):+.6f}")
        ph = [cos(on[h], cn[h]) for h in range(HEADS // 8)]
        print("   per head cos: " + " ".join(f"{v:+.4f}" for v in ph))
        print(f"   noPE cols relRMS {rr(on[:, :D-RD], cn[:, :D-RD]):.4f}"
              f"   rope cols relRMS {rr(on[:, D-RD:], cn[:, D-RD:]):.4f}")
    else:
        print(f"no {co}: the chain's o is not dumped (rerun full_decode.py --dump-all attn{L})")
    qn = q.float()
    print(f"L{L} qr rms {qr.float().square().mean().sqrt():.5f}  q rms {qn.square().mean().sqrt():.5f}"
          f"  k_now rms {k_now.float().square().mean().sqrt():.5f}  kv rms {kv.float().square().mean().sqrt():.5f}")
    print(f"L{L} scores: max {s.max():.3f} min {s.min():.3f}  den {den.min():.4f}..{den.max():.4f}")

    if a.gemm:
        paths = [os.path.join(a.chain, f"attn{L}_r{k}_o_q.bin") for k in range(8)]
        if not all(os.path.exists(f) for f in paths):
            raise SystemExit("--gemm needs all 8 ranks' attn{L}_r*_o_q.bin (rerun full_decode.py --dump-all attn{L})")
        gold = np.fromfile(os.path.join(PRE, "layers", "step1", f"L{L:02d}_apost_x.f32"), dtype=np.float32)
        for q in (128, 32):
            sp = gemm_ref(paths, L, pos, quant=q)
            print(f"L{L} s from the CHAIN's o (o·wo_a, act_quant block {q}, wo_b): rms {sp.square().mean().sqrt():.5f}"
                  f"  vs golden L{L:02d}_apost_x relRMS {rr(sp.numpy(), gold):.4f} cos {cos(sp.numpy(), gold):+.6f}")


if __name__ == "__main__":
    main()
