"""Call SGLang's real decode kernel (FlashMLA from the sgl-flash-mla wheel) offline, on L0's own data.

Why: the decode attention SGLang actually runs is NOT `inference/kernel.py`'s tilelang `sparse_attn_kernel`. That one
is the checkpoint's reference implementation, and §12.19 proved the megakernel is bit-identical to it. The dispatch in
`sglang/srt/layers/attention/deepseek_v4_backend.py` instead sends `is_dsv41 and is_sm100` decode to
`flash_mla_with_kvcache` (`use_legacy=not swa_interleaved`, i.e. False for the v41_new pool). §12.35 left the L0 y gap
at 4.98e-3 = 1.3 bf16 ulp with no faithful arithmetic variant able to do better -- so the remaining question is what
*that* kernel computes on the same bytes.

Everything is the reference's own state except the current token's K (the decode step writes it, so the dump predates
it -- `window_kv` says so too):

    pool   /tmp/dec_coef8/r00_L{L:02d}_window_cache.bin   [rows, page * 528] uint8, v41_new (512 e4m3 + 16 e8m0)
    loc    meta_addr_rank00.json: positions -> swa_out_cache_loc   (loc = pos + 256 for this request)
    order  meta_addr_rank00.json: swa_page_indices (row seq_lens_casual == pos), NEWEST FIRST
    q      bf16(rope(bf16(sum wq_b parts)))            <- attn{L}_r{r}_wq_b_part.bin
    sink   layers.{L}.attn.attn_sink[r * 8 : r * 8 + 8]

The noPE columns [0, 448) are independent of the inverse-RoPE convention, so they are the first thing to read: a
difference there is the attention arithmetic itself, not a rope disagreement.

    l0_flashmla.py --pos 31906 --layer 0 --ranks 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import attn_np as A                      # noqa: E402  (weight reader, window_kv, deq_row)
import tilelang_check as TL              # noqa: E402  (parts, bf16r, rope_fma, freqs_at, act_quant helpers)

PRE = "/tmp/dec_coef8"
D, RD, W = 512, 64, 128
BYTES = 528                              # v41_new: 512 e4m3 + 16 e8m0 (one per 32 values)


def pool_and_indices(L, pos):
    meta = json.load(open(os.path.join(PRE, "meta_rank00.json")))["buffers"]
    addr = json.load(open(os.path.join(PRE, "meta_addr_rank00.json")))
    shape = meta[f"L{L}.window_cache"]["shape"]
    page = addr["pool"]["swa_page"]
    rows = shape[0]
    pool = np.fromfile(os.path.join(PRE, meta[f"L{L}.window_cache"]["file"]), dtype=np.uint8)
    pool = pool.reshape(rows, page, 1, BYTES)                      # flash_mla: (blocks, page_block, h_k=1, bytes)

    locs = addr["swa_out_cache_loc"]
    poss = addr["positions"]
    loc_of = {int(p): int(c) for p, c in zip(poss, locs)}
    # the decode token's own slot: the request's locs are contiguous, so it is one past the last prefill token
    own = loc_of.get(pos)
    if own is None:
        own = max(loc_of.values()) + 1
    # swa_page_indices of the prefill row that ends at `pos` is NEWEST FIRST; rebuild the same shape for this step
    idx = np.array([own - i for i in range(W)], dtype=np.int32)
    return pool, own, idx, addr


def k_row_bytes(L, rank, pos, eps=1e-20):
    """The current token's cache entry, built exactly as kv_post.cuh -> act_quant does: 528 bytes."""
    p = f"layers.{L}.attn."
    kvn = A.read(p + "kv_norm.weight").float().numpy()
    c, s = TL.freqs_at(pos)
    T = 8
    wkv = TL.parts(np.fromfile(os.path.join(DUMP, f"attn{L}_r{rank}_wkv_part.bin"), np.uint8),
                   T, D, 2)[:, 0, :].sum(0)
    k = TL.rms_norm_parts8(wkv[None, :], kvn, eps, quant=False)
    k = TL.rope_fma(torch.from_numpy(k), c, s)[0]                   # [512] bf16, rope'd
    kv = torch.from_numpy(np.asarray(k, np.float32))
    v = kv.reshape(16, 32)                                          # 16 blocks of 32 values
    amax = v.abs().amax(1).clamp_min(1e-4)
    ex = torch.ceil(torch.log2(amax / 448.0))
    q = (v / torch.pow(2.0, ex)[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).numpy().tobytes() + (ex + 127).to(torch.uint8).numpy().tobytes()


def bits(a, b):
    a = np.ascontiguousarray(np.asarray(a, np.float32)).view(np.uint32)
    b = np.ascontiguousarray(np.asarray(b, np.float32)).view(np.uint32)
    return int((a == b).sum()), int(a.size)


def rr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/chain_a0")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--pos", type=int, default=31906)
    ap.add_argument("--ranks", default="0")
    ap.add_argument("--legacy", default="0,1", help="try this many use_legacy values (comma list of 0/1)")
    ap.add_argument("--save", default=None, help="write the inverse-RoPE'd output as attn{L}_r{r}_o_q.bin here")
    a = ap.parse_args()
    global DUMP
    DUMP = a.dump
    L, pos = a.layer, a.pos

    pool, own, idx, addr = pool_and_indices(L, pos)
    print(f"  pool {pool.shape} uint8 (page {addr['pool']['swa_page']}, {addr['pool']['swa_bytes_per_token']} B/token, "
          f"{addr['pool']['kv_layout']})")
    print(f"  decode token slot loc {own} = page {own // addr['pool']['swa_page']} slot {own % addr['pool']['swa_page']}")
    print(f"  indices (newest first) {idx[:4].tolist()} .. {idx[-2:].tolist()}   n={idx.size}")

    kb = k_row_bytes(L, int(a.ranks.split(",")[0]), pos)
    pg, sl = own // addr["pool"]["swa_page"], own % addr["pool"]["swa_page"]
    stale = pool[pg, sl, 0].tobytes()
    pool[pg, sl, 0] = np.frombuffer(kb, np.uint8)
    print(f"  wrote the current token's 528-byte K to [{pg}, {sl}] (was {'all zero' if not any(stale) else 'non-zero'})")

    kc = torch.from_numpy(pool).to("cuda")
    ind = torch.from_numpy(idx).view(1, 1, -1).to("cuda")
    topk = torch.tensor([idx.size], dtype=torch.int32, device="cuda")
    scale = float(D ** -0.5)

    import flash_mla
    from flash_mla.flash_mla_interface import FlashMLASchedMeta
    from sglang.srt.layers.attention.deepseek_v4_mega_backend import pack_query as _pack

    p = f"layers.{L}.attn."
    sink_all = A.read(p + "attn_sink").float()
    c, s = TL.freqs_at(pos)

    for r in [int(x) for x in a.ranks.split(",")]:
        q_bf = TL.bf16r(TL.parts(np.fromfile(os.path.join(DUMP, f"attn{L}_r{r}_wq_b_part.bin"), np.uint8),
                                8, 8 * D, 1)[0][0])
        q = TL.rope_fma(torch.from_numpy(q_bf.reshape(8, D)), c, s)
        qt8 = torch.from_numpy(np.asarray(q, np.float32)).to("cuda").bfloat16().view(1, 8, D)
        # "pads each rank to a 64-head tile": the real 8 heads in slots 0..7, the rest zeros
        qt = qt8.new_zeros(1, 1, 64, D)
        qt[0, 0, :8] = qt8[0]
        sink = sink_all[r * 8:r * 8 + 8].to("cuda")
        sink = torch.cat([sink, sink.new_zeros(64 - 8)])      # _local_attn_sink(64): padded with zeros
        ker = torch.frombuffer(bytearray(open(os.path.join(DUMP, f"attn{L}_r{r}_o_q.bin"), "rb").read()),
                               dtype=torch.bfloat16).float().reshape(-1, 8, D)[0].numpy()
        print(f"\n  rank {r}:  our o_q rms {np.sqrt((ker ** 2).mean()):.6f}")
        for ul in [int(x) for x in a.legacy.split(",")]:
            try:
                out, lse = flash_mla.flash_mla_with_kvcache(
                    q=qt, k_cache=kc, block_table=None, cache_seqlens=None, head_dim_v=D,
                    tile_scheduler_metadata=FlashMLASchedMeta(), softmax_scale=scale, causal=False,
                    is_fp8_kvcache=True, indices=ind, attn_sink=sink, topk_length=topk,
                    use_legacy=bool(ul))
            except Exception as e:
                print(f"    use_legacy={ul}: {type(e).__name__}: {str(e)[:200]}")
                continue
            o = out[0, 0, :8].float().cpu().numpy()
            if a.save:
                os.makedirs(a.save, exist_ok=True)
                oo = TL.rope_fma(o, c, s, inverse=True, width=D)
                u = np.ascontiguousarray(np.asarray(oo, np.float32)).view(np.uint32)
                open(os.path.join(a.save, f"attn{L}_r{r}_o_q.bin"), "wb").write(
                    (u >> 16).astype(np.uint16).tobytes())      # bf16 = the high half
                print(f"    saved {a.save}/attn{L}_r{r}_o_q.bin")
            for tag, oo in (("as returned", o),
                            ("+ inverse rope", TL.rope_fma(o, c, s, inverse=True, width=D)),
                            ("+ forward rope", TL.rope_fma(o, c, s, width=D))):
                b, n = bits(oo[:, :D - RD], ker[:, :D - RD])
                print(f"    use_legacy={ul} {tag:15s} noPE bits {b}/{n}  relRMS {rr(oo, ker):.3e}  "
                      f"cos {cos(oo, ker):+.9f}  rms {np.sqrt((oo ** 2).mean()):.6f}")


if __name__ == "__main__":
    main()
