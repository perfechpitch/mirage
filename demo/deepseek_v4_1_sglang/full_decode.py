"""The whole DeepSeek-V4.1-Flash decode: ONE process, 8 GPUs, one `Harness.launch` a sublayer.

Why not torchrun (e2e_decode.py): one process a rank with `launch_stream` puts nothing on the host to line the
ranks up, so a rank can be at launch n+1 while a peer is still at n. `sum_store` then writes exchange set
(n+1) % 2 while the peer polls set n % 2, and `hc_post_combine`'s CTAs never end (chain_spike.py's docstring;
the watchdog dumps: "SMs entered 148/148, ended 116"). Here every sublayer goes through `Harness.launch` ->
`launch_func`: each GPU's buffers are reset on its stream, the GPUs are lined up host-side, then the kernel --
the drift is impossible, and the host reset is the one that does not have to rely on a reader re-arming a buffer
(the serving build, STATIC_RESET_IN_KERNEL, resets 3 of a layer's 10 buffers and leaves 7 to their readers).

The layers are built the way the layer demos build them: `attn_demo.build` / `moe_demo.build` on 8 weight dicts,
heads = 64 / 8. Each gets its sublayer's mHC tensors (mhc_check.py: streams, streams_out, hc_fn, hc_small, h_pre,
x_out, gamma_next), so the kernel's post writes the next sublayer's input: hc_mult = 4 is the model's structure,
not an option. The chain is x_out -> the next kernel's x and streams_out -> the next kernel's streams.

python full_decode.py --checkpoint DIR [--layers 40] [--pos 4096] [--steps 2] [--checkpoint-out DIR]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from deepseek_v4_1_csa2 import demo as attn_demo            # noqa: E402
from deepseek_v4_1_csa2 import weights as attn_weights      # noqa: E402
from deepseek_v4_1_moe import demo as moe_demo              # noqa: E402
from deepseek_v4_1_moe import weights as moe_weights        # noqa: E402
from mirage.mpk.static_harness import Harness               # noqa: E402
from mirage.mpk import compiler                             # noqa: E402

A = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
A.add_argument("--checkpoint", required=True, help="the model's directory (config.json, the safetensors)")
A.add_argument("--layers", type=int, default=40, help="sublayer pairs to build (the model has 40)")
A.add_argument("--gpus", type=int, default=8, help="the TP size: the GPUs this process drives")
A.add_argument("--pos", type=int, default=4096, help="the position of the T tokens (the caches hold [0, pos])")
A.add_argument("--steps", type=int, default=2, help="untimed decode steps after the first")
A.add_argument("--flush-mb", type=int, default=0, help="L2 emptied before each launch (0: not)")
A.add_argument("--timeout", type=float, default=120.0, help="a launch longer than this is a hang (s)")
A.add_argument("--out", default="/tmp/full_decode", help="the build directories")
A.add_argument("--flag", action="append", default=[], help="an nvcc flag for every layer (repeatable)")
A.add_argument("--rms", action="store_true", help="each sublayer's x_out/streams_out rms and nonfinite count")
A.add_argument("--save-x", default=None, metavar="FILE", help="the last sublayer's x_out to this file (bf16 raw)")
A.add_argument("--dump-dir", default=None, metavar="DIR",
               help="every sublayer's x_out and streams_out (bf16 raw, [T, HID] and [T, 20480]) plus the logits, to "
                    "DIR/layer{L:02d}_{x_out|streams_out}.bin and DIR/logits.bin: the chain's per-sublayer output to "
                    "compare against a reference offline. Written after the first step (the warm one), so a run costs "
                    "one extra step's wait, and no launch.")
A.add_argument("--dump-all", default=None, metavar="NAMES",
               help="a comma list of sublayers (attn0,moe0,...): also dump every graph node's buffer under DIR as "
                    "NAMES_r{rank}_{node}.bin, with NAMES_nodes.txt giving each one's shape and dtype. The node buffers "
                    "the launch already writes, so nothing is rebuilt -- it is how to see inside a sublayer, whose only "
                    "other numbers are its residual outputs.")
A.add_argument("--golden", default=None, help="a golden jsonl (golden_bench32k.jsonl): the row to compare with")
A.add_argument("--row", type=int, default=0, help="with --golden: the row index")
A.add_argument("--inputs-dir", default=None, metavar="DIR",
               help="a prefill dump (dump_prefill.py + dump_hook.py): its caches feed the chain instead of random ones, and "
                    "the step is the decode after that prefill")
A.add_argument("--data", default="/opt/dlami/nvme/saboxu/data/long-code-arena/bench32k/*.jsonl",
               help="with --inputs-dir: the jsonl whose row's prompt gives the decode token (its last one)")
A.add_argument("--token", type=int, default=None, help="with --inputs-dir: the decode token (default: the row's last prompt one)")
A.add_argument("--save-logits", default=None, metavar="FILE", help="the decode step's logits (fp32 raw) to this file")
a = A.parse_args()

NG = a.gpus
T, HID, HC = attn_demo.T, 5120, 4
devs = [torch.device("cuda", g) for g in range(NG)]
read = attn_weights.checkpoint_reader(a.checkpoint)
cfg = attn_weights.model_config(a.checkpoint)
heads = 64 // NG
FLUSH = a.flush_mb << 20
FLAGS = list(a.flag)          # no STATIC_RESET_IN_KERNEL: the host reset path (launch_func)
L40 = a.layers

# The kernels' build cache (compile_static's MIRAGE_STATIC_BUILD_CACHE): without it every run nvcc's all 80 layers
# again, even when <out>/a{L}/build holds the very same layer.cu from the last run -- the directory is only where a
# build is *written*, the cache is what makes the next run skip it. Keyed by (md5(layer.cu), the flags, code_tag()),
# so it also stays correct when the templates change. A fixed path, not one under <out>: a run with another --out
# must still hit it (--out/... + --dump-dir changed between two runs of the same layers is what cost 40 minutes).
os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE", "/tmp/mirage_static_cache")

PRE = a.inputs_dir
META, ADDR = None, None
if PRE:   # the dump's last prefill row is the decode step's history: the caches hold [0, pos - 1]
    META = [json.load(open(os.path.join(PRE, f"meta_rank{r:02d}.json"))) for r in range(NG)]
    ADDR = [json.load(open(os.path.join(PRE, f"meta_addr_rank{r:02d}.json"))) for r in range(NG)]
    a.pos = int(ADDR[0]["seq_lens"][0])


def log(*msg):
    print("[full]", *msg, flush=True)


assert heads * NG == 64, "the TP size splits the 64 heads"
assert T == int(os.environ.get("STATIC_TOKENS", "8")), "STATIC_TOKENS is the kernel's T"


def mhc_into(wg, d, L, kind, gamma):
    """A sublayer's mHC tensors on GPU d (mhc_check.py: the same set, the same formulas). gamma: the next
    sublayer's norm weight, the post's input to the next kernel's x."""
    p = f"layers.{L}."
    fn, base, scale = (read(p + f"hc_{kind}_{n}").float() for n in ("fn", "base", "scale"))
    hi = fn.bfloat16()
    fn_w = torch.cat([hi, (fn - hi.float()).bfloat16(),
                      torch.zeros(128 - 2 * fn.shape[0], fn.shape[1], dtype=torch.bfloat16)])
    wg["hc_fn_w"] = fn_w.to(d).contiguous()
    wg.update(streams=torch.zeros(T, HC * HID, dtype=torch.bfloat16, device=d),
              streams_out=torch.zeros(T, HC * HID, dtype=torch.bfloat16, device=d),
              hc_fn=fn.to(d).contiguous(), hc_small=torch.cat([base, scale]).to(d).contiguous(),
              h_pre=torch.zeros(T, HC, device=d),
              # NOT h_post/h_comb here: "h_comb" in w is the dispatch key that tells the demos "the framework runs the
              # post" (dsv41_static_layer.py sets them for the serving path; see demo.py's `if "h_comb" in w`). Adding
              # them to the chain's tensors silently skipped this sublayer's own dsv41_mhc_nodes, so streams_out was
              # never written and every dump came out zero. Compare coefficients with mhc_check.py instead.
              x_out=torch.zeros(T, HID, dtype=torch.bfloat16, device=d),
              gamma_next=gamma.to(d).contiguous())


# ---- the prefill's real state (--inputs-dir): the pools' bytes, and the tables that index them ------------------------
def pool_rows(r, L, name, row_bytes):
    """The dumped pool buffer as [rows, row_bytes] on GPU r (None: this layer has no such buffer).

    The bytes are the kernel's own layout (v41_new's 528-byte window rows, v41_fp4_new's 288-byte compressed rows, the
    indexer's 128-row pages), so a row is a row and the tables only have to say which one."""
    e = META[r]["buffers"].get(f"L{L}.{name}")
    if e is None:
        return None
    b = torch.from_numpy(np.fromfile(os.path.join(PRE, e["file"]), dtype=np.uint8)).to(devs[r])
    return b.view(-1, row_bytes)


def pool_paged(r, L, name, row_bytes, skip, cap):
    """The pool from row `skip` on as token 0's slice of [T, cap, row_bytes] (a copy; 0 elsewhere).

    Two things make this a copy and not a view: the first token's slice must hold the whole history (a view's cap would be
    rows / T -- 18 pages, and a 32K prefill's compressed history is 125), and the request's first page is not page 0 (the
    pools reserve it: the compressed KV at row `skip`, the index K one page further)."""
    src = pool_rows(r, L, name, row_bytes)
    if src is None:
        return None
    m = src.shape[0] - skip
    assert m >= cap, f"L{L}.{name}: the pool has {m} rows past row {skip}, the history needs {cap}"
    full = torch.zeros(T * cap, row_bytes, dtype=torch.uint8, device=devs[r])
    full[:cap].copy_(src[skip:skip + cap])
    return full.view(T, cap, row_bytes)


def pool_first_page(r):
    """The request's first compressed page (the pools reserve page 0, the indexer's pages one further)."""
    return int(_lst(ADDR[r]["page_table"])[0]) if ADDR[r].get("page_table") else 1


def pool_cap(ratio):
    """The kernel's CAP: the compressed history of the decode step, rounded up to 256 (the indexer's scores are CAP / 8 byte
    blocks and its grid is whole 256-position chunks). Not the pool's size: a 32K prefill's history is 15952 blocks at ratio 2
    and 31905 at ratio 1, and a cache with fewer rows is read past its end."""
    n = (a.pos + 1) // max(ratio, 1) + 1        # (the decode token may complete the last block)
    return max(-(-n // 256) * 256, attn_demo.NC)


def bind_pool_state(wg, r, L, variant, ratio, cap):
    """This rank's caches of layer L become the prefill's: the window (the 128 rows the step reads, compacted into the
    [T, W, ROW] the kernel wants), the compressed KV and the index K (token 0's slice; only a full layer owns them -- a
    Reuse / Reindex layer gets its source's below), the compressor's state (a view).

    The window is not a view: sparse_attn indexes its [T, W, ROW] by window_rows = t * W + (p - W..p) % W, so the row for
    position p has to sit at flat row p % W of token 0's slice (window_tables' table, make_tables)."""
    src = pool_rows(r, L, "window_cache", attn_demo.ROW)
    if src is not None:
        W = attn_demo.W
        win = torch.zeros(T, W, attn_demo.ROW, dtype=torch.uint8, device=devs[r])
        p0, r0 = int(_lst(ADDR[r]["positions"])[0]), int(_lst(ADDR[r]["swa_out_cache_loc"])[0])
        pos = a.pos
        srcs = torch.arange(pos, pos - W, -1, dtype=torch.long, device=devs[r])   # positions pos .. pos - W + 1
        win[0].index_copy_(0, (srcs % W), src[(srcs - p0 + r0)])                 # (row(p) = r0 + p - p0)
        wg["window_cache"] = win
    first_page = pool_first_page(r)
    if variant.startswith("full"):
        skip = first_page * (256 // ratio)          # the request's compressed KV starts at its first page, in 128-row units
        cc = pool_paged(r, L, "compress_cache", attn_demo.CROW, skip, cap)
        if cc is not None:
            wg["compress_cache"] = cc
        ik = pool_paged(r, L, "index_k", attn_demo.IROW, skip, cap)   # (the same offset: an index page per 128 positions)
        if ik is not None:
            wg["index_k"] = ik
    e = META[r]["buffers"].get(f"L{L}.cmp_state")
    if e is not None and "cmp_state" in wg:
        st = torch.from_numpy(np.fromfile(os.path.join(PRE, e["file"]), dtype=np.float32)).to(devs[r])
        wg["cmp_state"] = st.view(-1, ratio, 2, attn_demo.D)      # the pool's rows: [slot * ratio + ring][2 D]


def _lst(v):
    """A dumped metadata field as a list (the last row's tensor came out as a plain int when it had one element).

    The dumper now records some tables per request, i.e. as a list of lists, where an older dump wrote the single
    request's flat; the row being replayed is request 0, so a one-level unwrap covers both."""
    if not isinstance(v, list):
        return [v]
    if v and isinstance(v[0], list):
        return v[0]
    return v


def make_tables(r):
    """The decode step's locations on GPU r, one set for every attention layer.

    The prefill's last chunk row says where the window is: position p0 (the chunk's first) sits at row out_loc, and one
    request's window rows are consecutive (its pool pages are), so row(p) = out_loc + (p - p0). The dumped list of 128
    rows is the same series backwards -- asserted, so a different allocator fails loudly instead of silently reading
    another request's rows. SGLang's padding rows follow StepTables: position 0, window row 0 then -1, page 0."""
    d, ad, W = devs[r], ADDR[r], attn_demo.W
    pos = a.pos
    z = lambda *s, dt=torch.int32: torch.zeros(*s, dtype=dt, device=d)
    t = {"positions": z(T), "window_rows": torch.full((T, W), -1, dtype=torch.int32, device=d),
         "window_loc": z(T), "req_slots": z(T, dt=torch.int64), "pages": {}}
    t["positions"][0] = pos
    t["req_slots"][0] = int(ad["req_pool_indices"][0])
    p0, r0 = int(_lst(ad["positions"])[0]), int(_lst(ad["swa_out_cache_loc"])[0])
    row = lambda p: r0 + (p - p0)
    spi = [int(v) for v in _lst(ad.get("swa_page_indices") or [])]
    n = min(len(spi), int(_lst(ad.get("swa_topk_lengths") or [len(spi)])[0]), W)
    for i in range(n):                    # spi[0] is the newest position (p0), and the rows descend with it
        assert spi[i] == row(p0 - i), f"rank {r}: the window's row for position {p0 - i} is {spi[i]}, not {row(p0 - i)}"
    assert n == W, f"rank {r}: the window has {n} rows, not {W}"
    attn_demo.window_tables(t)             # window_loc = t W + pos % W, window_rows = t W + (pos - i) % W (the compact cache)
    pool_rows_total = int(ad["pool"]["swa_rows"]) * int(ad["pool"]["swa_page"])
    assert t["window_loc"][0] < pool_rows_total, f"rank {r}: rows out of the pool"
    t["window_loc"][1:] = torch.arange(1, T, device=d, dtype=torch.int32) * W   # (window_tables': each padding row its own
    t["window_rows"][1:] = torch.arange(1, T, device=d, dtype=torch.int32)[:, None] * W + \
        ((t["positions"][1:, None] - torch.arange(W, device=d)) % W)           #  zero slice, never token 0's real rows)

    def pages_for(cap):
        if cap not in t["pages"]:
            p = torch.zeros(T, cap // 128, dtype=torch.int32, device=d)
            p[0] = torch.arange(cap // 128, dtype=torch.int32, device=d)   # (token 0's slice = the whole pool)
            t["pages"][cap] = p
        return t["pages"][cap]

    t["pages_for"] = pages_for
    return t


def embed_row(tok):
    """The checkpoint's embedding row [HID] bf16: the decode token is the layer chain's x (the residual stream starts here)."""
    from safetensors import safe_open
    index = json.load(open(os.path.join(a.checkpoint, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(a.checkpoint, index["embed.weight"]), "pt") as sf:
        return sf.get_slice("embed.weight")[tok].to(torch.bfloat16)


def decode_token():
    """The token the decode step feeds: the row's last prompt token (the prompt of 31906 -> position 31905 listens to 0..31904)."""
    if a.token is not None:
        return a.token
    from transformers import AutoTokenizer
    row = [json.loads(line) for f in sorted(glob.glob(a.data)) for line in open(f)][a.row]
    ids = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True).encode(row["prompt"], add_special_tokens=False)
    log(f"row {a.row}: prompt {len(ids)} tokens, the decode step is position {a.pos} -> token {ids[a.pos]}")
    return ids[a.pos]


# ---- build the chain: attn L0 -> moe L0 -> attn L1 -> ... (each its own kernel) -------------------------
K = []          # (name, kernel, per-GPU weight dicts, per-GPU node buffers)
WSD = {}        # layer -> its attention ws (a Reuse / Reindex layer reads the source layer's tensors)
x = None        # the previous sublayer's x_out, per GPU (None: the first layer's x is random)
Rs = None       # the previous sublayer's streams_out, per GPU
TAB = [make_tables(r) for r in range(NG)] if PRE else None
def normed_row(e, L):
    """The first sublayer's input, SGLang's combine with no pre-mix (deepseek_v4_mhc._combine: `if apply_pre is None:
    return norm(x[:, 0, :])`), whose kernel is hc_combine_norm: value = bf16(sum_c pre_c x_c) -- one-hot on stream 0,
    so just the embedding -- then bf16(value rsqrt(mean value^2 + eps) * gamma) with gamma = layers.L.attn_norm.weight.

    The chain has to make this row itself: the NORM inside hc_post_combine produces every *later* sublayer's x from the
    previous post, and there is no post before layer 0. Without the gamma the attention reads the raw embedding -- the
    scale is harmless (act_quant is per-32-group) but the gamma is elementwise: its 12% spread makes x differ from
    SGLang's by ~12% in direction, which is the size of the layer-0 attention error."""
    g = read(f"layers.{L}.attn_norm.weight").float()
    v = e.float().bfloat16().float()          # the unfused combine stores bf16 before the norm reads it
    return (v * torch.rsqrt(v.square().mean() + cfg["norm_eps"]) * g).bfloat16()


X = None
S0 = None
if PRE:         # the residual stream starts at the decode token's embedding (row 0); SGLang's padding rows stay 0
    tok = decode_token()
    X = [torch.zeros(T, HID, dtype=torch.bfloat16, device=d) for d in devs]
    e = embed_row(tok).float()
    x0 = normed_row(e, 0)
    for r in range(NG):
        X[r][0].copy_(x0)
    # hc_broadcast (deepseek_v4_1's hc_broadcast): the initial mHC residual is the embedding in EVERY hc_mult stream,
    # not zeros -- zeros would make the first layer's post mix 4 empty streams and the whole chain start wrong
    S0 = [torch.zeros(T, HC * HID, dtype=torch.bfloat16, device=d) for d in devs]
    for r in range(NG):
        S0[r][0].copy_(e.bfloat16().repeat(HC))


def kv_src(L):
    """The compressed KV (and index K) this layer reads: the last kv_source_layers entry at or below it (model.py L1172)."""
    return max(k for k in cfg["kv_source_layers"] if k <= L)


def idx_src(L):
    """The top-k this layer reads: the last index_source_layers entry at or below it."""
    return max(k for k in cfg["index_source_layers"] if k <= L)


t0 = time.perf_counter()
for L in range(L40):
    variant, ratio = attn_weights.variant_of(cfg, L), cfg["compress_ratios"][L]
    csrc = L == cfg["candidate_source_layer"]
    rows = (a.pos + 2 + 255) // 256 * 256
    cap = pool_cap(ratio) if PRE else max(((a.pos + 1) // max(ratio, 1) + 255) // 256 * 256, attn_demo.NC)

    # -- attention --
    shards = [attn_weights.layer_tensors(read, cfg, L, NG, r, rows, d) for r, d in enumerate(devs)]
    ws = [{**s, **attn_demo.state_tensors(variant, ratio, csrc, cap, d)} for s, d in zip(shards, devs)]
    for r, (wg, d) in enumerate(zip(ws, devs)):
        if PRE:
            bind_pool_state(wg, r, L, variant, ratio, cap)          # the prefill's caches (leaves the graph's own buffers)
            t = TAB[r]
            wg.update(positions=t["positions"], window_rows=t["window_rows"], window_loc=t["window_loc"],
                      req_slots=t["req_slots"])
        else:
            attn_demo.random_inputs(wg, variant, ratio, a.pos)     # the caches, random and valid at pos
            cc = wg.get("compress_cache")                          # random_inputs masks window_cache's e4m3 bytes
            if cc is not None:                                     # but not these: 0x7F/0xFF are NaN in e4m3
                cc[..., :attn_demo.D // 2].bitwise_and_(0xFE)
            ik = wg.get("index_k")
            if ik is not None:
                ik[..., :64].bitwise_and_(0xFE)
        mhc_into(wg, d, L, "attn", read(f"layers.{L}.ffn_norm.weight"))
        if X is not None:
            wg["x"] = X[r]
        else:
            wg["x"] = x[r] if x is not None else \
                (torch.randn(T, HID, generator=torch.Generator().manual_seed(1)) * 0.02).bfloat16().to(d).contiguous()
        if Rs is not None:
            wg["streams"] = Rs[r]
        elif S0 is not None:
            wg["streams"] = S0[r]       # (layer 0: the embedding in every stream)
    if variant in ("reuse", "reindex"):   # the group shares its source layer's launch: the same tensors, not a copy
        src = WSD[kv_src(L)]
        for r, wg in enumerate(ws):
            wg["compress_cache"] = src[r]["compress_cache"]
            if variant == "reuse":
                wg["compress_idx"] = WSD[idx_src(L)][r]["compress_idx"]
            else:
                wg["index_k"] = src[r]["index_k"]
                wg["candidates"] = WSD[cfg["candidate_source_layer"]][r]["candidates"]
    if PRE:   # the page table follows the cache it indexes (a Reuse layer's is its source's)
        for r, wg in enumerate(ws):
            cc = wg.get("compress_cache")
            if cc is not None:
                wg["compress_pages"] = TAB[r]["pages_for"](cc.shape[1])
    sk, bufs, _ = attn_demo.build(ws, variant, ratio, csrc, heads, os.path.join(a.out, f"a{L}"),
                                  extra_flags=FLAGS)
    K.append((f"attn{L}", sk, ws, bufs))
    WSD[L] = ws
    x = [wg["x_out"] for wg in ws]
    Rs = [wg["streams_out"] for wg in ws]
    log(f"a{L} {variant}: built, {(time.perf_counter() - t0) / 60:.1f} min, "
        f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")

    # -- MoE (the last one also runs the LM head) --
    wsm = [moe_weights.layer_tensors(read, L, NG, r, d) for r, d in enumerate(devs)]
    for wg, rt in zip(wsm, moe_weights.routed_tensors(read, L, NG, list(range(NG)), devs)):
        wg.update(rt)
    last = L == L40 - 1
    gamma = read("norm.weight") if last else read(f"layers.{L + 1}.attn_norm.weight")
    import head_demo                                            # noqa: E402  (the head's tensors)
    for r, (wg, d) in enumerate(zip(wsm, devs)):
        wg["x"] = x[r]
        wg["y"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=d)   # (the head path ignores it)
        mhc_into(wg, d, L, "ffn", gamma)
        if Rs is not None:
            wg["streams"] = Rs[r]
        if last:   # (11-20, the demo adds head_nodes when "head_w" is in w)
            hw = head_demo.head_tensors(read, NG, r, d)
            wg.update({k: hw[k] for k in ("head_w", "logits", "head_ids") if k in hw})
    sk, bufs, _ = moe_demo.build(wsm, os.path.join(a.out, f"m{L}"), extra_flags=FLAGS)
    K.append((f"moe{L}", sk, wsm, bufs))
    x = [wg["x_out"] for wg in wsm]
    Rs = [wg["streams_out"] for wg in wsm]
    log(f"m{L}: built, {(time.perf_counter() - t0) / 60:.1f} min, "
        f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")

hs = [(name, Harness(sk), ws, bufs) for name, sk, ws, bufs in K]
log(f"chained {len(hs)} kernels on {NG} GPUs in {(time.perf_counter() - t0) / 60:.1f} min")


def rms_line(ws, tag):
    """The last GPU's view: the same bits on every GPU is a separate check (chain_spike's)."""
    out = []
    for nm in ("x_out", "streams_out"):
        t = ws[0].get(nm)
        if torch.is_tensor(t) and t.numel() and t.dtype == torch.bfloat16:
            f = t.float()
            out.append(f"{nm} rms {f.square().mean().sqrt().item():.4g} "
                       f"nf {int((~torch.isfinite(f)).sum())}")
    return f"[full]   {tag}: " + ", ".join(out)


def step(time_it: bool = False):
    spans = []
    for name, h, ws, _ in hs:
        h.launch(FLUSH, timeout_s=a.timeout)
        if time_it:
            spans.append((name, h.span_us()))
        if a.rms and not time_it:
            print(rms_line(ws, name), flush=True)
    return spans


step()                                   # warm: the caches and the code


if a.dump_dir:                           # the chain's own numbers, every sublayer, one file each
    os.makedirs(a.dump_dir, exist_ok=True)
    want = set(a.dump_all.split(",")) if a.dump_all else set()
    for name, _h, wsx, bufs in hs:
        # the sublayer's own input rows (not a node buffer: the caller's tensor). For a sublayer whose input is not the
        # previous sublayer's output -- layer 0's -- this is the row to compare with SGLang's normed x.
        xt = wsx[0].get("x")
        if torch.is_tensor(xt) and xt.dtype == torch.bfloat16:
            xt.detach().cpu().contiguous().view(torch.int16).numpy().tofile(
                os.path.join(a.dump_dir, f"{name}_x.bin"))
        # the mHC triplet this sublayer's post applied, straight from the kernel's hc_coef buffer: pre|post|comb|0s as
        # [T, 32] fp32. This is the one record that separates "the sublayer's output is wrong" from "the mix is wrong"
        # -- and it needs no extra node (hc_coef is in the graph already), so it never costs a rebuild.
        cf = bufs[0].get("hc_coef") if bufs else None
        if torch.is_tensor(cf) and cf.numel() >= T * 32:
            cf.detach().cpu().contiguous().view(torch.float32).numpy()[: T * 32].astype(np.float32).tofile(
                os.path.join(a.dump_dir, f"{name}_coef.bin"))
        if name in want:                 # every graph node's buffer of the named sublayers, plus the name list: the
            with open(os.path.join(a.dump_dir, f"{name}_nodes.txt"), "w") as f:   # only way to see inside a sublayer
                for r, b in enumerate(bufs):
                    for k, t in b.items():
                        f.write(f"{r}\t{k}\t{tuple(t.shape)}\t{t.dtype}\n")
                        t.detach().cpu().contiguous().view(torch.int16).numpy().tofile(
                            os.path.join(a.dump_dir, f"{name}_r{r}_{k}.bin"))
    for name, h, ws, _ in hs:            # (ws is the per-GPU list; rank 0's has the gather_f32 logits)
        for nm in ("x_out", "streams_out"):
            t = ws[0].get(nm)
            if torch.is_tensor(t) and t.numel():
                t.detach().cpu().contiguous().view(torch.int16).numpy().tofile(
                    os.path.join(a.dump_dir, f"{name}_{nm}.bin"))
    last_ws = hs[-1][2]
    lg = last_ws[0].get("logits")
    if torch.is_tensor(lg) and lg.numel():
        lg.detach().float().cpu().contiguous().numpy().astype(np.float32).tofile(
            os.path.join(a.dump_dir, "logits.bin"))
    # the mHC coefficients the post step used: R'[j] = sum_i comb[i][j] R[i] + post[j] out. Dumping them is what tells a
    # wrong coefficient set (a per-stream scale/mix, which leaves the other streams right) from a wrong sublayer output
    # (which moves every stream the same way) -- the two look alike in the residuals alone. The chain's path does not
    # materialize them (see mhc_into: h_post/h_comb in w is the "framework runs the post" dispatch key), so this only
    # fires for a --out tree whose tensors carry them; mhc_check.py is how to compare coefficients.
    for name, _h, wsx, _bufs in hs:
        for nm in ("h_post", "h_comb"):
            t = wsx[0].get(nm)
            if torch.is_tensor(t) and t.numel():
                t.detach().float().cpu().contiguous().view(-1).numpy().astype(np.float32).tofile(
                    os.path.join(a.dump_dir, f"{name}_{nm}.bin"))
    log(f"dumped {len(hs)} sublayers' x_out/streams_out, coefficients and the logits to {a.dump_dir}")


def topk_line():
    """The decode step's top-32 next-token logprobs against the golden's own.

    Token equality is not the criterion here (golden_forced.py: this row's step 0 has a 0.16 top-1 probability and exact
    ties among the top-k, so bf16 rounding flips an argmax). The distribution is what has to match: the golden row's topk
    is the same log-softmax, so a correct chain lands on the same ids with logprobs within a few 1e-3."""
    ws = hs[-1][2]
    lg = ws[0]["logits"][0].float().cpu()      # ([T, vocab] fp32 on EVERY GPU: gather_f32's output, not a per-rank slice)
    if a.save_logits:
        np.array(lg.numpy(), dtype=np.float32).tofile(a.save_logits)
        log(f"logits [vocab {lg.numel()}] written to {a.save_logits}")
    lp = torch.log_softmax(lg, dim=-1)
    vals, idx = lp.topk(32)
    got = [(float(v), int(i)) for v, i in zip(vals, idx)]
    log("this kernel: " + "  ".join(f"{i}:{v:.4f}" for v, i in got[:8]))
    if not a.golden:
        return
    g = [json.loads(l) for l in open(a.golden)][a.row]
    want = [(float(t[0]), int(t[1])) for t in g["topk"][0]]
    log("golden     : " + "  ".join(f"{i}:{v:.4f}" for v, i in want[:8]))
    gi = {i: v for v, i in want}
    oi = {i: v for v, i in got}
    common = set(gi) & set(oi)
    err = [abs(gi[i] - oi[i]) for i in common]
    log(f"top-32 overlap {len(common)}/32, logprob |diff| max {max(err):.4f} mean {sum(err) / len(err):.4f}, "
        f"max|logit diff| {max(err) * 1.0:.3f} (a correct chain: overlap 32, mean a few 1e-3)")
    if int(g["ids"][0]) in oi:
        rank = sorted(oi, key=lambda i: -oi[i]).index(int(g["ids"][0]))
        log(f"golden's token {g['ids'][0]}: this kernel's logp {oi[int(g['ids'][0])]:.4f} "
            f"vs golden {gi.get(int(g['ids'][0]), float('nan')):.4f} (rank {rank})")
        if int(g["ids"][0]) in gi:
            log(f"the same token's |diff| {abs(oi[int(g['ids'][0])] - gi[int(g['ids'][0])]):.4f}")
    else:
        log(f"golden's token {g['ids'][0]} is not in this kernel's top-32 (logp {float(lp[int(g['ids'][0])]):.4f}, "
            f"golden {gi.get(int(g['ids'][0]), float('nan')):.4f})")


topk_line()
if a.golden:                             # the row's greedy token from the head's own argmax
    g = [json.loads(l) for l in open(a.golden)][a.row]
    ids = g["ids"]
    got = int(hs[-1][2][0]["head_ids"][0].item()) if "head_ids" in hs[-1][2][0] else -1
    log(f"golden row {a.row} ({g['subset']}), prompt {g['prompt_tokens']} tokens: "
        f"this kernel's next token {got}, golden's {ids[0]}, {'MATCH' if got == ids[0] else 'MISMATCH'}")
    log("NOTE: no prefill ran here (the caches are random and valid at pos), so a mismatch is expected; the "
        "number above is a wiring check, not a correctness check of the model.")

for s in range(a.steps):
    step()
    if a.rms:
        log(f"step {s + 1}/{a.steps} done")

spans = step(time_it=True)
per = {n: v for n, v in spans}
log("medians: " + " ".join(f"{n} {v:.1f}" for n, v in spans[:6]) + (" ..." if len(spans) > 6 else ""))
log(f"a step of {len(spans)} launches: sum of spans {sum(v for _, v in spans):.1f} us")

if a.save_x:
    v = hs[-1][2][0]["x_out"].detach().cpu().contiguous().view(torch.int16)
    v.numpy().tofile(a.save_x)
    log(f"the last sublayer's x_out [T={T}, {HID}] written to {a.save_x}")

for _, sk, _, _ in K:
    sk.finalize()
log("done")
