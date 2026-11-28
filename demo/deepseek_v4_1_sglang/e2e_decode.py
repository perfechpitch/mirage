"""One whole-model decode step of DeepSeek-V4.1-Flash, 40 layers, TP=8, measured without SGLang.

This chains the same static layers the serving hook uses (dsv41_static_layer.StaticAttn / StaticMoE), one rank per
GPU (as the hook runs them), so the all-reduce is in the kernels (sum_store over each rank's exchange region) and
nothing of SGLang is imported. A step is 40 attention launches + 40 MoE launches, the last MoE also computing the
LM head: the time between the first launch and the last one is a real end-to-end TPOT (per-launch gaps included),
not a sum of separately measured layers.

What is real and what is not
  real      the 40 layer kernels (plans, weights, all-reduce, mHC, the head), the launch sequence, the
            dependencies between the layers, the KV caches at the prompt's position (~32K rows read)
  stand-in  x: the prompt's last token through the checkpoint's embedding table (the routers see a real row);
            the KV caches: filled the way the layer demos fill them (random valid encodings, no NaN), since no
            prefill runs here; Engram (L1, L14) is off (its kv is written by the attention kernel, its lookup
            needs SGLang's tables): worth about 2 x 9 us of the step, see _dsv41_bs1_findings.md

The prompts are bench32k's (data/long-code-arena/bench32k/*.jsonl), so the position is a real 32K ISL.

  torchrun --nproc_per_node=8 e2e_decode.py --checkpoint DIR --row 0 --reps 20
  DSV41_E2E_LAYERS=2 torchrun --nproc_per_node=8 e2e_decode.py --checkpoint DIR   # a 2-layer smoke test
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DSV41_STATIC_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
os.environ.setdefault("DSV41_STATIC_ROOT", ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.append(os.path.join(ROOT, "demo"))

A = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
A.add_argument("--checkpoint", required=True)
A.add_argument("--inputs-dir", default=None,
               help="a prefill dump (/tmp/dec_io2): the caches become SGLang's real bytes instead of random fills, and "
                    "Engram's L1/L14 run on the dump's own n-gram rows. Every rank reads its own meta_rank{R}.json")
A.add_argument("--golden", default=None, help="the golden jsonl: its ids[0] is the token the prefill sampled")
A.add_argument("--token", type=int, default=None, help="the decode token to feed (default: golden ids[0], else the prompt's last)")
A.add_argument("--data", default="/opt/dlami/nvme/saboxu/data/long-code-arena/bench32k/*.jsonl")
A.add_argument("--row", type=int, default=0, help="which bench32k row")
A.add_argument("--pos", type=int, default=None, help="the position (default: the row's own prompt_tokens)")
A.add_argument("--layers", type=int, default=int(os.environ.get("DSV41_E2E_LAYERS", "40")), help="the chain's layers")
A.add_argument("--reps", type=int, default=20, help="timed steps (before each, 512 MB is read to empty L2)")
A.add_argument("--warmup", type=int, default=3)
A.add_argument("--out", default="/tmp/e2e_decode")
A.add_argument("--pdl", type=int, default=0)
A.add_argument("--pool-mb", type=float, default=8.0, help="the exchange pool's region size (the largest layer's)")
A.add_argument("--sync-every", type=int, default=0,

               help="layers between two cross-rank barriers (0: only at the end). Each static layer's sum_store is an "
                    "all-reduce over the exchange regions, so all ranks must be in the same kernel: without a barrier "
                    "the ranks drift apart and the polls deadlock")
A.add_argument("--spans", action="store_true",
               help="after the timed loop, run one more step and report every launch's own span (the kernel's "
                    "start-barrier-to-last-SM-end device time, from its stamps): sum(spans) vs the step time separates "
                    "GPU occupancy from the host's launch gaps")
A.add_argument("--dump", default=None,
               help="after the first step, write every layer's x_out and streams_out (bf16) and the logits to this "
                    "directory: two runs' dumps must be bit-identical (sync vs async mode proves the fast path, two "
                    "async runs prove repeatability)")
a = A.parse_args()
os.environ.setdefault("STATIC_TOKENS", "1")     # (before dsv41_static_layer is imported: its T is a module constant)
assert a.layers <= 40

import torch.distributed as dist                                          # noqa: E402

RANK = int(os.environ.get("RANK", "0"))
LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
assert WORLD == 8, f"the merge is TP=8: WORLD_SIZE={WORLD}"
torch.cuda.set_device(LOCAL)
dist.init_process_group("nccl", device_id=torch.device("cuda", LOCAL))
GROUP = dist.group.WORLD


def log(*msg):
    if RANK == 0:
        print("[e2e] " + " ".join(str(m) for m in msg), flush=True)


rows = []
for pat in a.data.split(","):
    for f in sorted(glob.glob(pat)):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
if not rows:
    sys.exit(f"no rows in {a.data}")
row = rows[a.row % len(rows)]
pos = a.pos if a.pos is not None else int(row["prompt_tokens"])
log(f"row {a.row % len(rows)}: {row['id'][:90]}")
log(f"prompt_tokens {row['prompt_tokens']} -> decode at position {pos}")

from transformers import AutoTokenizer                                    # noqa: E402
from safetensors import safe_open                                        # noqa: E402
from deepseek_v4_1_sglang import dsv41_static_layer as Lm                # noqa: E402

T, HID, HC = Lm.T, Lm.HID, Lm.HC
MHC = os.environ.get("DSV41_E2E_MHC", "1") != "0"   # 0 is a diagnosis only: the model has hc_mult=4, so no mHC is not the model
# TPB: the layer kernels' own start barrier (-DSTATIC_TP_BARRIER, dsv41_static_layer). With it the ranks line up
# inside the kernel, so the host barrier between launches is not needed -- and must not be paid for: this script is
# the pure chain's timing harness, and a dist.barrier() per sublayer inflates the TPOT it reports.
# With TPB=0 nothing lines the ranks up, which is the 11-25 configuration that returned NaN from the first Reuse
# layer: DSV41_E2E_NOSYNC=1 reproduces it on purpose.
TPB = os.environ.get("DSV41_STATIC_TPB", "1") != "0"
NOSYNC = os.environ.get("DSV41_E2E_NOSYNC", "0") != "0"
# The in-kernel TP barrier is meant to make the host barrier unnecessary. This forces BOTH, on a build that has the
# in-kernel one: clean -> the exchange protocol and the resets are sound and the in-kernel barrier alone is what fails;
# still NaN -> the defect is not the barrier's reach (see FINDINGS 12.42.4).
FORCE_BAR = os.environ.get("DSV41_E2E_FORCE_BARRIER", "0") != "0"
# FULL_SYNC: cudaSynchronize every sublayer too, not just the host barrier. A reuse layer reads its source layer's
# compress_idx, which the PREVIOUS launch writes; only the device-side wait proves that write landed before this
# launch's resets run. Costlier per sublayer (not the timed path), and the correctness check that distinguishes
# "the reset protocol is wrong" from "the launches were allowed to overlap across a producer/consumer pair".
FULL_SYNC = os.environ.get("DSV41_E2E_FULL_SYNC", "0") != "0"
# MASK_FF: diagnosis only -- clear the dump's 0xFF bytes in the e4m3 caches (0xFF is NaN). If the NaN goes away,
# the chain reads those bytes; if not, the NaN is computed (a tiling/reset defect), not carried in with the KV.
MASK_FF = os.environ.get("DSV41_E2E_MASK_FF", "0") != "0"
# CLR_IDX: diagnosis only -- set every layer's compress_idx back to its INITIAL value (-1) before the step, i.e.
# reproduce "the Reuse layer read the producer's list before the producer published it". sparse_attn skips every
# block on -1, so m = -1e30 and sum = 0: P / sum is __fdividef(0, 0), which is arithmetic NaN (0x7fff in bf16).
ZERO_KV = os.environ.get("DSV41_E2E_ZERO_KV", "0") != "0"
CLR_IDX = os.environ.get("DSV41_E2E_CLR_IDX", "0") != "0"
# TRACE: comma-separated attention layers whose buffers are censused immediately before and after their launch
# (GPU idle). A buffer already polluted BEFORE the launch was read wrong; one that becomes polluted after it was
# written wrong. Diagnosis only (it synchronizes, so it is not the timed path).
TRACE = {int(v) for v in os.environ.get("DSV41_E2E_TRACE", "").split(",") if v.strip()}
# PTRS: after building, print every node buffer's device address per layer and flag two names sharing one. A buffer
# that never has 0xFF written to it (reset -1, by_reader 0: o_q) coming back all 0xFF means another buffer's writes
# landed on the same bytes -- an allocation alias, not arithmetic.
PTRS = os.environ.get("DSV41_E2E_PTRS", "0") != "0"
# DSV41_E2E_HOSTT: time the host side of each launch. The GPU idles between kernels by more than the kernels take
# (spans 3067 us of a 9187 us step), so the question is whether the host is the bottleneck at issue time: with
# DSV41_E2E_ASYNC=0 the host also synchronizes after each launch, which alone would make it one.
HOSTT = os.environ.get("DSV41_E2E_HOSTT", "0") != "0"
# DSV41_E2E_PROFILE: cProfile over the timed loop, the top 25 by cumulative time. The host side of an attention launch
# costs 135 us against the MoE's 20 us while its kernel runs 40 us: which callee holds it is a fact to read, not infer.
PROFILE = os.environ.get("DSV41_E2E_PROFILE", "0") != "0"
_HW = {}
# DSV41_E2E_DUMP_LOGITS: write the step's logits next to --out. The serving build (STATIC_RESET_IN_KERNEL) and the
# host-reset build are two spellings of the same arithmetic, so their logits must agree bit for bit: that is the
# correctness check a fix to the reset protocol has to pass.
DUMP_LOGITS = os.environ.get("DSV41_E2E_DUMP_LOGITS", "")
# DSV41_E2E_CHAINDUMP: write every sublayer's x_out and streams_out in full_decode's chain format (bf16 raw, [T, 5120]
# and [T, 20480]), so cmp_band.py / cmp_all.py can judge this chain against the same oracles /tmp/chain_ok was judged
# by. Row 0 is the step's token in both, so a T = 1 chain and a T = 8 one are comparable.
CHAINDUMP = os.environ.get("DSV41_E2E_CHAINDUMP", "")
# DSV41_E2E_GRAPH: submit the step as one CUDA graph per launch segment instead of one launch_stream per sublayer.
# The megakernel build is the one this is meant for (-DSTATIC_RESET_IN_KERNEL: the buffers' resets run inside the
# kernel off a launch counter, so a replay re-arms itself), and correctness is checked the only way that settles it:
# the same run with and without the graphs, every x_out / streams_out / logits compared as int bits.
# DSV41_E2E_GRAPH: submit the step as one CUDA graph per launch segment instead of one launch_stream per sublayer.
# EXPERIMENTAL, and NOT bit-idempotent -- do not use it for a number anyone will quote. What it does right: a single
# replay is bit-identical to the launch chain (160 x_out/streams_out + the logits, compared as int bits), and it is
# 4.5 ms against 5.01 ms. What it does wrong: the SECOND and later replays are not. Measured (11-28, 40 layers,
# --reps 10 --warmup 3): the logits walk 1.12x -> 2.76x of the reference band while the launch chain is bit-stable
# over 13 steps, and a per-buffer fingerprint (DSV41_E2E_SIG) puts the first divergence at attn3.x_out with 155/160
# sublayer outputs then differing. At 4 layers only the LAST sublayer differs, and steps 2..7 are identical to each
# other, so it is a different fixed point rather than an accumulating error -- and the step it starts on moves with
# the warm-up count, which makes it a race, not a parity bug. The suspect is the reset protocol's per-launch state
# (core.cuh kernel_begin: gen at g.gen_ptr, dbuf_par -> rg_set()'s exchange set) reaching a different fixed point
# once the host stops pacing the ranks between launches. Left in the tree as the harness for that work.
GRAPH = os.environ.get("DSV41_E2E_GRAPH", "0") != "0"
# DSV41_E2E_SIG: print a bitwise fingerprint of the logits after every step (rank 0). Every step repeats the same
# token at the same position, so any change between steps is a replay that is not idempotent -- the reset protocol
# question (core.cuh kernel_begin). Diagnosis only.
SIG = os.environ.get("DSV41_E2E_SIG", "0") != "0"
# TAG: comma-separated layers whose o_q is filled with 0xAB just before the launch. After it, 0xAB everywhere means
# nothing wrote the tensor (the node did not run); 0xFF everywhere means the kernel's reset filled it (the node ran
# but produced NaN through the sentinel); anything else means the node wrote real values. Distinguishes "not run"
# from "ran and produced NaN" -- the two readings of a 0xFFFF bf16 census.
TAG = set()
for _v in os.environ.get("DSV41_E2E_TAG", "").split(","):
    if _v.strip():
        _l, _, _n = _v.partition(":")
        TAG.add((int(_l), _n or "o_q"))
assert T == int(os.environ["STATIC_TOKENS"])
cfg = Lm.attn_weights.model_config(a.checkpoint)
variants = [Lm.attn_weights.variant_of(cfg, L) for L in range(40)]
log(f"{a.layers} layers: " + " ".join(f"{L}:{variants[L]}" for L in range(a.layers)))

# ---- the step's input (the token itself is decided below: with a prefill dump it is the token the PREFILL
# sampled, which is not the prompt's last one -- see full_decode.decode_token) ---------------------------------
tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
ids = tok.encode(row["prompt"], add_special_tokens=False)
idx = json.load(open(os.path.join(a.checkpoint, "model.safetensors.index.json")))["weight_map"]
log(f"tokenized {len(ids)} ids (the row's own count {row['prompt_tokens']})")

dev = torch.device("cuda", LOCAL)
sources = Lm.Sources(cfg)
positions = torch.full((T,), pos, dtype=torch.int32, device=dev)

# ---- the prefill's real state (--inputs-dir) ----------------------------------------------------------------
# full_decode's PRE loader, one rank per process: this rank's pools are read from the dump and copied into the
# kernel's own tensors. The row layouts are already the kernel's (528-byte window rows, 288-byte compressed rows,
# 68-byte index rows), so a row is a row and the tables only say which one.
#
# This is also what fixes the chain's NaN: random fills put 0x7F / 0xFF (NaN in e4m3) into the compressed KV, and
# the chain pushes them through every mHC residual until x is all NaN by L3. SGLang's bytes cannot contain them.
PRE = a.inputs_dir
META = ADDR = None
if PRE:
    import numpy as np

    def _lst(v):
        """A dumped metadata field as a list (one-element tensors came out as plain ints; newer dumps nest per
        request, and the row being replayed is request 0)."""
        if not isinstance(v, list):
            return [v]
        if v and isinstance(v[0], list):
            return v[0]
        return v

    META = json.load(open(os.path.join(PRE, f"meta_rank{RANK:02d}.json")))
    ADDR = json.load(open(os.path.join(PRE, f"meta_addr_rank{RANK:02d}.json")))
    lay = ADDR["pool"].get("kv_layout")
    assert lay == "v41_new", f"the dump's kv_layout is {lay!r}: the kernel reads v41_new's rows"
    # THE POSITION MUST BE THE DUMP'S. The caches hold [0, seq_lens-1]; the row's own prompt_tokens can be a
    # different request (bench32k row 0 is 32344, this dump is 31906), and then the window's row for position p is
    # a row the prefill never wrote -- 0xFF, which is NaN in e4m3. That is exactly the attn-L3 NaN: not the reset
    # protocol, not the barrier, a position off by a different request's token count.
    dump_pos = int(_lst(ADDR["seq_lens"])[0])
    if a.pos is not None and a.pos != dump_pos:
        raise SystemExit(f"--pos {a.pos} is not the dump's position {dump_pos}: the caches hold [0, {dump_pos - 1}], "
                         f"so every window row past it was never written (0xFF in e4m3 is NaN)")
    if pos != dump_pos:
        log(f"position {pos} (row prompt_tokens) -> {dump_pos} (the dump's seq_lens: the only position these "
            f"caches describe)")
    pos = dump_pos
    positions = torch.full((T,), pos, dtype=torch.int32, device=dev)   # (rebuild: pos moved)
    log(f"inputs {PRE}: rank {RANK}'s pools, layout {lay}, "
        f"{len(META['buffers'])} buffers")

    def _pool_rows(L, name, row_bytes):
        """This rank's dumped pool buffer as [rows, row_bytes] on the GPU (None: this layer has no such buffer)."""
        e = META["buffers"].get(f"L{L}.{name}")
        if e is None:
            return None
        b = torch.from_numpy(np.fromfile(os.path.join(PRE, e["file"]), dtype=np.uint8)).to(dev)
        if b.numel() % row_bytes:
            raise SystemExit(f"L{L}.{name}: the dump is {b.numel()} bytes, not a multiple of this kernel's "
                             f"{row_bytes}-byte row")
        return b.view(-1, row_bytes)

    def _pool_paged(L, name, row_bytes, skip, cap):
        """The pool from row `skip` on as token 0's slice of [T, cap, row_bytes] (the request's first page is not
        pool page 0: the pools reserve it)."""
        src = _pool_rows(L, name, row_bytes)
        if src is None:
            return None
        m = src.shape[0] - skip
        assert m >= cap, f"L{L}.{name}: {m} rows past row {skip}, the history needs {cap}"
        out = torch.zeros(T * cap, row_bytes, dtype=torch.uint8, device=dev)
        out[:cap].copy_(src[skip:skip + cap])
        return out.view(T, cap, row_bytes)

    def _first_page():
        return int(_lst(ADDR["page_table"])[0]) if ADDR.get("page_table") else 1

    def _fill_real_kv(sa, L):
        """Layer L's caches become the prefill's (full_decode's bind_pool_state, for this rank).

        The window is not a range copy: sparse_attn reads [T, W, ROW] at row t*W + (p - W..p) % W, so position p's
        row has to sit at flat row p % W of token 0's slice."""
        w, variant = sa.w, sa.variant
        src = _pool_rows(L, "window_cache", Lm.attn_demo.ROW)
        if src is not None:
            W = Lm.attn_demo.W
            p0, r0 = int(_lst(ADDR["positions"])[0]), int(_lst(ADDR["swa_out_cache_loc"])[0])
            ps = torch.arange(pos, pos - W, -1, dtype=torch.long, device=dev)   # positions pos .. pos - W + 1
            win = w["window_cache"]
            win.zero_()
            win[0].index_copy_(0, ps % W, src[ps - p0 + r0])                    # (row(p) = r0 + p - p0)
        if variant.startswith("full"):      # a Reuse / Reindex layer shares its source's tensors (Sources)
            skip = _first_page() * (256 // sa.ratio)
            cap = w["compress_cache"].shape[1]
            cc = _pool_paged(L, "compress_cache", Lm.attn_demo.CROW, skip, cap)
            if cc is not None:
                w["compress_cache"].copy_(cc)
            ik = _pool_paged(L, "index_k", Lm.attn_demo.IROW, skip, cap)
            if ik is not None and "index_k" in w:
                w["index_k"].copy_(ik)
        e = META["buffers"].get(f"L{L}.cmp_state")
        if e is not None and "cmp_state" in w:
            # The pool's slots are the REQUEST slots; the kernel's are its T rows, read at index req_slots[t]
            # (state_tensors' arange, full_decode's StepTables: row 0 the live one, the padding rows pointing at
            # slot 0). Put the request's slot where the kernel will look for it: its own row when the build has
            # room (T = 8, the pool's own layout), else row 0 (T = 1) with req_slots[0] moved with it.
            st = torch.from_numpy(np.fromfile(os.path.join(PRE, e["file"]), dtype=np.float32)).to(dev)
            slots = st.view(-1, sa.ratio, 2, Lm.attn_demo.D)
            first_slot = int(_lst(ADDR["req_pool_indices"])[0])
            row = first_slot if first_slot < w["cmp_state"].shape[0] else 0
            w["cmp_state"][row].copy_(slots[first_slot])
            if "req_slots" in w:
                w["req_slots"].zero_()
                w["req_slots"][0] = row

# ---- Engram (layers 1 and 14) ---------------------------------------------------------------------------------
# deepseek_v4.py:4360: the Engram REPLACES layer L's residual before its attention -- R'' = engram_gate(R'), with
# R'' mixing the n-gram embedding rows into the token's hc streams. The gate runs in the MoE kernel of L-1 (after
# its post), so that kernel cannot write the next attention's x: its post computed x from R', one step stale. The
# host re-derives it from R'' -- full_decode's eg_next_x, the same three reads (hc_coef, ffn_norm, streams_eg).
EG_LAYERS = sorted(int(v) for v in cfg.get("engram_layer_ids", []))        # [1, 14]
EG_HOST = {g - 1: g for g in EG_LAYERS}                                    # {0: 1, 13: 14}: which MoE hosts it
EG_ON = bool(PRE) and bool(EG_HOST)
_read = Lm.reader(a.checkpoint)


def _hc_pre(sm):
    """The MoE's ffn_pre mix (hc_coef's first HC floats, fp32): model.py carries ONE mix across sublayers, so the
    attention of L reads the pre of MoE L-1, not its own triplet (using L's own was 13.6% off)."""
    t = sm.bufs[0].get("hc_coef")
    if t is None:
        raise RuntimeError(f"MoE L{sm.layer}: no hc_coef buffer (a build without the mHC nodes?)")
    cf = t.contiguous().view(torch.float32).reshape(-1) if t.dtype != torch.float32 else t.reshape(-1)
    return cf[:HC].detach().float().cpu()


def _apply_pre(R, pre, L, kind, d):
    """norm(pre . R) with layers.L.{kind}_norm.weight: hc_pre then attn_norm/ffn_norm. bf16, as hc_post_combine's
    own write is."""
    v = R.float().view(HC, HID)
    y = (pre.to(d)[:, None] * v.to(d)).sum(0)
    g = _read(f"layers.{L}.{kind}_norm.weight").float().to(d)
    return (y.bfloat16().float() * torch.rsqrt(y.square().mean() + cfg["norm_eps"]) * g).bfloat16()


def eg_next_x(sm, L, d):
    """Layer L's attention input, from the R'' the MoE L-1 just produced (synchronous: the buffer is read on the
    host, so the launch must have finished -- the price of the re-derivation, paid twice a step)."""
    return _apply_pre(sm.w["streams_eg"][0].reshape(-1), _hc_pre(sm), L, "attn", d)


EG_EMB = {}
if EG_ON:
    def _eg_rows(src):
        """The n-gram rows the prefill gathered for the Engram hosted by MoE `src`: [T, 6144] bf16.

        A dump without the probe is a hard error, not zeros: a zero emb makes the gate add gate(0)*value to every
        stream, which reads as a small modelling error instead of a missing input."""
        f = os.path.join(PRE, "layers", "step1", f"L{src:02d}_eg_emb.f32")
        if not os.path.exists(f):
            raise SystemExit(f"{f}: the dump has no Engram rows (re-dump with dump_hook.py's engram probe): "
                             f"layer {src}'s residual is REPLACED by engram(R, hash_ids), so a chain that skips it "
                             f"is not the model from layer {src + 1} on")
        v = np.fromfile(f, dtype=np.float32)
        t = torch.zeros(T, v.size, dtype=torch.bfloat16, device=dev)
        t[0].copy_(torch.from_numpy(v).to(dev).bfloat16())
        return t
    EG_EMB = {L: _eg_rows(EG_HOST[L]) for L in EG_HOST}
    log(f"Engram: layer(s) {EG_LAYERS} hosted in MoE {sorted(EG_HOST)}, rows {tuple(next(iter(EG_EMB.values())).shape)}")

# ---- the step's input: the decode token's embedding, normalized as the chain's first sublayer reads it ----------
# full_decode's PRE start, and the two things e2e_decode used to get wrong: (a) the token is the one the PREFILL
# sampled (golden ids[0]), not the prompt's last -- the prefill already consumed that one, so the first decode step
# sits at position len(prompt) feeding the prefill's sample; (b) x is normed_row(e, 0) = bf16(bf16(e) *
# rsqrt(mean(e^2) + eps) * attn_norm.weight[0]), NOT the raw embedding (the gamma's 12% spread is exactly the
# layer-0 attention error). S0 is the embedding in EVERY hc_mult stream (hc_broadcast): zeros make the first
# layer's post mix 4 empty streams and the whole chain starts wrong.
if a.token is not None:
    dec_tok = a.token
elif PRE:
    gp = a.golden or os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_bench32k.jsonl")
    if os.path.exists(gp):
        grows = [json.loads(l) for l in open(gp)]
        dec_tok = int(grows[a.row % len(grows)]["ids"][0])
    else:
        dec_tok = ids[-1]
else:
    dec_tok = ids[-1]
with safe_open(os.path.join(a.checkpoint, idx["embed.weight"]), "pt") as h:
    emb = h.get_tensor("embed.weight")[dec_tok].to(torch.bfloat16)
nm0 = "layers.0.attn_norm.weight"
with safe_open(os.path.join(a.checkpoint, idx[nm0]), "pt") as h:
    g0 = h.get_tensor(nm0)
v = emb.float()
X0 = (v * torch.rsqrt(v.square().mean() + cfg["norm_eps"]) * g0.float()).bfloat16()   # [HID], the first x
S0 = emb.repeat(HC)                                                                     # [HC*HID]
log(f"decode token {dec_tok}{' (golden ids[0])' if dec_tok != ids[-1] else ''}; "
    f"x0 norm {X0.float().norm():.3f} (normed), stream0 norm {emb.float().norm():.3f}")

# one region per layer kernel (attention, MoE) that has an exchange; the first take sets the slot size
pool = Lm.ExchangePool(GROUP, dev, 2 * a.layers + 4)
pool.take(int(a.pool_mb * 2**20))

attn, moe = [], []
# One window table for every attention layer. window_tables is a pure function of this step's positions, and all
# layers read the same positions, so a per-layer call recomputes the same [T] and [T, W] tables 40 times: measured
# 105 us of host time each (attn's host side is 135 us a launch against the MoE's 20). They are one set of buffers
# here, rebound into each layer (StaticAttn._rb: the mechanism bind_pools uses) and filled once in step(). Values
# are identical, so the numbers cannot move.
SHARE_TABLES = os.environ.get("DSV41_E2E_SHARE_TABLES", "1") != "0"
TABLES = None
if SHARE_TABLES:
    TABLES = {"positions": positions, "window_loc": torch.zeros(T, dtype=torch.int32, device=dev),
              "window_rows": torch.zeros(T, Lm.attn_demo.W, dtype=torch.int32, device=dev)}
t0 = time.perf_counter()
for L in range(a.layers):
    sa = Lm.StaticAttn(L, a.checkpoint, RANK, WORLD, dev, a.out, pos + 8, sources, bool(a.pdl), pool, None,
                       mhc=MHC, stats=False, engram_kv=None, weights_from=None)
    if TABLES is not None:
        shared = {k: v for k, v in TABLES.items() if k in sa.w}
        if len(shared) == len(TABLES):
            sa.w.update(shared)
            for k, v in shared.items():
                sa._rb(k, v.data_ptr())
            sa.tables_shared = True
        elif RANK == 0:
            log(f"  L{L}: window tables not shared (missing {sorted(set(TABLES) - set(shared))})")
    attn.append(sa)
    # the last layer's MoE also computes the head (norm.weight): its post makes the head's input
    head = L == a.layers - 1 and a.layers == 40
    # An Engram host computes the gate AFTER its post, so its own x_out is stale: norm_next is off and the next
    # attention's x is re-derived on the host from streams_eg (eg_next_x). StaticMoE asserts exactly that pair.
    en = bool(EG_ON and L in EG_HOST)
    sm = Lm.StaticMoE(L, a.checkpoint, RANK, WORLD, dev, a.out, bool(a.pdl), pool, norm_next=not en,
                      mhc=MHC, stats=False, engram=en, gamma_name="norm.weight" if head else None,
                      engram_kv=None, head=head)
    moe.append(sm)
    log(f"L{L} {variants[L]}: attn+MoE built, {(time.perf_counter() - t0) / 60:.1f} min, "
        f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
log(f"built {a.layers} layers in {(time.perf_counter() - t0) / 60:.1f} min on rank {RANK}")

# The caches are either the prefill's real bytes (--inputs-dir) or, with no prefill, the layer demos' random fill.
#
# ORDER MATTERS, and it was the whole of attn L3's 0x7fff. The serving build (STATIC_RESET_IN_KERNEL) memsets every
# buffer whose reader re-arms it (by_reader) to its sentinel, 0xFF -- which is NaN in e4m3 -- and "@prime" replays
# that once. Done AFTER a fill, it wipes the bytes just loaded and the first launch reads a NaN window; L3 is the
# first Reuse layer, its caches are the ones re-armed by its own reader, and it was the first layer to show it.
# So: prime EVERY layer first (the post-build sentinel state, as the host reset path leaves it), then fill.
for sa in attn:
    try:
        sa.sk.rebind("@prime", 0)
    except Exception as e:
        log(f"  @prime not available: {type(e).__name__}")
torch.cuda.synchronize()

for L, sa in enumerate(attn):
    if PRE:
        _fill_real_kv(sa, L)
    else:
        Lm.attn_demo.random_inputs(sa.w, sa.variant, sa.ratio, pos)
        # random_inputs masks window_cache's e4m3 bytes with 0xFE but NOT compress_cache's, so ~2/256 of the
        # compressed KV bytes are 0x7F / 0xFF -- NaN in e4m3. One layer never notices (its y is discarded); a
        # 40-layer chain pushes them through every mHC residual until x is all NaN by L3 (ratio 2's first layer).
        # Mask them here: the same random bytes, only the NaN encodings removed.
        cc = sa.w.get("compress_cache")
        if cc is not None:                    # rows are D//2 value bytes then D//16 scale bytes
            cc[..., :Lm.attn_demo.D // 2].bitwise_and_(0xFE)
        ik = sa.w.get("index_k")              # rows are 64 value bytes then 4 scale bytes
        if ik is not None:
            ik[..., :64].bitwise_and_(0xFE)
    Lm.attn_demo.window_tables(sa.w)          # (the real dump's rows too: the table is a function of positions)
if ZERO_KV:                                   # diagnosis only: K = 0 everywhere (value AND scale bytes), so
    for sa in attn:                           # S = q . 0 = 0 and no KV byte can produce a NaN; any NaN left is
        for nm in ("compress_cache", "index_k", "window_cache"):   # the q side or the kernel's own arithmetic
            t = sa.w.get(nm)
            if t is not None:
                t.zero_()
if MASK_FF:                                   # diagnosis only: 0xFF is NaN in e4m3, and the dump has some in its
    for sa in attn:
        for nm, nv in (("compress_cache", Lm.attn_demo.D // 2), ("index_k", 64)):   # value bytes. Masking says
            t = sa.w.get(nm)                  # whether the chain reads them at all
            if t is not None:
                t.view(torch.uint8)[..., :nv].bitwise_and_(0xFE)
torch.cuda.synchronize()
# Every rank builds its own copy, and the build times differ (nvcc per rank). A static layer's sum_store collects
# all ranks' parts, so a rank that launches first spins inside the kernel until the last rank launches: barrier here.
dist.barrier()
zeros_R = torch.zeros(T, HC, HID, dtype=torch.bfloat16, device=dev)
zeros_R[0].copy_(S0.view(HC, HID).to(dev))   # hc_broadcast: the prefill's embedding in every stream at row 0
# T = 1: one row. T = 8 (the design size): row 0 is the step's real input and the padding rows stay zero (SGLang's
# padding rows are zeros too; duplicating the token into them made the chain's stats differ from the reference).
x = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)
x[0].copy_(X0.to(dev))
p = positions
DEBUG = os.environ.get("DSV41_E2E_DEBUG", "0") != "0"


def _trace(layer, tag):
    """TRACE: every float/uint buffer this layer holds that is polluted (any 0xFF byte, any non-finite value).

    Called before and after a launch with the GPU idle (the caller synchronizes): the difference says whether the
    layer WROTE the pollution or READ it. A buffer that is already polluted BEFORE its first launch is the one the
    reset protocol failed to prepare -- window_cache is not in any layer's reset list, but the polled ones are."""
    rows = []
    for nm, t in (layer.w or {}).items():
        if not torch.is_tensor(t) or t.numel() == 0:
            continue
        if t.dtype in (torch.float32, torch.bfloat16, torch.float16):
            nf = int((~torch.isfinite(t.detach().flatten())).sum())
            if nf:
                # The payload says which: 0xFF.. is this codebase's not-written sentinel (REARM), anything else
                # arithmetic (0x7FFF in bf16 is the fp32 0x7fffffff __fdividef produces).
                pat = torch.unique(t.detach().view(torch.int16).flatten()[~torch.isfinite(t.detach().flatten())]
                                   ) if t.dtype == torch.bfloat16 else None
                ps = "" if pat is None else f" bits {[hex(v & 0xFFFF) for v in pat.tolist()[:4]]}"
                rows.append(f"{nm} {tuple(t.shape)} {t.dtype}: {nf}/{t.numel()} NOT FINITE{ps}")
        b = t.detach().view(torch.uint8) if t.dtype == torch.uint8 else None
        if b is not None:
            n = int((b == 0xFF).sum())
            if n:
                rows.append(f"{nm} {tuple(t.shape)} uint8: {n}/{b.numel()} bytes 0xFF")
        elif t.is_floating_point() is False:   # int32/int64: the step's tables. -1 is this build's 0xFF sentinel here
            f = t.detach().flatten()
            if t.numel() <= 16:                # positions, window_loc: always show
                rows.append(f"{nm} {tuple(t.shape)} {t.dtype}: {f.tolist()}")
            elif t.numel() <= 4096:
                n = int((f < 0).sum())
                if n:
                    rows.append(f"{nm} {tuple(t.shape)} {t.dtype}: {n}/{f.numel()} negative (first {f[:6].tolist()})")
            else:
                n = int((f < 0).sum())
                if n:
                    rows.append(f"{nm} {tuple(t.shape)} {t.dtype}: {n}/{f.numel()} negative (first {f[:6].tolist()})")
        elif t.numel() <= 16:                  # attn_sink (8 fp32): a NaN sink makes every exp() NaN
            rows.append(f"{nm} {tuple(t.shape)} {t.dtype}: {[round(v, 5) for v in t.detach().flatten().float().tolist()]}")
    # the node buffers: the kernel's own intermediate results, and where the by_reader resets live (host.cuh's
    # s.resets is built from these, not from w). All uint8; a build-time memset leaves the by_reader ones 0xFF.
    nb = getattr(layer, "bufs", None)
    if isinstance(nb, list) and nb:
        for si, sd in enumerate(nb):           # one dict per slot: the kernel writes slot si, not just bufs[0]
            for nm, t in (sd or {}).items():
                if not torch.is_tensor(t) or t.numel() == 0:
                    continue
                n = int((t.detach().view(torch.uint8) == 0xFF).sum())
                if n:
                    head = t.detach().view(torch.uint8).flatten()[:8].tolist()
                    # which QUARTER is polluted: a combine="slots" layout is [K parts][...], so a whole untouched part
                    # is one contiguous 0xFF run (which part of the K loop never ran or never landed)
                    u = t.detach().view(torch.uint8).flatten()
                    q4 = [int((u[i * u.numel() // 4:(i + 1) * u.numel() // 4] == 0xFF).sum()) for i in range(4)]
                    rows.append(f"[{si}]node:{nm} {tuple(t.shape)}: {n}/{t.numel()} bytes 0xFF"
                                f"{'  head ' + ''.join(f'{v:02x}' for v in head) if n == t.numel() * t.element_size() else ''}"
                                f"  quarters {q4}")
    return f"  [{tag}] " + ("; ".join(rows[:22]) if rows else "all clean")


def _progress(layer, tag=""):
    """DEBUG: dump this kernel's task counters and start barrier, so a spinning node is visible while it never returns."""
    sk = layer.sk
    try:
        import numpy as np
        lay = sk.layout()
        # the layer's module wraps only this process's GPU: its index is 0, not RANK
        cnt = np.frombuffer(sk.read(0, "counters"), dtype=np.uint32)
        bar = np.frombuffer(sk.read(0, "start_barrier"), dtype=np.int64)
        per = lay["counters_per_line"]
        hot = [(i * per + j, int(v)) for i in range(lay["counter_lines"])
               for j, v in enumerate(cnt[i * per:(i + 1) * per]) if v]
        st = np.frombuffer(sk.read(0, "stamps"), dtype=np.int64).reshape(lay["num_sms"], lay["num_stamps"])
        entered = int((st[:, lay["stamp_start"]] > 0).sum())
        t0 = int((st[:, lay["stamp_task0"]] > 0).sum())
        t1 = int((st[:, lay["stamp_task1"]] > 0).sum())
        # Every rank, to a file: the point is whether the other seven are even in the kernel.
        with open(f"{a.out}/hang_rank{RANK}_{tag.replace(' ', '_')}.txt", "w") as fh:
            fh.write(f"{tag} rank {RANK}: {len(hot)}/{cnt.size} nonzero\n")
            fh.write(f"  (line*{per}+slot -> value): {hot[:60]}\n")
            fh.write(f"  SMs entered {entered}/{lay['num_sms']}, task0 started {t0}, task1 started {t1}, "
                     f"ended {int((st[:, lay['stamp_end']] > 0).sum())}\n")
            try:                                  # the timing build: every task stamps its start and end
                fh.write("  per SM stamps (start, task0, task1, end):\n")
                for smx in range(st.shape[0]):
                    row = [int(v) for v in st[smx]]
                    if any(row):
                        fh.write(f"    SM{smx}: {row}\n")
            except Exception:
                pass
            # What the spinning nodes poll are tensors, not counters: list them (metadata only -- any CUDA op
            # here would queue behind the kernel that never returns, and the dump would never be written).
            fh.write("  buffers (name, shape, dtype, device ptr):\n")
            for nm, t in list(layer.w.items())[:40]:
                try:
                    if not torch.is_tensor(t) or t.numel() == 0:
                        continue
                    fh.write(f"    {nm} {tuple(t.shape)} {t.dtype} @0x{t.data_ptr():x}\n")
                except Exception as e:
                    fh.write(f"    {nm}: unreadable {type(e).__name__}\n")
        log(f"    {tag} rank {RANK}: {len(hot)}/{cnt.size} counters nonzero, start_barrier {int(bar[0])}")
        log(f"    (line*{per}+slot -> value): {hot[:40]}")
        log(f"    SMs entered: {entered}/{lay['num_sms']}, ended: "
            f"{int((st[:, lay['stamp_end']] > 0).sum())}")
    except Exception as e:
        print(f"[e2e] {tag} rank {RANK} counters unreadable: {type(e).__name__}: {e}", flush=True)


X_EG = torch.zeros(T, HID, dtype=torch.bfloat16, device=dev)   # (the host re-derived x of an Engram layer)
_CAP = False        # True while a CUDA graph is being captured: nothing inside may touch the host
_G = []             # one graph per segment (see _segments)
_SEGS = []
_SX = _SR = None    # the [T, HID] x and the [T, HC, HID] residual the next sublayer reads


def _sublayer(i: int) -> None:
    """Layer i's two launches (attention, then MoE) and the state the next sublayer reads.

    With mHC in the kernels a sublayer does not write y: the attention's sum_store only sends its part (output=None)
    and the kernel's post writes x_out, the normed input of the next sublayer (gamma = the next norm's weight). So
    the chain is x_out -> the next kernel's x, and streams_out -> the next kernel's R.

    Everything here is device work on the current stream plus dict lookups, so the same function is what a captured
    graph replays (DSV41_E2E_GRAPH). _SX / _SR only decide which ADDRESSES the launches bind, and every one of them
    is a layer buffer, so a replay reads whatever the previous segment's kernels wrote -- no host value is involved.
    Host-only instrumentation is therefore gated on _CAP rather than left in: a torch.cuda.synchronize() inside a
    capture is an error, and a dist.barrier() is worse."""
    global _SX, _SR
    xx, R = _SX, _SR
    if not _CAP:
        if DEBUG:                            # every rank, so a rank that never reaches the launch is visible
            print(f"[e2e] rank {RANK} launch attn L{i} ({variants[i]})", flush=True)
        if TRACE and i in TRACE:
            torch.cuda.synchronize(dev)
            log(f"attn L{i} BEFORE" + "\n" + _trace(attn[i], f"L{i} before"))
        if TAG:
            for (tl, tn) in TAG:
                if tl != i:
                    continue
                if tn in (attn[i].w or {}):
                    t = attn[i].w[tn]
                else:
                    t = (attn[i].bufs[0] or {}).get(tn) if attn[i].bufs else None
                if t is not None:
                    t.view(torch.uint8).fill_(0xAB)
                    log(f"attn L{i}: {tn} tagged 0xAB")
    _t0 = time.perf_counter() if (HOSTT and not _CAP) else 0.0
    attn[i].run(xx, p, R)
    if not _CAP:
        _wait(attn[i], f"attn L{i}")
        _sync()                              # the ranks in lockstep before the next kernel (see _sync)
        if HOSTT:
            _HW["attn"] = _HW.get("attn", 0.0) + time.perf_counter() - _t0
        if TRACE and i in TRACE:
            torch.cuda.synchronize(dev)
            log(f"attn L{i} AFTER" + "\n" + _trace(attn[i], f"L{i} after"))
    R = attn[i].w["streams_out"].view(T, HC, HID)
    xx = attn[i].w["x_out"]                  # the post's next input = this MoE's x
    _t0 = time.perf_counter() if (HOSTT and not _CAP) else 0.0
    moe[i].run(xx, R, emb=EG_EMB.get(i))
    if not _CAP:
        _wait(moe[i], f"moe L{i}")
        _sync()
        if HOSTT:
            _HW["moe"] = _HW.get("moe", 0.0) + time.perf_counter() - _t0
    # R'' when this MoE hosts an Engram (its residual REPLACES R', so the next attention's post reads R''), R'
    # otherwise -- full_decode.py:498 takes the same branch. Using streams_out here was the harness's one real
    # divergence from the reference chain: bit-exact at L0, 56% off at L1, and 1.35x of the logits band instead
    # of 0.87x (FINDINGS 12.45).
    R = (moe[i].w["streams_eg"] if "streams_eg" in moe[i].w else moe[i].w["streams_out"]).view(T, HC, HID)
    # (norm_next) the next attention's x; the last layer's: the head's input. An Engram host writes no x_out --
    # EG_LAYERS follows every host with its own layer, which overwrites xx from R'' first.
    xx = moe[i].w.get("x_out") if not (EG_ON and i in EG_HOST) else xx
    if not _CAP and a.sync_every and (i + 1) % a.sync_every == 0 and i + 1 < a.layers:
        _sync()
    _SX, _SR = xx, R


def _segment(lo: int, hi: int) -> None:
    for i in range(lo, hi + 1):
        _sublayer(i)


def _segments():
    """The step's launch ranges, split at every layer whose x the HOST has to derive.

    An Engram layer's attention reads norm(pre . R''), and R'' is a device value the previous kernels just wrote, so
    that read needs a real synchronize -- which a CUDA graph cannot contain. Everything between two such layers is a
    pure launch stream and becomes one graph: without this, the harness pays 80 host round trips (1.79 ms of the
    5.04 ms step) to submit work the GPU could have started itself."""
    cuts = [c for c in (sorted(EG_LAYERS) if EG_ON else []) if 0 < c < a.layers]
    segs, lo = [], 0
    for c in cuts:
        if c > lo:
            segs.append((lo, c - 1))
        lo = c
    segs.append((lo, a.layers - 1))
    return segs


def _seg_entry(lo: int):
    """The (x, R) a segment's first sublayer binds: the embedding or the Engram layer's re-derived x, and zeros or the
    Engram host's R''. Only used at capture time -- a replay reads the buffers."""
    if lo == 0:
        return x, zeros_R
    return X_EG, moe[lo - 1].w["streams_eg"].view(T, HC, HID)


def capture_graph() -> None:
    """One uncaptured pass over every segment (the warm-up a capture needs), then one graph per segment."""
    global _CAP, _G, _SEGS, _SX, _SR
    assert not _CAP and not _G
    _SEGS = _segments()
    _CAP = True
    try:
        _SX, _SR = _seg_entry(0)
        for k, (lo, hi) in enumerate(_SEGS):
            if k:
                _SX, _SR = _seg_entry(lo)
            _segment(lo, hi)
        torch.cuda.synchronize(dev)
        _G = []
        for lo, hi in _SEGS:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _SX, _SR = _seg_entry(lo)
                _segment(lo, hi)
            _G.append(g)
    finally:
        _CAP = False
    if RANK == 0:
        log(f"captured {len(_G)} graphs over segments {_SEGS} ({a.layers} layers, "
            f"{sum(h - l + 1 for l, h in _SEGS)} sublayer pairs)")


def _sig():
    """Diagnosis (rank 0): a bitwise fingerprint of the chain's answer after every step.

    A replay of a captured graph must produce the same bits as the launch it replaced, and every step of this harness
    re-runs the SAME token at the SAME position, so all steps must agree with each other. Printing the first step a
    fingerprint changes names the replay count at which a captured graph stops being idempotent -- a question about
    the reset protocol (core.cuh kernel_begin's double-buffered parity), not about the arithmetic."""
    if not SIG or RANK != 0:
        return
    _SIG_IDX[0] += 1
    # every sublayer's two outputs: the first layer that changes between two steps is where the drift starts, and
    # which one of the 80 launches it is decides between "a set was not reset" and "a set was read one launch early"
    cur = {}
    for i in range(a.layers):
        for nm, lay in (("attn", attn[i]), ("moe", moe[i])):
            for t in ("x_out", "streams_out"):
                v = lay.w.get(t)
                if v is not None:
                    cur[f"{nm}{i}.{t}"] = int(v.view(torch.int32).to(torch.int64).sum())
    lay = moe[-1].w.get("logits")
    if lay is not None:
        fp = lay.detach().view(torch.int32).reshape(-1).cpu()
        _SIG_LOG.append((_SIG_IDX[0], int(fp[0]), int(fp.argmax())))
    if _SIG_FIRST[0] is None:
        cur = {}
        # every tensor of the first few layers (not just the outputs): attn1 is a Reuse layer, so it reads its source
        # layer's compress_cache / compress_idx, and a drift in one of THOSE with an unchanged x_out names the kernel
        # whose side buffer is not being re-armed. X_EG is the host-written Engram input, hashed to separate "the host
        # fed a different x" from "the kernel read the same x and wrote different bits".
        for i in range(min(3, a.layers)):
            for nm, lay in (("attn", attn[i]), ("moe", moe[i])):
                for k, v in list(lay.w.items()):
                    if torch.is_tensor(v) and v.numel():
                        cur[f"{nm}{i}.{k}"] = int(v.view(torch.int32).to(torch.int64).sum()) \
                            if v.dtype != torch.uint8 else int(v.to(torch.int64).sum())
                b = lay.bufs[0] if lay.bufs else None
                for k, v in (b or {}).items():
                    if torch.is_tensor(v) and v.numel():
                        cur[f"{nm}{i}/buf.{k}"] = int(v.view(torch.int32).to(torch.int64).sum()) \
                            if v.dtype != torch.uint8 else int(v.to(torch.int64).sum())
        cur["X_EG"] = int(X_EG.view(torch.int32).to(torch.int64).sum())
        _SIG_FIRST[0] = cur
        log(f"  sig step {_SIG_IDX[0]:>2}: baseline ({len(cur)} tensors of layers 0-2)")
        return
    bad = [k for k in cur if _SIG_FIRST[0].get(k) != cur[k]]
    tag = "" if not bad else f"  <- {len(bad)} differ: {bad[:4]}"
    log(f"  sig step {_SIG_IDX[0]:>2}: {'same' if not bad else 'drift'}{tag}")


_SIG_LOG = []
_SIG_FIRST = [None]
_SIG_IDX = [0]


def step():
    """One decode step: the layers' launches in order (the last MoE's kernel also runs the head), as 3 graph replays
    joined by the 2 host-side Engram derivations."""
    global _SX, _SR
    if CLR_IDX:                              # diagnosis: the Reuse layers' top-k back to the pre-producer value
        done = set()
        for sa in attn:
            t = sa.w.get("compress_idx")
            if t is not None and t.data_ptr() not in done:
                done.add(t.data_ptr())
                t.fill_(-1)
    if DEBUG:
        print(f"[e2e] rank {RANK} entering the step: rows {T}, positions {int(p[0])}", flush=True)
    if TABLES is not None:                   # once for the step: every attention layer reads these (see attn's build)
        Lm.attn_demo.window_tables(TABLES)
    if _G:
        for k, (lo, hi) in enumerate(_SEGS):
            if lo:
                # The attention of an Engram layer reads norm(pre . R''), R'' the residual the MoE of lo-1 just
                # built. The kernel wrote its own x_out from R' (its post), one sublayer stale: re-derive on the host.
                torch.cuda.synchronize(dev)
                X_EG[0].copy_(eg_next_x(moe[lo - 1], lo, dev))
            _G[k].replay()
        return
    _SX, _SR = _seg_entry(0)
    for i in range(a.layers):
        if EG_ON and i in EG_LAYERS:
            torch.cuda.synchronize(dev)
            X_EG[0].copy_(eg_next_x(moe[i - 1], i, dev))
            _SX = X_EG
        _sublayer(i)


def _sync():
    """The ranks back in lockstep. A static layer's sum_store collects every rank's part through the exchange regions,
    so all ranks must run the same kernel: their streams are independent, so without this they drift apart."""
    # torch.cuda.synchronize() waits for THIS process's GPU only: it leaves the ranks free to be at different
    # launches, and sum_store then writes exchange set (n+1)%2 while a peer polls set n%2 (hc_post_combine's 4
    # CTAs never end). The barrier is what launch_func does host-side for a one-process run.
    # With -DSTATIC_TP_BARRIER the kernel does this itself (core.cuh kernel_begin, before the resets), so the host
    # round trip is only paid when the build has no kernel barrier -- the timed path stays pure otherwise.
    if FULL_SYNC:
        # The decisive variant for a reuse layer: its compress_idx is the SOURCE layer's indexer_topk output, written
        # by the launch before it. A host barrier does not wait for the GPU, so the next launch can read that buffer
        # mid-write; this waits for the kernel to be DONE, then lines the ranks up. (Paid per sublayer: not the
        # timed path.)
        torch.cuda.synchronize()
        dist.barrier()
        return
    if NOSYNC or (TPB and not FORCE_BAR):
        return
    dist.barrier()


def _wait(layer, tag: str, timeout: float = None):
    """DEBUG: a watchdog over the layer's launch. A thread dumps the spinning kernel's counters and ends the process
    when the launch outlives `timeout` (sk.done() is useless here: its event belongs to launch_func, not
    launch_stream, so it says done at once). DSV41_E2E_ASYNC=1 skips the host synchronize, so the watchdog times the
    launches exactly as the fast path queues them."""
    if not DEBUG:
        return
    import os as _os
    import threading
    timeout = timeout if timeout is not None else float(_os.environ.get("DSV41_E2E_TIMEOUT", "60"))
    fired = []
    t0 = time.monotonic()

    def watchdog():
        while not fired and time.monotonic() - t0 < timeout:
            time.sleep(0.1)
        if not fired:
            log(f"  HUNG at {tag} (>{timeout:g}s)")
            _progress(layer, tag)
            time.sleep(45)          # the other ranks' watchdogs fire a little later: give them their dump first
            _progress(layer, tag + "+45s")   # a second sample: a count that creeps is slowness, a frozen one a deadlock
            time.sleep(30)
            _os._exit(3)

    th = threading.Thread(target=watchdog, daemon=True)
    th.start()
    if _os.environ.get("DSV41_E2E_ASYNC", "0") == "0":
        torch.cuda.synchronize()
    fired.append(1)
    if _os.environ.get("DSV41_E2E_ASYNC", "0") == "0":
        xo = layer.w.get("x_out")
        so = layer.w.get("streams_out")
        log(f"  ok {tag}: x_out norm {xo.float().norm():.4g} rms {xo.float().square().mean().sqrt():.4g}"
            + (f", streams_out rms {so.float().square().mean().sqrt():.4g}" if so is not None else ""))


if PTRS:                                      # diagnosis: two names on one address would explain a 0xFF write into a
    for li, lay in enumerate(attn):           # buffer nothing resets (o_q: reset -1, by_reader 0)
        seen = {}
        for sd in (lay.bufs or []):
            for nm, t in (sd or {}).items():
                if not torch.is_tensor(t) or t.numel() == 0:
                    continue
                p = t.data_ptr()
                alias = seen.get(p)
                seen[p] = nm
                log(f"  L{li} attn {nm:20s} 0x{p:x} {t.numel() * t.element_size():7d}B"
                    + (f"  ALIASES {alias}" if alias else ""))

step()
torch.cuda.synchronize()
log("one step done" + (f"; logits {tuple(moe[-1].w['logits'].shape)}" if moe[-1].head else ""))
# Which sublayer first went bad. Every layer keeps its own x_out, so one pass locates the break: this is the line
# that turns "the output is NaN" into "L3 or L4, by timing" (FINDINGS_correctness.md §4). Ran on the default path
# too, not only under DEBUG: a run whose numbers are NaN must not read as a clean step.
def _fin(t):
    return t is None or bool(torch.isfinite(t).all())

bad = []
for i in range(a.layers):
    if not _fin(attn[i].w.get("x_out")):
        bad.append(f"attn L{i}")
    if not _fin(moe[i].w.get("x_out")):
        bad.append(f"moe L{i}")
lg = moe[-1].w.get("logits") if moe[-1].head else None
if lg is not None and not _fin(lg):
    bad.append("head")
if RANK == 0:
    if bad:
        log(f"NOT FINITE in {len(bad)} sublayers, the first is {bad[0]}: {', '.join(bad[:8])}"
            + (f", ... ({len(bad)})" if len(bad) > 8 else ""))
        # WHERE THE NaN CAME FROM: 0xFF is this codebase's "not written yet" sentinel (config.cuh REARM), so a NaN whose
        # payload is all ones is a buffer nobody wrote, not arithmetic. A real arithmetic NaN has a payload of zeros.
        who = bad[0].split()[0]
        idx = int(bad[0].split("L")[1]) if "L" in bad[0] else None
        lay = (attn if who == "attn" else moe)[idx] if idx is not None else None
        t = lay.w.get("x_out") if lay is not None else None
        if t is not None:
            bits = t.detach().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32).flatten()
            fin = torch.isfinite(t.detach().flatten())
            pat = torch.unique(bits[~fin]).tolist()
            log(f"  {bad[0]} x_out: {int((~fin).sum())}/{t.numel()} not finite; their bit patterns "
                f"{[hex(b & 0xFFFFFFFF) for b in pat[:8]]}")
            if t.dtype == torch.bfloat16:
                cnt = {hex(b & 0xFFFF): int((bits == b).sum()) for b in pat[:4]}
                log(f"  same bits elsewhere in the tensor: {cnt}")
        # The shared tensors: a `reuse` layer reads the source layer's compress_cache/compress_idx by reference, so a
        # reset protocol that fires in the source's launch is visible here. 0xFF is the not-written-yet sentinel.
        # WHICH BUFFER first went non-finite: x_out is the last thing the kernel writes, so a NaN there says nothing
        # about where it started. Every other float tensor this sublayer (and the one before it) wrote is checked, so
        # the first bad value names the node instead of leaving it to inference.
        for L in (max(0, (idx or 0) - 1), idx or 0):
            src = attn if L < (idx or 0) else (attn if who == "attn" else moe)
            lay2 = src[L]
            for nm2, t2 in (lay2.w or {}).items():
                if not torch.is_tensor(t2) or t2.numel() == 0 or not t2.is_floating_point():
                    continue
                nf = int((~torch.isfinite(t2.detach().flatten())).sum())
                if nf:
                    log(f"  {type(lay2).__name__} L{L}.{nm2} {tuple(t2.shape)} {t2.dtype}: {nf}/{t2.numel()} not finite")
        for nm in ("compress_cache", "compress_idx", "candidates", "index_k"):
            for L in (max(0, (idx or 0) - 1), idx or 0):
                tt = (attn[L].w or {}).get(nm) if L < len(attn) else None
                if not torch.is_tensor(tt) or tt.numel() == 0:
                    continue
                bb = tt.detach().view(torch.uint8)
                log(f"  attn L{L}.{nm} {tuple(tt.shape)} {tt.dtype}: 0xFF {int((bb == 0xFF).sum())}/{bb.numel()} bytes, "
                    f"same object as L{idx}: {tt.data_ptr() == attn[idx].w.get(nm, tt).data_ptr()}")
    else:
        log(f"all {a.layers} attn + {a.layers} moe x_out finite"
            + (f", logits {tuple(lg.shape)} finite" if lg is not None else ""))

if DEBUG:   # now the GPU is idle: what the polled buffers hold AFTER the step says whether a producer
            # wrote late (a missing release/order) or never wrote the region at all (a tiling bug)
    with open(f"{a.out}/post_rank{RANK}.txt", "w") as fh:
        for i, lay in enumerate(attn):
            for nm, t in lay.w.items():
                if not torch.is_tensor(t) or t.numel() == 0 or t.dtype != torch.float32:
                    continue
                b = t.detach().view(torch.uint8)
                nff = int((b == 0xFF).sum())
                if nff == 0:
                    continue
                fh.write(f"attn{i}.{nm} {tuple(t.shape)}: 0xFF {nff}/{b.numel()}")
                if t.ndim >= 2 and t.shape[0] == T:
                    per = (b.reshape(T, -1) == 0xFF).sum(1).tolist()
                    fh.write(f" per token {per}")
                fh.write("\n")

    # The exchange region: the ONLY cross-rank state (FINDINGS 4.3), and the one thing the dump above cannot see --
    # it lists the layer's declared tensors, not the region the sum_store publishes into. It is zero-filled at
    # allocation (_exchange_region), so on its own a region that a peer never wrote reads all zero, and a 0xFF here
    # can only have been pushed in by someone: either says which witness failed.
    if pool.buf is not None:
        with open(f"{a.out}/exchange_rank{RANK}.txt", "w") as fh:
            b = pool.buf.detach().contiguous().view(torch.uint8)
            slot = int(pool.slot)
            for k in range(int(pool.taken)):
                r = b[k * slot:(k + 1) * slot]
                nz = int((r != 0).sum())
                nff = int((r == 0xFF).sum())
                fh.write(f"region {k:>2} slot {slot}: nonzero {nz} of {r.numel()}, 0xFF {nff}\n")

    # The kernel's own scratch (wo_a_part, sparse_attn_ml, the exchange y_sent of each layer ...): 0xFF-filled and
    # polled, and invisible to the dumps above. DSV41_RESET_DEBUG=1 makes each layer print their addresses on
    # stderr before the run ([rl] lines: core.cuh / host.cuh), so with the stderr captured to DSV41_E2E_DBGLOG
    # this reads every one of them back IN THIS PROCESS -- the pointers are this process's, so another process
    # cannot read them (dump_buffers.py documents that and is for the standalone case).
    # It is the state at the END of the step, so a NaN cascade smears everything after the first bad layer: to
    # read the census right at the failure, run DSV41_E2E_LAYERS=5, whose last layer is the first one that goes
    # NaN (attn L4), instead of a 6-layer chain where L5 is last.
    # (dbglog, not log: `log` is this module's print helper at line 72, and naming a local `log` here shadowed it
    # -- ast.parse cannot see that, the run fails with "'str' object is not callable" at the census print.)
    dbglog = os.environ.get("DSV41_E2E_DBGLOG")
    if dbglog and os.path.exists(dbglog):
        import ctypes
        import importlib.util
        from cuda.bindings import runtime as rt

        def _dev_bytes(ptr, n):
            """n bytes at a raw device address of THIS process (the layer's own allocation)."""
            buf = ctypes.create_string_buffer(n)
            (err,) = rt.cudaMemcpy(buf, ptr, n, rt.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            if err != rt.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpy({ptr:#x}, {n}): {err}")
            return buf.raw[:n]

        spec = importlib.util.spec_from_file_location(
            "_db", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump_buffers.py"))
        db = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(db)
        ents = [e for e in db.resets(dbglog) if e["rank"] == RANK]
        # the layer's declarations, for the buffer names: <out>/rank<R>/<attn|moe><layer>_t<T>/build/layer.cu
        def _cu(kind):
            return os.path.join(a.out, f"rank{RANK}", f"{kind}0_t{T}", "build", "layer.cu")
        cu = next((p for p in map(_cu, ("attn", "moe")) if os.path.exists(p)), None)
        decls = db.declarations(cu) if cu else []
        with open(f"{a.out}/buffers_rank{RANK}.txt", "w") as fh:
            if not ents:
                fh.write(f"no [rl] rank {RANK} line in {dbglog}: DSV41_RESET_DEBUG=1 was not on, or the stderr went "
                         "elsewhere\n")
            for label, n, v, rd, n_nan, nff, nz, first in db.analyze(
                    RANK, ents, decls, copy=lambda p, k: _dev_bytes(p, k)):
                fh.write(f"{label} bytes {n} v {v} by_reader {rd}: NaN {n_nan} 0xFF {nff} nonzero {nz}"
                         f" first NaN {first}\n")
        log(f"buffer census -> {a.out}/buffers_rank{RANK}.txt ({len(ents)} reset entries)")

if a.dump:
    # The step above ran every layer once from the deterministic init (random_inputs seeds each tensor's own
    # generator), so two processes with the same arguments start from the same bytes: the dumps must match bit
    # for bit. That is what proves the async (no host synchronize) path equals the fully synchronized one.
    import numpy as np
    os.makedirs(a.dump, exist_ok=True)
    for i in range(a.layers):
        for name, lay in (("attn", attn[i]), ("moe", moe[i])):
            for t in ("x_out", "streams_out"):
                v = lay.w.get(t)
                if v is not None:
                    v.detach().cpu().contiguous().numpy().tofile(
                        os.path.join(a.dump, f"{name}{i:02d}_{t}.bin"))
    if moe[-1].head and "logits" in moe[-1].w:
        moe[-1].w["logits"].detach().cpu().contiguous().numpy().tofile(os.path.join(a.dump, "logits.bin"))
    log(f"dumped {a.layers} layers' x_out/streams_out and the logits to {a.dump}")

if os.environ.get("DSV41_E2E_ONE", "0") != "0":
    # One step and the verdict. The finite check above is the whole point of this mode: a chain without a barrier
    # can spin in the *next* launch (the drift shows up as a hang once the first step has run), so it must not walk
    # into the warmup's dist.barrier() and the timing loop.
    log("DSV41_E2E_ONE: the one step and its verdict, no timing loop")
    os._exit(0)

if GRAPH:
    capture_graph()

for _ in range(a.warmup):
    step()
    _sig()
torch.cuda.synchronize()
dist.barrier()
start, end = torch.cuda.Event(True), torch.cuda.Event(True)
evs = []
_prof = None
if PROFILE:
    import cProfile
    _prof = cProfile.Profile()
    _prof.enable()
for _ in range(a.reps):
    torch.empty(512 << 20, dtype=torch.uint8, device=dev).zero_()   # empty L2, as the layer demos do
    torch.cuda.synchronize()
    _HW.clear()
    start.record()
    step()
    end.record()
    torch.cuda.synchronize()
    _sig()
    evs.append(start.elapsed_time(end) * 1e3)
    if HOSTT and RANK == 0:
        at, mo = _HW.get("attn", 0.0) * 1e6, _HW.get("moe", 0.0) * 1e6
        log(f"  host per step: attn {at:.0f} us, moe {mo:.0f} us, total {at + mo:.0f} us "
            f"of a {evs[-1]:.0f} us step ({100 * (at + mo) / evs[-1]:.0f}%)")
evs.sort()
med = evs[len(evs) // 2]
if _prof is not None:
    _prof.disable()
    if RANK == 0:
        import pstats
        st = pstats.Stats(_prof)
        st.sort_stats("tottime")
        print(f"\n[e2e] cProfile over {a.reps} steps, top 25 by self time (accumulated over all {a.reps} steps):",
              flush=True)
        st.print_stats(25)
        print("[e2e] top 25 by cumulative time:", flush=True)
        st.sort_stats("cumulative")
        st.print_stats(25)
log(f"{a.layers} layers at position {pos}: median step {med:.1f} us = {med / 1000:.2f} ms "
    f"(p10 {evs[len(evs) // 10]:.1f}, p90 {evs[9 * len(evs) // 10]:.1f}); {1e6 / med:.0f} tok/s")
if RANK == 0:
    os.makedirs(a.out, exist_ok=True)
    json.dump({"row": row["id"], "pos": pos, "layers": a.layers, "tp": WORLD, "reps": a.reps,
               "step_us_median": med, "step_us_p10": evs[len(evs) // 10], "step_us_p90": evs[9 * len(evs) // 10],
               "tok_s": 1e6 / med, "variants": variants[:a.layers]},
              open(os.path.join(a.out, "e2e_decode.json"), "w"), indent=1)
    log(f"wrote {os.path.join(a.out, 'e2e_decode.json')}")
    if DUMP_LOGITS:
        # The last sublayer's output is the chain's answer for any layer count (logits only exist when the last MoE has
        # the head, i.e. at the model's own depth): both spellings must agree bit for bit.
        last = moe[-1].w
        torch.save({k: last[k].detach().float().cpu() for k in ("x_out", "streams_out", "logits") if last.get(k) is not None},
                   os.path.join(a.out, DUMP_LOGITS))
        log(f"wrote {os.path.join(a.out, DUMP_LOGITS)}")
    if CHAINDUMP:
        os.makedirs(CHAINDUMP, exist_ok=True)
        for li in range(a.layers):
            for pre, lay in (("attn", attn[li]), ("moe", moe[li])):
                for key in ("x_out", "streams_out"):
                    t = lay.w.get(key)
                    if t is not None:
                        # raw bf16 bytes, as the chain dump writes them (numpy has no bf16: view as uint16)
                        t.detach().contiguous().cpu().view(torch.uint16).numpy().tofile(
                            os.path.join(CHAINDUMP, f"{pre}{li}_{key}.bin"))
        log(f"wrote the chain dump to {CHAINDUMP}")
if a.spans:
    # Per-launch attribution. The spans must be read AFTER the launches finish: span_us() is a host read of the
    # kernel's own stamps, so calling it inside the loop would synchronize and destroy the very gaps being measured.
    # Every kernel stamps [start barrier, last SM end] in globaltimer, so the values survive to be read here.
    from mirage.mpk.static_harness import Harness
    step()
    torch.cuda.synchronize()
    dist.barrier()
    names, spans = [], []
    for i in range(a.layers):
        names += [f"attn{i}", f"moe{i}"]
        spans += [Harness(attn[i].sk).span_us(), Harness(moe[i].sk).span_us()]
    if RANK == 0:
        tot = sum(spans)
        for n, v in zip(names, spans):
            print(f"[span] {n:8s} {v:8.1f} us", flush=True)
        print(f"[span] sum {tot:.1f} us over {len(spans)} launches; the timed step was {med:.1f} us "
              f"-> host gaps {med - tot:.1f} us ({100 * (med - tot) / med:.1f}%)", flush=True)
        top = sorted(zip(spans, names), reverse=True)[:8]
        print("[span] the 8 heaviest: " + "  ".join(f"{n} {v:.0f}" for v, n in top), flush=True)
        attn_sum = sum(v for n, v in zip(names, spans) if n.startswith("attn"))
        moe_sum = tot - attn_sum
        print(f"[span] attention total {attn_sum:.1f} us ({100 * attn_sum / med:.0f}% of the step), "
              f"MoE total {moe_sum:.1f} us ({100 * moe_sum / med:.0f}%)", flush=True)
        os.makedirs(a.out, exist_ok=True)
        json.dump({"spans_us": dict(zip(names, spans)), "sum_us": tot, "step_us_median": med},
                  open(os.path.join(a.out, "spans.json"), "w"), indent=1)

dist.barrier()
dist.destroy_process_group()
