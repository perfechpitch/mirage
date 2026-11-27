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
assert T == int(os.environ["STATIC_TOKENS"])
cfg = Lm.attn_weights.model_config(a.checkpoint)
variants = [Lm.attn_weights.variant_of(cfg, L) for L in range(40)]
log(f"{a.layers} layers: " + " ".join(f"{L}:{variants[L]}" for L in range(a.layers)))

# ---- the step's input: the prompt's last token through the embedding table ------------------------------------
tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
ids = tok.encode(row["prompt"], add_special_tokens=False)
last = ids[-1]
idx = json.load(open(os.path.join(a.checkpoint, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(a.checkpoint, idx["embed.weight"]), "pt") as h:
    x0 = h.get_tensor("embed.weight")[last].to(torch.bfloat16).contiguous().view(1, -1)   # [1, HID]
log(f"tokenized {len(ids)} ids (the row's own count {row['prompt_tokens']}); x = token {last}'s embedding, "
    f"norm {x0.float().norm():.3f}")

dev = torch.device("cuda", LOCAL)
sources = Lm.Sources(cfg)
positions = torch.full((T,), pos, dtype=torch.int32, device=dev)

# one region per layer kernel (attention, MoE) that has an exchange; the first take sets the slot size
pool = Lm.ExchangePool(GROUP, dev, 2 * a.layers + 4)
pool.take(int(a.pool_mb * 2**20))

attn, moe = [], []
t0 = time.perf_counter()
for L in range(a.layers):
    sa = Lm.StaticAttn(L, a.checkpoint, RANK, WORLD, dev, a.out, pos + 8, sources, bool(a.pdl), pool, None,
                       mhc=MHC, stats=False, engram_kv=None, weights_from=None)
    attn.append(sa)
    # the last layer's MoE also computes the head (norm.weight): its post makes the head's input
    head = L == a.layers - 1 and a.layers == 40
    sm = Lm.StaticMoE(L, a.checkpoint, RANK, WORLD, dev, a.out, bool(a.pdl), pool, norm_next=True,
                      mhc=MHC, stats=False, engram=False, gamma_name="norm.weight" if head else None,
                      engram_kv=None, head=head)
    moe.append(sm)
    log(f"L{L} {variants[L]}: attn+MoE built, {(time.perf_counter() - t0) / 60:.1f} min, "
        f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free")
log(f"built {a.layers} layers in {(time.perf_counter() - t0) / 60:.1f} min on rank {RANK}")

# no prefill runs here: fill the caches as the layer demos do (random, valid at `pos`)
for sa in attn:
    Lm.attn_demo.random_inputs(sa.w, sa.variant, sa.ratio, pos)
    Lm.attn_demo.window_tables(sa.w)
    # random_inputs masks window_cache's e4m3 bytes with 0xFE but NOT compress_cache's, so ~2/256 of the
    # compressed KV bytes are 0x7F / 0xFF -- NaN in e4m3. One layer never notices (its y is discarded); a
    # 40-layer chain pushes them through every mHC residual until x is all NaN by L3 (ratio 2's first layer).
    # Mask them here: the same random bytes, only the NaN encodings removed.
    cc = sa.w.get("compress_cache")
    if cc is not None:                        # rows are D//2 value bytes then D//16 scale bytes
        cc[..., :Lm.attn_demo.D // 2].bitwise_and_(0xFE)
    ik = sa.w.get("index_k")                  # rows are 64 value bytes then 4 scale bytes
    if ik is not None:
        ik[..., :64].bitwise_and_(0xFE)
    # The serving build (STATIC_RESET_IN_KERNEL) memsets the buffers whose reader re-arms them (by_reader) once, at
    # build: everything above wrote into them afterwards, so the first launch would consume those bytes as if a
    # producer had published them. Re-arm them here (the host reset path does exactly this before every launch).
    try:
        sa.sk.rebind("@prime", 0)
    except Exception as e:
        log(f"  @prime not available: {type(e).__name__}")
torch.cuda.synchronize()
# Every rank builds its own copy, and the build times differ (nvcc per rank). A static layer's sum_store collects
# all ranks' parts, so a rank that launches first spins inside the kernel until the last rank launches: barrier here.
dist.barrier()
zeros_R = torch.zeros(T, HC, HID, dtype=torch.bfloat16, device=dev)
x = x0.to(dev).expand(T, -1).contiguous()   # T = 1: one row; T = 8 (the design size): the same row eight times
p = positions
DEBUG = os.environ.get("DSV41_E2E_DEBUG", "0") != "0"


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


def step():
    """One decode step: the layers' launches in order (the last MoE's kernel also runs the head).

    With mHC in the kernels a sublayer does not write y: the attention's sum_store only sends its part (output=None)
    and the kernel's post writes x_out, the normed input of the next sublayer (gamma = the next norm's weight). So
    the chain is x_out -> the next kernel's x, and streams_out -> the next kernel's R."""
    xx, R = x, zeros_R                       # R: [T, HC, HID] (bind_streams' shape; the layers store it 2D)
    if DEBUG:                                # every rank, so a rank that never reaches the launch is visible
        print(f"[e2e] rank {RANK} entering the step: rows {xx.shape[0]}, positions {int(p[0])}", flush=True)
    for i in range(a.layers):
        if DEBUG:
            print(f"[e2e] rank {RANK} launch attn L{i} ({variants[i]})", flush=True)
        attn[i].run(xx, p, R)
        _wait(attn[i], f"attn L{i}")
        _sync()                              # the ranks in lockstep before the next kernel (see _sync)
        R = attn[i].w["streams_out"].view(T, HC, HID)
        xx = attn[i].w["x_out"]              # the post's next input = this MoE's x
        moe[i].run(xx, R)
        _wait(moe[i], f"moe L{i}")
        _sync()
        R = moe[i].w["streams_out"].view(T, HC, HID)
        xx = moe[i].w["x_out"]               # (norm_next) the next attention's x; the last layer's: the head's input
        if a.sync_every and (i + 1) % a.sync_every == 0 and i + 1 < a.layers:
            _sync()


def _sync():
    """The ranks back in lockstep. A static layer's sum_store collects every rank's part through the exchange regions,
    so all ranks must run the same kernel: their streams are independent, so without this they drift apart."""
    # torch.cuda.synchronize() waits for THIS process's GPU only: it leaves the ranks free to be at different
    # launches, and sum_store then writes exchange set (n+1)%2 while a peer polls set n%2 (hc_post_combine's 4
    # CTAs never end). The barrier is what launch_func does host-side for a one-process run.
    # With -DSTATIC_TP_BARRIER the kernel does this itself (core.cuh kernel_begin, before the resets), so the host
    # round trip is only paid when the build has no kernel barrier -- the timed path stays pure otherwise.
    if NOSYNC or TPB:
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

for _ in range(a.warmup):
    step()
torch.cuda.synchronize()
dist.barrier()
start, end = torch.cuda.Event(True), torch.cuda.Event(True)
evs = []
for _ in range(a.reps):
    torch.empty(512 << 20, dtype=torch.uint8, device=dev).zero_()   # empty L2, as the layer demos do
    torch.cuda.synchronize()
    start.record()
    step()
    end.record()
    torch.cuda.synchronize()
    evs.append(start.elapsed_time(end) * 1e3)
evs.sort()
med = evs[len(evs) // 2]
log(f"{a.layers} layers at position {pos}: median step {med:.1f} us = {med / 1000:.2f} ms "
    f"(p10 {evs[len(evs) // 10]:.1f}, p90 {evs[9 * len(evs) // 10]:.1f}); {1e6 / med:.0f} tok/s")
if RANK == 0:
    os.makedirs(a.out, exist_ok=True)
    json.dump({"row": row["id"], "pos": pos, "layers": a.layers, "tp": WORLD, "reps": a.reps,
               "step_us_median": med, "step_us_p10": evs[len(evs) // 10], "step_us_p90": evs[9 * len(evs) // 10],
               "tok_s": 1e6 / med, "variants": variants[:a.layers]},
              open(os.path.join(a.out, "e2e_decode.json"), "w"), indent=1)
    log(f"wrote {os.path.join(a.out, 'e2e_decode.json')}")

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
