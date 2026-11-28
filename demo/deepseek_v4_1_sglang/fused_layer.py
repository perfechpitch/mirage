"""Fuse ONE layer's attention and its MoE into ONE static megakernel (TP=8, the model's mHC), and measure the win.

Why: with one kernel per sublayer a 40-layer decode is 80 launches, and the gap between them (CTA teardown, the next
grid's distribution, the TP start barrier) costs ~17 us a launch -- 1.24 ms of a 4.32 ms step that no kernel-level
change can reach. Fusing attn(L)+moe(L) makes it 40 launches.

How the two graphs become one (FINDINGS 12.49): the four blockers are already solved -- STATIC_MAX_BUFS/STATIC_MAX_MAPS,
the tensor-name prefix, and build_graph(bind=), which turns the MoE's x into a real dependency edge on the attention's
x_out instead of an attach_input stub. Two more things this file establishes:

  * the weights are the HARNESS's, not a demo's. fused_spike.py builds from the plain demos' layer_tensors and
    random_inputs, which invent tensors the model does not have (its oa_q is a 5120-wide placeholder; the model's
    fused tap is G*OL) and therefore produce a different graph. Here StaticAttn/StaticMoE are constructed normally and
    their own `w` dicts are what the fused graph is built on, so the graph is the one the 40-layer chain runs.
  * the plan is the two REAL hand plans concatenated per SM. A synthesized round-robin plan does not compile: each
    task's grid is coupled to its ring/column split (act_quant's NC = K / grid.y) and the task checks that itself.

The exchange region is safe without any change: a sum_store's exchange buffer is its OWN out_buffer slot
(sum_store_layer's out_buffer with exchange=True), so the attention's all-reduce and the MoE's land in different
slots of the same set -- a peer cannot overwrite one with the other.

    torchrun --nproc_per_node=8 fused_layer.py --checkpoint DIR [--layer 3] [--pos 31906] [--reps 20]

Prints, per rank 0: the fused kernel's task counts, a bitwise comparison of the one-kernel result against the
two-kernel chain (x_out, streams_out), and the two timings (2 launches vs 1 launch, same work).
"""
import argparse
import copy
import json
import os
import re
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.append(os.path.join(ROOT, "demo"))
os.environ.setdefault("STATIC_TOKENS", "1")     # (before dsv41_static_layer: its T is a module constant)

A = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
A.add_argument("--checkpoint", required=True)
A.add_argument("--layer", type=int, default=3, help="the attention/MoE layer to fuse")
A.add_argument("--pos", type=int, default=31906, help="the position the tables are built for")
A.add_argument("--reps", type=int, default=20)
A.add_argument("--warmup", type=int, default=3)
A.add_argument("--out", default="/tmp/fused_layer")
a = A.parse_args()

RANK = int(os.environ.get("RANK", "0"))
LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
assert WORLD == 8, f"TP=8: WORLD_SIZE={WORLD}"
dist.init_process_group("nccl", device_id=torch.device("cuda", LOCAL))
GROUP = dist.group.WORLD
torch.cuda.set_device(LOCAL)   # (the fused kernel's own init: cudaMemset of a region allocated on another device is
                               # cudaErrorInvalidValue, and nothing else here sets this process's current device)

import dsv41_static_layer as Lm                                    # noqa: E402
from mirage.mpk import compiler                                    # noqa: E402
from mirage.mpk.static_megakernel import StaticMegakernel          # noqa: E402
from mirage.mpk.static_schedule import generate_code               # noqa: E402

attn_demo, moe_demo = Lm.attn_demo, Lm.moe_demo
T, HID, HC = Lm.T, Lm.HID, Lm.HC
L, pos = a.layer, a.pos
dev = torch.device("cuda", LOCAL)


def log(msg: str) -> None:
    if RANK == 0:
        print(f"[fuse] {msg}", flush=True)


CK = a.checkpoint
cfg = Lm.attn_weights.model_config(CK)
variant = Lm.attn_weights.variant_of(cfg, L)
ratio = cfg["compress_ratios"][L]
csrc = L == cfg["candidate_source_layer"]
heads = 64 // WORLD
pool = Lm.ExchangePool(GROUP, dev, 32)   # (the two layers + the fused kernel)
sources = Lm.Sources(cfg)
positions = torch.full((T,), pos, dtype=torch.int32, device=dev)

# ---- the two layers exactly as the 40-layer chain builds them (their own w dicts, mHC on: the model has hc_mult = 4) ----
t0 = time.perf_counter()
# a "reuse"/"reindex" layer reads its KV/top-k from an earlier layer's tensors (Sources): those layers must be built
# here first, in order, exactly as the 40-layer chain builds them (the group is replaced together)
warm = {}
if os.environ.get("FUSE_WARM", "1") != "0":
    for l in range(L):
        v = Lm.attn_weights.variant_of(cfg, l)
        warm[l] = Lm.StaticAttn(l, CK, RANK, WORLD, dev, a.out, pos + 8, sources, False, pool, None,
                                mhc=True, stats=False, engram_kv=None, weights_from=None)
sa = Lm.StaticAttn(L, CK, RANK, WORLD, dev, a.out, pos + 8, sources, False, pool, None,
                   mhc=True, stats=False, engram_kv=None, weights_from=None)
sm = Lm.StaticMoE(L, CK, RANK, WORLD, dev, a.out, False, pool, norm_next=True, mhc=True, stats=False,
                  engram=False, gamma_name=None, engram_kv=None, head=False)
log(f"L{L} {variant}: attn+MoE built in {(time.perf_counter() - t0) / 60:.1f} min")

# the step's inputs: x (the attention's) and R (the streams); the MoE reads the attention's x_out
x = (torch.randn(T, HID, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)
R = (torch.randn(T, HC, HID, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)

# ---- the fused graph: two build_graphs on one mpk ---------------------------------------------------------------
sms = torch.cuda.get_device_properties(LOCAL).multi_processor_count - int(os.environ.get("STATIC_FREE_SMS_ATTN", 0))


class FusedMK(StaticMegakernel):
    """A StaticMegakernel whose buffer/map/counter limits have a FLOOR while the two sublayers' graphs are being built.

    dsv41_mhc_nodes sets max_bufs to 64 and then to the slots in use when it returns; called once per sublayer, that
    lowers the ceiling under the first sublayer's slots before the second one allocates. The floor holds the ceiling
    up until both graphs are in, then it is dropped so the #defines carry the exact counts (each extra slot costs
    every launch)."""

    def _floor(self, key: str, minimum: int) -> None:
        self.__dict__["_floor_" + key] = minimum

    def _get(self, key: str, dflt: int) -> int:
        return max(self.__dict__.get("_limit_" + key, 0) or 0,
                   self.__dict__.get("_floor_" + key, 0) or 0) or dflt

    def _set(self, key: str, v: int) -> None:
        self.__dict__["_limit_" + key] = v

    max_bufs = property(lambda s: s._get("bufs", 24), lambda s, v: s._set("bufs", v))
    max_maps = property(lambda s: s._get("maps", 32), lambda s, v: s._set("maps", v))
    max_counter_lines = property(lambda s: s._get("counter_lines", 16), lambda s, v: s._set("counter_lines", v))


mpk = FusedMK(num_gpus=WORLD, num_sms=sms)
mpk._floor("bufs", 96)          # the two sublayers' slots together (config.cuh's 24 is one sublayer's)
mpk._floor("maps", 96)          # (config.cuh's 32 is one sublayer's)
mpk._floor("counter_lines", 48)  # (config.cuh's 16 is one sublayer's)


def scoped(tag: str, record: dict = None):
    """Prefix every graph tensor this sublayer declares, so the two sublayers' names stay distinct.

    record: name -> DTensor for every attach_input, so a tensor a node writes through a buffer slot (hc_post_combine's
    x_out, the next sublayer's input) can be looked up afterwards."""
    old_attach, old_new = mpk.attach_input, mpk.new_tensor

    def attach(*args, **kw):
        kw["name"] = tag + kw["name"]
        t = old_attach(*args, **kw)
        if record is not None:
            record[kw["name"]] = t
        return t

    def new(*args, **kw):
        kw["name"] = tag + kw["name"]
        return old_new(*args, **kw)

    mpk.attach_input, mpk.new_tensor = attach, new
    return old_attach, old_new


captured = {}
oa, on = scoped("a.", captured)
os.environ["HC_DRY"] = os.environ.get("HC_DRY_ATTN", "0")
try:
    attn_demo.build_graph(mpk, sa.w, sa.variant, sa.ratio, csrc, heads)
finally:
    mpk.attach_input, mpk.new_tensor = oa, on
# The attention's post writes x_out through a buffer slot, so nothing in its own graph depends on it (the next layer's
# KERNEL reads it). The MoE's x now does: alias x_out to the post's output, which is how producer_of finds the writer
# (the mechanism moe_demo.head_nodes uses for its own x_out). Without this the MoE's x would be a stub with no edge.
attn_x_out = captured["a.x_out"]
compiler.PRODUCER_ALIAS[attn_x_out.guid] = captured["a.streams_out"].guid
log(f"attn graph: {len(mpk._bufs)} slots, {len(mpk._maps)} maps (x_out -> streams_out aliased)")

CONTROL = bool(os.environ.get("FUSE_NO_MOE"))   # (a control: the attention sublayer alone, through THIS builder)
if not CONTROL:
    oa, on = scoped("m.", captured)
    os.environ["HC_DRY"] = os.environ.get("HC_DRY_MOE", "1")
    # dsv41_mhc_nodes lowers max_bufs to the slots in use when it returns; the second sublayer needs the ceiling back
    mpk.max_bufs = 96
    try:
        _bind = None if os.environ.get("FUSE_NO_EDGE") else (lambda k, t: attn_x_out if k == "x" else None)
        moe_demo.build_graph(mpk, sm.w, bind=_bind)
    finally:
        mpk.attach_input, mpk.new_tensor = oa, on
# both graphs in: drop the floor and hand the #defines the exact counts. The counter lines matter: the nodes were
# built against the floor, so if the #define stayed at config.cuh's 16 a fused graph's 34th counter would write past
# G::cnt -- an illegal access, not a static assert.
for k2 in ("bufs", "maps", "counter_lines"):
    mpk.__dict__.pop("_floor_" + k2, None)
mpk.max_bufs, mpk.max_maps = len(mpk._bufs), len(mpk._maps)
mpk.max_counter_lines = mpk._counter_lines
nodes = compiler.graph_nodes(mpk)
log(f"fused graph: {len(mpk._bufs)} slots, {len(mpk._maps)} maps, {len(nodes)} nodes")

edge = compiler.producer_of(attn_x_out, nodes)     # the MoE's front must wait for this node
log(f"the MoE's x producer: {edge.name if edge is not None else 'NONE (a stub: no edge!)'}")

# ---- the plan: the two sublayers' OWN verified schedules, concatenated per SM -----------------------------------
# Not hand_plan again: it needs the layer's cap/cmp_rows (derived in the demo's build from its own arguments), and a
# wrong guess gives different grids -- and a task body bounded by them walks off its buffer (the fused build's
# hc_mix came out 32x too big before). The two layers are already built and their schedules are exactly the plans the
# 40-layer chain runs: read those, shift the MoE's graph indices by the attention's node count, and put both lists on
# the same SM. The grids come from the same place, so every buffer comes out the size its task expects.
n_attn = 0
for n in nodes:   # (attn's build ran first on an empty mpk: its indices are 0 .. n_attn-1)
    if compiler.tname(mpk, n.outputs[0]).startswith("a."):
        n_attn = max(n_attn, n.graph_idx + 1)
merged, per_sm = {}, [[] for _ in range(sms)]
OUT_A = os.path.join(a.out, f"rank{RANK}", f"attn{L}" + (f"_t{T}" if T != 8 else ""))   # (StaticAttn's own out dir)
OUT_M = os.path.join(a.out, f"rank{RANK}", f"moe{L}" + (f"_t{T}" if T != 8 else ""))
SCHEDULES = [("a.", sa, os.path.join(OUT_A, "schedules", "schedule_gpu0.json"))]
if not CONTROL:
    SCHEDULES.append(("m.", sm, os.path.join(OUT_M, "schedules", "schedule_gpu0.json")))
for tag, layer, sched in SCHEDULES:
    d = json.load(open(sched))
    # the sublayer's own graph indices are not 0..n-1 (taps and scratch nodes sit among them): map them by ORDER, and
    # check the names so a reordered graph cannot be silently mismatched
    sub = sorted(((int(k), v["name"]) for k, v in d["nodes"].items()))
    fused = [n.graph_idx for n in nodes if compiler.tname(mpk, n.outputs[0]).startswith(tag)]
    assert [nm for _, nm in sub] == [n.name for n in nodes if n.graph_idx in fused], \
        f"{tag}: the fused graph's nodes are not the built layer's (order differs)"
    nid = dict(zip([i for i, _ in sub], fused))   # the sublayer's graph index -> the fused graph's
    for i, nm in sub:
        merged[nid[i]] = tuple(d["nodes"][str(i)]["grid"])
    for sm_, q in enumerate(d["worker_task_queues"]):
        per_sm[sm_] += [(nid[d["all_tasks"][t]["node"]], tuple(d["all_tasks"][t]["pos"])) for t in q]
    log(f"{tag[:-1]}: {len(sub)} nodes, {len(d['all_tasks'])} tasks from the layer's own schedule")
plan = {"grids": merged, "lists": [copy.deepcopy(per_sm) for _ in range(WORLD)]}
k = sum(len(l) for l in per_sm)
log(f"{k} tasks, {k / sms:.1f} per SM (the table holds {62})")

FOUT = os.path.join(a.out, f"fused_rank{RANK}")   # (each rank its own: 8 nvcc runs in one dir clobber each other)
os.makedirs(FOUT, exist_ok=True)
paths, info = compiler.compile_plan(mpk, plan, os.path.join(FOUT, "schedules"), validate_grids=False)

# ---- the buffers: allocated from the kernel's OWN declaration -----------------------------------------------
# The generated layer.cu names every buffer of every GPU ("buf.N", "<kind> <name> <bytes> <reset> <by_reader>"). A
# fused graph gets two buffer SLOTS under one tensor name more readily than one sublayer does (a GEMM's K-parts slot
# and its [T, N] view), so a name-keyed dict built from the nodes can hand the kernel a buffer of the wrong size --
# and its init memsets that size into it. The declaration is what the kernel will do: allocate from it, not from the
# graph's names.
os.makedirs(os.path.join(FOUT, "build"), exist_ok=True)
CU = os.path.join(FOUT, "build", "layer.cu")
with open(CU, "w") as f:
    f.write(generate_code(mpk, [json.load(open(p)) for p in paths], False))
decl = {}
for m in re.finditer(r'"buf\.\d+", "(out|tensor|exchange|scratch) ([^" ]+) (\d+)', open(CU).read()):
    kind, nm, nb = m.group(1), m.group(2), int(m.group(3))
    if kind in ("out", "tensor"):
        decl[nm] = max(decl.get(nm, 0), nb)
weights = {**{f"a.{k2}": v for k2, v in sa.w.items()}, **{f"m.{k2}": v for k2, v in sm.w.items()}}
bufs = {nm: torch.zeros(nb, dtype=torch.uint8, device=dev) for nm, nb in decl.items()
        if nm not in weights or not nm.startswith(("a.", "m."))}
bufs_keep = list(bufs.values())
missing = sorted(set(decl) - set(weights) - set(bufs))
if missing:
    log(f"MISSING tensors ({len(missing)}): {missing}")
    sys.exit(1)
log(f"{len(bufs)} buffers from the kernel's declaration, {len(weights)} tensors from the two layers' w")
gpu_tensors = [{**weights, **bufs}]
flags = ["-DSTATIC_RESET_IN_KERNEL"] + os.environ.get("DSV41_STATIC_FLAGS", "").split()
Lm.tpb_flag(flags, WORLD, False, "the fused build")
fpool = Lm.ExchangePool(GROUP, dev, 2)   # (kept referenced: the region must outlive the kernel; its own slot size is
                                         # the fused kernel's, which is bigger than one sublayer's)
fsk = mpk.compile_static(paths, gpu_tensors=gpu_tensors, out_dir=os.path.join(FOUT, "build"), extra_flags=flags,
                         device=LOCAL, rank=RANK, exchange=fpool.take(mpk.exchange_region_bytes()))
torch.cuda.synchronize(dev)
log("fused kernel compiled and loaded")

# ---- the correctness gate: the one-kernel result against the two-kernel chain, as int bits ----------------------
Lm.set_launch_stream()
sa.run(x, positions, R=R)
sa.sk.wait()
torch.cuda.synchronize(dev)
log("the attention kernel ran")
sm.run(sa.w["x_out"], R=R)
sm.sk.wait()
torch.cuda.synchronize(dev)
log("the MoE kernel ran")
get = {"a.x_out": lambda: sa.w["x_out"], "m.x_out": lambda: sm.w["x_out"], "m.streams_out": lambda: sm.w["streams_out"]}
ref = {nm: f().clone() for nm, f in get.items()}


def bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


for _nm, _t in (("a.streams", R), ("m.streams", R), ("a.x", x)):
    if _nm in decl:   # (the control's graph has no MoE)
        fsk.rebind(_nm, _t.data_ptr())
REPEATS = 5   # (a step is many launches: the in-kernel reset has to make every one of them the same)
for it in range(REPEATS):
    fsk.launch_stream(Lm._STREAM[0], 0)
    torch.cuda.synchronize(dev)
    for nm, f in get.items():
        same = int((bits(f()) == bits(ref[nm])).sum())
        if it == 0 or same != ref[nm].numel():
            log(f"  launch {it + 1} {nm}: {same}/{ref[nm].numel()} bit-identical to the two-kernel chain"
                f"{'' if same == ref[nm].numel() else '  <-- DIFFERS'}")
for nm in ("a.x_out", "m.x_out"):
    t = get[nm]()
    log(f"{nm} rms {t.float().square().mean().sqrt():.5g}")

# ---- timing: 2 launches vs 1, same work -------------------------------------------------------------------------
def timeit(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    e0, e1, out = torch.cuda.Event(True), torch.cuda.Event(True), []
    for _ in range(reps):
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        out.append(e0.elapsed_time(e1) * 1e3)
    out.sort()
    return out[len(out) // 2], out[0]


def chain():
    sa.sk.launch_stream(Lm._STREAM[0], 0)
    sm.sk.launch_stream(Lm._STREAM[0], 0)


def fused():
    fsk.launch_stream(Lm._STREAM[0], 0)


c_med, c_min = timeit(chain, a.reps, a.warmup)
f_med, f_min = timeit(fused, a.reps, a.warmup)
log(f"two launches (attn + moe): median {c_med:.1f} us (min {c_min:.1f})")
log(f"one fused launch        : median {f_med:.1f} us (min {f_min:.1f})")
log(f"-> {(c_med - f_med):+.1f} us a layer (one launch instead of two), x40 = "
    f"{-40 * (c_med - f_med) / 1e3:+.2f} ms on a 4.32 ms step")
fsk.finalize()
dist.destroy_process_group()
