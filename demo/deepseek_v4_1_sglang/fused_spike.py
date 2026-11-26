"""Feasibility spike: TWO sublayers (one attention + one MoE) fused into ONE static megakernel, TP=8.

Why this file exists: with one kernel per sublayer a decode step is 80 launches, and each layer's sum_store
all-reduce needs every TP rank at the same launch. That works only if something keeps the ranks in lockstep
(SGLang does; a bare Python loop of launch_stream does not -> the exchange-set parity drifts and hc_post_combine
spins for ever). `launch_func` with num_gpus > 1 lines the GPUs up host-side and the kernel's start barrier does
the rest, so a single kernel per chunk is reliable where 80 launches are not.

The plain demos (no mHC) are enough to prove the mechanism:
    plain attention ends with sum_store writing y          (output=a("y"))
    plain MoE       starts with act_quant reading x        (input=a("x"))
so the boundary is one tensor: y of the attention is x of the MoE.

  cross-layer dependency. Two graph tensors over one buffer is not an edge: `compiler.producer_of` looks the
  writer up by guid, and a graph input has none. So the MoE's x is attached to the attention's y buffer and
  registered in `compiler.PRODUCER_ALIAS` (the mechanism moe_demo.head_nodes already uses for `x_out`), which
  is what makes the compiler see "the MoE's front waits for the attention's sum_store".

  names. Every graph tensor is prefixed with its sublayer ("s0.", "s1.") so two sublayers' "x", "y", "x_q" do
  not collide; the demos' per-layer `mpk.max_bufs = 25` is neutralised (a fused graph holds every sublayer's
  slots) and the true count is set after the build.

  python fused_spike.py --checkpoint DIR [--layer 3] [--gpus 8] [--sublayers 2] [--reps 10]
"""
import argparse
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.append(os.path.join(ROOT, "demo"))

AP = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
AP.add_argument("--checkpoint", required=True)
AP.add_argument("--layer", type=int, default=3, help="the attention/MoE layer of the backbone")
AP.add_argument("--gpus", type=int, default=8, help="the TP size (one process, launch_func lines the GPUs up)")
AP.add_argument("--sublayers", type=int, default=2, choices=(2, 4), help="2: attn + MoE; 4: two layers' worth")
AP.add_argument("--pos", type=int, default=4096)
AP.add_argument("--reps", type=int, default=10)
AP.add_argument("--out", default="/tmp/fused_spike")
a = AP.parse_args()
os.environ.setdefault("STATIC_TOKENS", "1")

from deepseek_v4_1_csa2 import demo as attn_demo           # noqa: E402
from deepseek_v4_1_csa2 import weights as attn_weights    # noqa: E402
from deepseek_v4_1_moe import demo as moe_demo             # noqa: E402
from deepseek_v4_1_moe import weights as moe_weights       # noqa: E402
from mirage.mpk import compiler                            # noqa: E402
from mirage.mpk.static_harness import Harness              # noqa: E402
from mirage.mpk.static_megakernel import StaticMegakernel   # noqa: E402

T, HID = attn_demo.T, attn_demo.HID
NG = a.gpus
devs = [torch.device("cuda", g) for g in range(NG)]
read = attn_weights.checkpoint_reader(a.checkpoint)
cfg = attn_weights.model_config(a.checkpoint)
L = a.layer
variant, ratio = attn_weights.variant_of(cfg, L), cfg["compress_ratios"][L]
csrc = L == cfg["candidate_source_layer"]
heads = 64 // NG
pos = a.pos


class FusedKernel(StaticMegakernel):
    """StaticMegakernel whose max_bufs the demos' build_graph cannot lower: a fused graph holds every sublayer's
    buffer slots, so the per-layer `mpk.max_bufs = 25` would trip _add_node's assert on the second sublayer."""

    def __init__(self, num_gpus: int = 1):
        super().__init__(num_gpus=num_gpus)
        self.__dict__["_mb"] = 4096        # during the build; the true count just before compile

    @property
    def max_bufs(self):
        return self.__dict__["_mb"]

    @max_bufs.setter
    def max_bufs(self, value):
        pass                                # the demos' per-layer setting: the fused graph decides


sms = torch.cuda.get_device_properties(0).multi_processor_count
mpk = FusedKernel(num_gpus=NG)
print(f"[spike] one kernel, {NG} GPUs x {sms} SMs, layer {L} ({variant}), {a.sublayers} sublayers", flush=True)


def prefix(name: str, tag: str) -> str:
    return f"{tag}.{name}"


def scoped(tag: str):
    """Prefix every graph tensor this sublayer declares, so the sublayers' names stay distinct."""
    old_attach, old_new = mpk.attach_input, mpk.new_tensor

    def attach(*args, **kw):
        kw["name"] = prefix(kw["name"], tag)
        return old_attach(*args, **kw)

    def new(*args, **kw):
        kw["name"] = prefix(kw["name"], tag)
        return old_new(*args, **kw)

    mpk.attach_input, mpk.new_tensor = attach, new
    return old_attach, old_new


def guid_of(name: str) -> int:
    for g, n in mpk._tensor_names.items():
        if n == name:
            return g
    raise KeyError(f"no graph tensor {name}: {sorted(v for v in mpk._tensor_names.values())}")


# ---- the sublayers' tensors, one dict per GPU ---------------------------------------------------------------
ws, tags = [], []
for s in range(a.sublayers):
    attention = s % 2 == 0
    tag = f"s{s}"
    if attention:
        rows = (pos + 2 + 255) // 256 * 256
        cap = max(((pos + 1) // max(ratio, 1) + 255) // 256 * 256, attn_demo.NC)
        per_gpu = [{**attn_weights.layer_tensors(read, cfg, L, NG, r, rows, d),
                    **attn_demo.state_tensors(variant, ratio, csrc, cap, d)} for r, d in enumerate(devs)]
        for wg in per_gpu:
            attn_demo.random_inputs(wg, variant, ratio, pos)
            attn_demo.window_tables(wg)
    else:
        per_gpu = [moe_weights.layer_tensors(read, L, NG, r, d) for r, d in enumerate(devs)]
        for wg, rt in zip(per_gpu, moe_weights.routed_tensors(read, L, NG, list(range(NG)), devs)):
            wg.update(rt)
        for r, wg in enumerate(per_gpu):   # the demo's driver makes x and y (layer_tensors holds the weights only)
            wg["y"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=devs[r])
            wg["x"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=devs[r])
    ws.append(per_gpu)
    tags.append(tag)

# ---- the fused graph ---------------------------------------------------------------------------------------
for s in range(a.sublayers):
    tag, attention = tags[s], s % 2 == 0
    if not attention:                     # the MoE reads the previous sublayer's y where it lies
        for r in range(NG):
            ws[s][r]["x"] = ws[s - 1][r]["y"]
    old_attach, old_new = scoped(tag)
    try:
        if attention:
            attn_demo.build_graph(mpk, ws[s][0], variant, ratio, csrc, heads)
        else:
            moe_demo.build_graph(mpk, ws[s][0])
    finally:
        mpk.attach_input, mpk.new_tensor = old_attach, old_new
    if not attention:                     # the edge the compiler cannot see on its own (see the module docstring)
        compiler.PRODUCER_ALIAS[guid_of(prefix("x", tag))] = guid_of(prefix("y", tags[s - 1]))
    print(f"[spike] {tag} ({'attn' if attention else 'moe'}): built, {len(mpk._bufs)} buffer slots so far", flush=True)

# ---- one plan over every node: graph order, round robin over the SMs ---------------------------------------
nodes = compiler.graph_nodes(mpk)
grids = {n.graph_idx: tuple(n.grid) for n in nodes}
lists = [[] for _ in range(sms)]
k = 0
for n in nodes:
    gx, gy, gz = grids[n.graph_idx]
    for x in range(gx):
        for y in range(gy):
            for z in range(gz):
                lists[k % sms].append((n.graph_idx, (x, y, z)))
                k += 1
plan = {"grids": grids, "lists": [copy.deepcopy(lists) for _ in range(NG)]}
print(f"[spike] {len(nodes)} nodes, {k} tasks, {k / sms:.1f} per SM (the table holds {62})", flush=True)
if k / sms > 60:
    print(f"[spike] WARNING: {k / sms:.1f} tasks per SM, the table holds 62", flush=True)

os.makedirs(a.out, exist_ok=True)
paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
nb = False
given = lambda n: n.info["out_buffer"] is not None and not n.info["exchange"] and \
    not (nb and n.info["out_buffer"](n)[1] >= 0 and not n.info["out_buffer"](n)[2])
mpk.__dict__["_mb"] = len(mpk._bufs) + 4          # the true STATIC_MAX_BUFS
bufs = [{compiler.tname(mpk, n.outputs[0]): torch.zeros(n.info["out_buffer"](n)[0], dtype=torch.uint8, device=devs[r])
         for n in nodes if given(n)} for r in range(NG)]
gpu_tensors = []
for r in range(NG):
    d = {**bufs[r]}
    for s in range(a.sublayers):
        d.update({prefix(k2, tags[s]): v for k2, v in ws[s][r].items()})
    gpu_tensors.append(d)
sk = mpk.compile_static(paths, gpu_tensors=gpu_tensors, out_dir=os.path.join(a.out, "build"),
                        extra_flags=["-DSTATIC_RESET_IN_KERNEL"])
for d in devs:
    torch.cuda.synchronize(d)

# ---- launch and time ---------------------------------------------------------------------------------------
h = Harness(sk)
h.launch(512 << 20, timeout_s=120)
print("[spike] the fused kernel returned", flush=True)
last_moe = ws[-1][0]
print(f"[spike] last sublayer y rms {last_moe['y'].float().square().mean().sqrt():.4g} "
      f"(0 would mean the chain never ran)", flush=True)
spans = []
for _ in range(a.reps):
    h.launch(512 << 20, timeout_s=120)
    spans.append(h.span_us())
spans.sort()
print(f"[spike] {a.sublayers} sublayers in ONE kernel, TP{NG}: median span {spans[len(spans) // 2]:.1f} us "
      f"(min {spans[0]:.1f})", flush=True)
sk.finalize()
