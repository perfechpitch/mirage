"""Chain N sublayers as N separate static kernels, ONE process, 8 GPUs, `launch_func` per launch.

The point: `Harness.launch` -> `StaticKernel.launch` -> `launch_func(flush)` lines the GPUs up host-side before every
launch (static_schedule.py: "L2 emptied on each GPU's stream first, then the GPUs lined up (more than one), then the
kernel"). One kernel per sublayer with that line-up is the cheap alternative to fusing layers: no kernel constant
has to grow (MAX_TASKS_PER_SM 64, MAX_MAPS 32, MAX_RESETS 24, MAX_BUFS, NODE_COUNTER_LINES are all sized for one
sublayer), and the exchange region's two sets stay in parity because every rank enters launch n together.

What hung in e2e_decode.py: torchrun, one process per rank, `launch_stream` -- nothing aligns the ranks, so rank r
can be at launch n+1 while rank r' is still at n. `sum_store` then writes set (n+1) % 2 while a peer polls set n % 2,
and `hc_post_combine`'s 4 CTAs never end (the watchdog dump: "SMs entered: 148/148, ended: 144").

  chain     attention -> MoE -> attention -> MoE (each its own kernel), `--steps` steps of 4 launches
  check     the chain's own numbers: every GPU's y bit-identical, y's rms not 0 (0: the chain never ran)
  time      each launch's span (us), median and max, and the sum over a step

  python chain_spike.py --checkpoint DIR [--layer 3] [--gpus 8] [--steps 20] [--reps 10]
"""
import argparse
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.append(os.path.join(ROOT, "demo"))

AP = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
AP.add_argument("--checkpoint", required=True)
AP.add_argument("--layer", type=int, default=3)
AP.add_argument("--gpus", type=int, default=8)
AP.add_argument("--steps", type=int, default=20, help="decode steps; each runs every kernel in the chain once")
AP.add_argument("--probe", action="store_true", help="time launch(0) and span_us() apart, then exit")
AP.add_argument("--reps", type=int, default=10, help="timed steps after --steps")
AP.add_argument("--pos", type=int, default=4096)
AP.add_argument("--flush-mb", type=int, default=0,
                help="L2 emptied before each launch (cold-cache timing); 0 (default): a serving step, nothing between layers")
AP.add_argument("--out", default="/tmp/chain_spike")
a = AP.parse_args()
os.environ.setdefault("STATIC_TOKENS", "1")

from deepseek_v4_1_csa2 import demo as attn_demo           # noqa: E402
from deepseek_v4_1_csa2 import weights as attn_weights    # noqa: E402
from deepseek_v4_1_moe import demo as moe_demo             # noqa: E402
from deepseek_v4_1_moe import weights as moe_weights       # noqa: E402
from mirage.mpk.static_harness import Harness              # noqa: E402

T, HID = attn_demo.T, attn_demo.HID
NG, L, pos = a.gpus, a.layer, a.pos
devs = [torch.device("cuda", g) for g in range(NG)]
read = attn_weights.checkpoint_reader(a.checkpoint)
cfg = attn_weights.model_config(a.checkpoint)
variant, ratio = attn_weights.variant_of(cfg, L), cfg["compress_ratios"][L]
csrc = L == cfg["candidate_source_layer"]
heads = 64 // NG
FLUSH = a.flush_mb << 20
FLAGS = [f for f in os.environ.get("CHAIN_FLAGS", "-DSTATIC_RESET_IN_KERNEL").split() if f]


def attn_layer(x: torch.Tensor = None):
    rows = (pos + 2 + 255) // 256 * 256
    cap = max(((pos + 1) // max(ratio, 1) + 255) // 256 * 256, attn_demo.NC)
    ws = [{**attn_weights.layer_tensors(read, cfg, L, NG, r, rows, d),
           **attn_demo.state_tensors(variant, ratio, csrc, cap, d)} for r, d in enumerate(devs)]
    for wg in ws:
        attn_demo.random_inputs(wg, variant, ratio, pos)
        attn_demo.window_tables(wg)
    if x is not None:                      # this kernel reads the previous kernel's y where it lies
        for wg, xr in zip(ws, x):
            wg["x"] = xr
    sk, bufs, _ = attn_demo.build(ws, variant, ratio, csrc, heads, os.path.join(a.out, f"attn{len(K)}"),
                                  extra_flags=FLAGS)
    return sk, ws


def moe_layer(x: torch.Tensor = None):
    ws = [moe_weights.layer_tensors(read, L, NG, r, d) for r, d in enumerate(devs)]
    for wg, rt in zip(ws, moe_weights.routed_tensors(read, L, NG, list(range(NG)), devs)):
        wg.update(rt)
    for r, wg in enumerate(ws):
        wg["y"] = torch.zeros(T, HID, dtype=torch.bfloat16, device=devs[r])
        wg["x"] = x[r] if x is not None else \
            (torch.randn(T, HID, generator=torch.Generator().manual_seed(1)) * 0.02).bfloat16().to(devs[r]).contiguous()
    sk, bufs, _ = moe_demo.build(ws, os.path.join(a.out, f"moe{len(K)}"),
                                 extra_flags=FLAGS)
    return sk, ws


K = []                                     # (name, kernel, per-GPU tensors) in launch order
layers = [("attn0", attn_layer), ("moe0", moe_layer), ("attn1", attn_layer), ("moe1", moe_layer)]
prev = None
for name, make in layers:
    sk, ws = make(prev)
    K.append((name, sk, ws))
    prev = [wg["y"] for wg in ws]
    print(f"[chain] {name}: built ({len(K)} kernels, {NG} GPUs)", flush=True)

hs = [(name, Harness(sk), ws) for name, sk, ws in K]
for name, h, _ in hs:
    print(f"[chain] {name}: serving={bool(h.layout.get('serving'))}", flush=True)


def step(timeout: float = 180.0, time_it: bool = False):
    spans = []
    for name, h, ws in hs:
        h.launch(FLUSH, timeout_s=timeout)
        if time_it:
            spans.append((name, h.span_us()))
        if os.environ.get("CHAIN_RMS", "0") == "1":   # which kernel first makes a NaN
            for nm in ("y", "x_out", "streams_out"):
                t = ws[0].get(nm)
                if torch.is_tensor(t) and t.numel():
                    f = t.float()
                    print(f"[chain]   {name}.{nm} rms {f.square().mean().sqrt().item():.4g} "
                          f"nonfinite {int((~torch.isfinite(f)).sum())}", flush=True)
    return spans


print(f"[chain] {a.steps} untimed steps, then {a.reps} timed", flush=True)
step()                                     # warm

if a.probe:
    N = 50
    for name, h, _ in hs:
        h.launch(0, timeout_s=180)
        t0 = time.perf_counter()
        for _ in range(N):
            h.sk.launch(0)                 # host enqueue only, no wait
        t1 = time.perf_counter()
        h.sk.wait()
        t2 = time.perf_counter()
        t3 = time.perf_counter()
        for _ in range(N):
            h.launch(0, timeout_s=180)     # enqueue + wait: one launch as a server makes it
        t4 = time.perf_counter()
        print(f"[chain] {name}: enqueue {(t1 - t0) / N * 1e6:6.1f} us, drain of {N} {t2 - t1:6.1f} ms, "
              f"full launch {(t4 - t3) / N * 1e6:6.1f} us, kernel span {h.span_us():6.1f} us", flush=True)
    for name, sk, _ in K:
        sk.finalize()
    raise SystemExit(0)


for s in range(a.steps):
    step()
    if (s + 1) % 5 == 0:
        print(f"[chain] step {s + 1}/{a.steps} done", flush=True)

last = hs[-1][2]
y = last[0]["y"]
same = sum(torch.equal(wg["y"].cpu().view(torch.int16), y.cpu().view(torch.int16)) for wg in last[1:])
print(f"[chain] y rms {y.float().square().mean().sqrt():.4g}, bit-identical on {same}/{NG - 1} other GPUs", flush=True)

tot, wall = [], []
per = {}
for _ in range(a.reps):
    t0 = time.perf_counter()
    spans = step(time_it=True)
    wall.append((time.perf_counter() - t0) * 1e6)
    tot.append(sum(us for _, us in spans))
    for name, us in spans:
        per.setdefault(name, []).append(us)
print(f"[chain] a step of {len(hs)} launches: sum of spans median {statistics.median(tot):.1f} us, "
      f"wall clock median {statistics.median(wall):.1f} us (max {max(wall):.1f})", flush=True)
print(f"[chain]   inter-launch overhead: {statistics.median(wall) - statistics.median(tot):.1f} us per step "
      f"({(statistics.median(wall) - statistics.median(tot)) / len(hs):.2f} us per launch)", flush=True)
for name, v in per.items():
    print(f"[chain]   {name:<6} median {statistics.median(v):6.1f} us", flush=True)

for name, sk, _ in K:
    sk.finalize()
