"""L2: one bench32k prompt through SGLang with the static layers CHECKED against SGLang's own layers.

Why this shape: the static chain's own harness (e2e_decode.py) fills its KV caches with random bytes, so its
output can never be compared with the model's. Here SGLang runs the real 32K prefill into its own pools, the
static layers read those pools (DSV41_STATIC_POOL=1), and with DSV41_STATIC_CHECK=1 each layer runs SGLang's
kernel first and the static kernel on the same input: rank 0 prints y relRMS per sublayer per step. The first
layer whose relRMS jumps is the one that is wrong, which is exactly what the end-to-end number cannot tell you.

Scope of the first run (fewest moving parts):
  eager (no CUDA graphs)          -- a graph would capture the static launches and hide the ordering
  DSV41_STATIC_AR=0               -- check mode needs each rank's own partial y (the hook asserts this)
  DSV41_STATIC_MHC=0              -- mHC out of the kernels: compare the plain sublayers first
  DSV41_STATIC_CHECK_STEPS=1      -- one printed line per sublayer per step
  --tokens 1                      -- the prefill and one decode step; the relRMS is what we are after

Run it (the `boot/` dir on PYTHONPATH is what loads dsv41_static_hook.py):
  PATH=.../venvs/sglang/bin:$PATH PYTHONPATH=$PWD/boot DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all \
    DSV41_STATIC_CHECK=1 DSV41_STATIC_AR=0 DSV41_STATIC_MHC=0 DSV41_STATIC_MAX_POS=33000 \
    SGL_TP=8 python l2_check.py --model DIR --row 0 --tokens 1
"""
import argparse
import glob
import json
import os
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/opt/dlami/nvme/saboxu/data/long-code-arena/bench32k/*.jsonl")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=10,
                    help="decode steps to generate (10: the trace hook's default step count)")
    ap.add_argument("--batch", type=int, default=1,
                    help="prompts of the same length in flight. The static layers are built for T rows (8 by "
                         "dsk41_static_layer's default), and check mode refuses any other row count: the per-layer "
                         "comparison runs on a decode step of exactly T rows, so the batch must equal T")
    ap.add_argument("--mem-fraction", type=float, default=0.40,
                    help="SGLang's share: the static layers re-read the weights from the checkpoint on top of it")
    ap.add_argument("--watchdog", type=float, default=3600.0)
    ap.add_argument("--drop-last", type=int, default=0,
                    help="prefill prompt[:-N] instead of the whole prompt (dump_prefill.py's --drop-last). 1 makes this "
                         "run's first decode step the one at position len(prompt) - 1, which is the step a prefill dump "
                         "taken with --drop-last 1 replays -- without it the two are a position apart and their "
                         "per-layer values are different tokens'")
    ap.add_argument("--engram-config", default='{"placement": "cpu"}')
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(a.data))
    rows = [json.loads(line) for f in files for line in open(f)]
    row = rows[a.row]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ids = tok.encode(row["prompt"], add_special_tokens=False)
    if a.drop_last:
        ids = ids[:-a.drop_last]
    print(f"row {row['id']}\n  {len(ids)} prompt tokens, generating {a.tokens}", flush=True)

    import sglang as sgl
    # DSV41_TRACE: reading values back, not serving. No cuda graph (above) and no flashinfer autotune, whose warmup is
    # a dummy forward at the decode batch shape -- a second model path with nothing to do with the step being traced,
    # and the one that faults (an HcState reaching hc_pre on the fused cross-layer mHC path) without producing a line.
    trace = (os.environ.get("DSV41_TRACE") or "") not in ("", "0")
    extra = {"engram_config": json.loads(a.engram_config)} if a.engram_config else {}
    if trace:
        extra["disable_flashinfer_autotune"] = True
    tp = int(os.environ.get("SGL_TP", "8"))
    ctx = -(-(len(ids) + a.tokens + 1024) // 1024) * 1024
    llm = sgl.Engine(model_path=a.model, tokenizer_path=a.model, trust_remote_code=True, tp_size=tp,
                     max_running_requests=a.batch, cuda_graph_max_bs_decode=a.batch, context_length=ctx,
                     max_total_tokens=ctx * a.batch + 4096, mem_fraction_static=a.mem_fraction,
                     disable_radix_cache=True, disable_cuda_graph=True, log_level="warning",
                     watchdog_timeout=a.watchdog, **extra)
    t0 = time.perf_counter()
    # the same prompt `batch` times: identical rows keep the check honest (every row
    # must produce the same per-layer numbers) while reaching the layer's T
    out = llm.generate(input_ids=[ids] * a.batch,
                       sampling_params={"max_new_tokens": a.tokens, "temperature": 0.0, "ignore_eos": True},
                       return_logprob=True)
    dt = time.perf_counter() - t0
    llm.shutdown()
    o = out[0]
    gids = list(o.get("output_ids") or [])
    print(f"generated {len(gids)} tokens in {dt:.1f} s; ids {gids}", flush=True)
    # a check that compared nothing must not read as a pass: the hook writes its printed-comparison
    # count per rank on exit, and a run whose guards never held has to fail here, not look quiet
    # A trace run compares nothing by design (stock SGLang, no static side), so the guard does not apply to it --
    # and the counter files are removed first, because reading a previous run's would be the same false pass.
    n = 0
    if trace:
        for p in glob.glob("/tmp/dsv41_checks_rank*.txt"):
            try:
                os.remove(p)
            except OSError:
                pass
    else:
        for p in glob.glob("/tmp/dsv41_checks_rank*.txt"):
            try:
                n = max(n, int(open(p).read().strip() or 0))
            except (OSError, ValueError):
                pass
    same = len({tuple(o.get("output_ids") or []) for o in out}) == 1
    print(f"  check: {n} sublayer comparisons printed on rank 0; the {a.batch} rows decoded identically: {same}",
          flush=True)
    if n == 0 and not trace:
        sys.exit("FAILED: no sublayer was compared (see the 'NOT checked' lines for why)")
    if a.out:
        json.dump({"row": row["id"], "prompt_tokens": len(ids), "ids": gids}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
