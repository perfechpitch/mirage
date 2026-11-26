"""The golden set: SGLang alone (no static layers) on a bench32k prompt, greedy, every step's token and logprob.

This is the reference the static megakernels are checked against. Two things it gives that nothing else does:
  the prefill      a real 32K context in SGLang's pools (the static chain's own harness fills its caches with
                   random bytes, so it can never be compared with the model's real output)
  the tokens       greedy decode's first N ids: DSV41_STATIC_CHECK only proves a layer's y (or x_out) against
                   SGLang's on one step; token equality over N steps proves the whole 40-layer chain end to end

Write one file per (row, tokens) and keep it: the hook run (--golden) compares against it.

  SGL_TP=8 python golden_set.py --model DIR --row 0 --tokens 32 --out /tmp/golden_row0.json
  # a second identical run must print the same ids (SGLang's own repeatability, the baseline's noise floor)

Note the memory knobs: the checkpoint's Engram tables (203 GB) live on the CPU (--engram-config), and
mem_fraction_static stays low: the static layers re-read the weights from the checkpoint (38.4 GB/GPU at TP8)
on top of SGLang's own copy.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="/opt/dlami/nvme/saboxu/data/long-code-arena/bench32k/*.jsonl")
ap.add_argument("--row", type=int, default=0, help="which row of the first (sorted) jsonl")
ap.add_argument("--id", default=None, help="instead of --row: the row's id (a prefix is enough)")
ap.add_argument("--tokens", type=int, default=32)
ap.add_argument("--rep", type=int, default=1, help="the prompt this many times (a longer context)")
ap.add_argument("--deterministic", action="store_true",
                help="SGLang's batch-invariant kernels: the baseline then does not depend on batch shape")
ap.add_argument("--graph", action="store_true", help="CUDA graphs (default: eager)")
ap.add_argument("--mem-fraction", type=float, default=0.5)
ap.add_argument("--watchdog", type=float, default=3600.0)
ap.add_argument("--engram-config", default='{"placement": "cpu"}')
ap.add_argument("--out", default=None)
a = ap.parse_args()

def main():
    files = sorted(glob.glob(a.data))
    rows = [json.loads(line) for f in files for line in open(f)]
    if a.id:
        row = next(r for r in rows if r["id"].startswith(a.id))
    else:
        row = rows[a.row]

    from transformers import AutoTokenizer                                          # noqa: E402
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    text = row["prompt"] * a.rep
    ids = tok.encode(text, add_special_tokens=False)
    print(f"row {row['id']}\n  {len(ids)} prompt tokens (the row's own count {row['prompt_tokens']}), "
          f"generating {a.tokens} greedy", flush=True)
    assert len(ids) < 32000 * a.rep + 1, "the prompt is longer than the row claims"

    import sglang as sgl                                                             # noqa: E402
    extra = {"engram_config": json.loads(a.engram_config)} if a.engram_config else {}
    if a.deterministic:
        extra["enable_deterministic_inference"] = True
    tp = int(os.environ.get("SGL_TP", "8"))
    ctx = -(-(len(ids) + a.tokens + 1024) // 1024) * 1024
    llm = sgl.Engine(model_path=a.model, tokenizer_path=a.model, trust_remote_code=True, tp_size=tp,
                     max_running_requests=1, cuda_graph_max_bs_decode=1, context_length=ctx,
                     max_total_tokens=ctx + 2048, mem_fraction_static=a.mem_fraction,
                     disable_radix_cache=True, disable_cuda_graph=not a.graph, log_level="warning",
                     watchdog_timeout=a.watchdog, **extra)
    t0 = time.perf_counter()
    out = llm.generate(input_ids=[ids], sampling_params={"max_new_tokens": a.tokens, "temperature": 0.0,
                                                         "ignore_eos": True},
                       return_logprob=True)
    dt = time.perf_counter() - t0
    llm.shutdown()

    o = out[0]
    gids = list(o.get("output_ids") or [])
    # per generated step: the chosen token's logprob (top-1's id is gids[i]; the logprob pins the logits' scale)
    lp = o.get("meta_info", {}).get("output_token_logprobs") or []
    print(f"generated {len(gids)} tokens in {dt:.1f} s ({dt / max(len(gids), 1) * 1e3:.0f} ms/token)")
    print(f"ids: {gids}")
    if lp:
        print(f"logprobs: {[round(x[0], 3) if x and x[0] is not None else None for x in lp]}")
    rec = {"row": row["id"], "prompt_tokens": len(ids), "rep": a.rep, "tokens": len(gids), "ids": gids,
           "text": o.get("text", ""), "logprobs": [x[0] if x else None for x in lp],
           "tp": tp, "deterministic": a.deterministic, "graph": a.graph, "model": a.model,
           "sglang": getattr(sgl, "__version__", None)}
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
