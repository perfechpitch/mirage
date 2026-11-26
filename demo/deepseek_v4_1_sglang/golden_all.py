"""The golden set over ALL of bench32k (163 rows, ~32K prompt tokens each), SGLang alone, greedy.

One engine for every row: loading the weights takes ~2 min, the rows are 163, so one load and a loop.
Each row is appended to the output file as soon as it is done, and a rerun skips the rows already there
(--resume is the default), so an interrupted run costs only the row it was on.

What it writes, per row: the greedy ids and their logprobs (the reference the static chain is checked
against), the prompt length, and the wall time. Two runs of the same row must print the same ids; that
repeatability is the baseline's noise floor, and any static-vs-golden mismatch is then the kernel's.

  PATH=venvs/sglang/bin:$PATH SGL_TP=8 python golden_all.py --model DIR --tokens 32 \
      --out /tmp/golden_bench32k.jsonl
  # then, a subset:  --rows 0-9   (a range) or --subset small_context
"""
import argparse
import glob
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/opt/dlami/nvme/saboxu/data/long-code-arena/bench32k/*.jsonl")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--concurrent", type=int, default=8,
                    help="rows in flight at once: a row's 32K prefill leaves the SMs idle while its KV pages "
                         "are written, so several rows fill the machine; the cap is the KV pool, not the SMs "
                         "(~0.69 GB per 32K row on TP8, ~63 GB of pool)")
    ap.add_argument("--rows", default=None, help="a range of row indices, e.g. 0-9 (default: every row)")
    ap.add_argument("--subset", default=None, help="only rows of this subset (a substring of the subset field)")
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument("--watchdog", type=float, default=3600.0)
    ap.add_argument("--deterministic", action="store_true",
                    help="--enable-deterministic-inference: batch-invariant kernels.  Without it two runs of "
                         "the same row drift (logprobs by ~0.1 and the tokens past the ~5th), so the golden "
                         "set would carry the engine's noise into every comparison")
    ap.add_argument("--det-env", action="store_true",
                    help="export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 but do NOT pass "
                         "--enable-deterministic-inference: the env alone flips the numeric kernels (qk-norm, "
                         "1-stage all-reduce, deterministic collectives), while the flag additionally swaps the "
                         "sampling backend to pytorch and forces radix/NCCL settings -- and the flag's full set "
                         "degenerated this model's distribution (probs ~0.9999), so the two are bisected apart")
    ap.add_argument("--engram-config", default='{"placement": "cpu"}')
    ap.add_argument("--topk", type=int, default=32,
                    help="top_logprobs_num: how many candidates to keep per step.  The token id alone is not "
                         "a usable reference on these prompts (near-flat distributions, ~0.13 run-to-run "
                         "noise), so the distribution is what gets stored and compared")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.det_env:
        os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"
        print("SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1 (env only, no server flag)", flush=True)

    files = sorted(glob.glob(a.data))
    rows = [json.loads(line) for f in files for line in open(f)]
    if a.subset:
        rows = [r for r in rows if a.subset in r["subset"]]
    if a.rows:
        lo, hi = (int(x) for x in a.rows.split("-"))
        rows = rows[lo:hi + 1]

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["row"])
            except Exception:
                pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows selected, {len(done)} already in {a.out}, {len(todo)} to run", flush=True)
    if not todo:
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    # every prompt is ~32K: one context long enough for the longest of them plus the generated tokens
    encoded = {r["id"]: tok.encode(r["prompt"], add_special_tokens=False) for r in todo}
    longest = max(len(v) for v in encoded.values())
    ctx = -(-(longest + a.tokens + 1024) // 1024) * 1024
    print(f"longest prompt {longest} tokens; context_length {ctx}; {a.concurrent} rows concurrent", flush=True)

    import sglang as sgl
    extra = {"engram_config": json.loads(a.engram_config)} if a.engram_config else {}
    tp = int(os.environ.get("SGL_TP", "8"))
    t_load = time.perf_counter()
    llm = sgl.Engine(model_path=a.model, tokenizer_path=a.model, trust_remote_code=True, tp_size=tp,
                     max_running_requests=a.concurrent,
                     cuda_graph_max_bs_decode=a.concurrent,
                     context_length=ctx,
                     # the KV pool has to hold every row in flight at once, not one
                     max_total_tokens=ctx * a.concurrent + 4096,
                     mem_fraction_static=a.mem_fraction,
                     disable_radix_cache=True, disable_cuda_graph=True, log_level="warning",
                     watchdog_timeout=a.watchdog,
                     enable_deterministic_inference=a.deterministic, **extra)
    print(f"engine up in {time.perf_counter() - t_load:.0f} s", flush=True)

    # SGLang's generate takes a batch: every row of the batch is admitted and scheduled together, which is its
    # own concurrency (a thread pool does not work: generate runs an asyncio loop and refuses a second one)
    tp = int(os.environ.get("SGL_TP", "8"))
    chunk = a.concurrent
    t_start = time.perf_counter()
    done_n = 0
    with open(a.out, "a") as fh:
        for lo in range(0, len(todo), chunk):
            batch = todo[lo:lo + chunk]
            bt = time.perf_counter()
            try:
                out = llm.generate(input_ids=[encoded[r["id"]] for r in batch],
                                   sampling_params={"max_new_tokens": a.tokens, "temperature": 0.0,
                                                    "ignore_eos": True},
                                   return_logprob=True, top_logprobs_num=a.topk)
            except Exception as e:
                print(f"batch {lo // chunk} FAILED: {type(e).__name__}: {e}", flush=True)
                for r in batch:
                    fh.write(json.dumps({"row": r["id"], "subset": r["subset"], "error": str(e)}) + "\n")
                fh.flush()
                done_n += len(batch)
                continue
            dt = time.perf_counter() - bt
            for r, o in zip(batch, out):
                gids = list(o.get("output_ids") or [])
                mi = o.get("meta_info", {}) or {}
                lp = mi.get("output_token_logprobs") or []
                # The next-token distributions of these 32K code prompts are nearly flat (the step-0
                # top token is at logprob -2.2, p=0.11) and the engine's own run-to-run noise is
                # ~0.13, so the greedy id flips whenever two candidates sit inside that band -- two
                # reruns forked at step 6 (3445/-1.56 vs 344/-1.69) and reconverged at step 8.  A
                # golden set keyed on the id alone would therefore demand bit-exactness the engine
                # cannot give.  The top-k distribution is the part that IS stable, so store it and
                # compare against a tolerance (compare_golden.py) instead of the argmax.
                topk = [[[float(t[0]), int(t[1])] for t in step]
                        for step in (mi.get("output_top_logprobs") or []) if step]
                rec = {"row": r["id"], "subset": r["subset"], "prompt_tokens": len(encoded[r["id"]]),
                       "tokens": len(gids), "ids": gids, "logprobs": [x[0] if x else None for x in lp],
                       "topk": topk, "topk_n": a.topk,
                       "seconds": round(dt, 1), "tp": tp, "deterministic": a.deterministic}
                fh.write(json.dumps(rec) + "\n")
                done_n += 1
            fh.flush()
            el = time.perf_counter() - t_start
            eta = el / done_n * (len(todo) - done_n) if done_n else 0
            print(f"[{done_n}/{len(todo)}] batch of {len(batch)} in {dt:.0f} s "
                  f"({dt / len(batch):.1f} s/row); elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min", flush=True)
    llm.shutdown()
    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min: {a.out}", flush=True)


if __name__ == "__main__":
    main()
