"""Teacher-forced golden: pin the token sequence, keep only the distributions.

The greedy golden (golden_all.py) turned out not to be a usable reference on these
prompts.  Measured on row 0, five reruns of SGLang alone:

  * the next-token distribution is nearly flat (step 0's best candidate sits at
    logprob -1.85, p=0.16), and every one of the 32 steps has at least one EXACT
    tie among its top-k -- two different token ids with bit-identical logprob,
    which is what bf16 logits give you on a 288k vocab;
  * so the argmax is a coin flip wherever the tie reaches rank 0.  Two runs forked
    at step 6 (3445 and 344, tied at -1.5747376680374146), and because decode is
    autoregressive the two branches then had nothing in common from step 7 on
    (top-5 overlap 0.00, logprob error 5-15).

Nothing about that is a kernel bug, but it makes token equality a criterion the
engine itself cannot meet.  The fix is to stop asking the engine to choose: feed it
prompt + a FIXED continuation and read the logprobs of the continuation positions.
Each position is then a distribution conditioned on a prefix we control, so a tie
is irrelevant -- both tied candidates are simply reported with the same value, and
the comparison (compare_golden.py) is well defined on every step.

That is also exactly the contract the static chain will be verified against: prefill
the prompt, then for each of the N steps force the golden token and compare the
distribution before feeding it.

What a row costs: one 32K prefill, no decode (max_new_tokens=1 with the continuation
already in the input).  It also means the reference cannot drift with the engine's
tie-breaking, so ONE run is the reference -- the repeat run is only a noise floor
check.

  python golden_forced.py --model DIR --rows 0-0 --force-from /tmp/topk_1.jsonl \
      --steps 32 --out /tmp/golden_forced.jsonl
  # no --force-from: the continuation is the row's own greedy run (one engine call,
  # then the forced pass), which keeps the tokens in-distribution for the model
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
    ap.add_argument("--rows", default=None, help="a range of row indices, e.g. 0-9")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--steps", type=int, default=32, help="continuation length to condition on")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--force-from", default=None,
                    help="a golden file whose 'ids' supply the continuation (so a rerun conditions on the "
                         "same tokens); without it each row is sampled greedily first")
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument("--watchdog", type=float, default=3600.0)
    ap.add_argument("--engram-config", default='{"placement": "gpu"}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.data))
    rows = [json.loads(line) for f in files for line in open(f)]
    if a.subset:
        rows = [r for r in rows if a.subset in r["subset"]]
    if a.rows:
        lo, hi = (int(x) for x in a.rows.split("-"))
        rows = rows[lo:hi + 1]

    forced = {}
    if a.force_from:
        for line in open(a.force_from):
            try:
                r = json.loads(line)
                if r.get("ids"):
                    forced[r["row"]] = r["ids"][:a.steps]
            except Exception:
                pass
        print(f"{len(forced)} continuations from {a.force_from}", flush=True)

    todo = list(rows)  # --force-from only pins the tokens; every row still runs
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    encoded = {r["id"]: tok.encode(r["prompt"], add_special_tokens=False) for r in todo}
    longest = max(len(v) for v in encoded.values())
    ctx = -(-(longest + a.steps + 1024) // 1024) * 1024
    print(f"longest prompt {longest} tokens; context_length {ctx}", flush=True)

    import sglang as sgl
    extra = {"engram_config": json.loads(a.engram_config)} if a.engram_config else {}
    tp = int(os.environ.get("SGL_TP", "8"))
    t0 = time.perf_counter()
    llm = sgl.Engine(model_path=a.model, tokenizer_path=a.model, trust_remote_code=True, tp_size=tp,
                     max_running_requests=1, cuda_graph_max_bs_decode=1,
                     context_length=ctx, max_total_tokens=ctx + 4096,
                     mem_fraction_static=a.mem_fraction,
                     disable_radix_cache=True, disable_cuda_graph=True, log_level="warning",
                     watchdog_timeout=a.watchdog, **extra)
    print(f"engine up in {time.perf_counter() - t0:.0f} s", flush=True)

    with open(a.out, "a") as fh:
        for r in todo:
            base = encoded[r["id"]]
            cont = forced.get(r["id"])
            dt0 = time.perf_counter()
            if cont is None:
                # one greedy call to pick the continuation; its ids are what gets pinned below
                g = llm.generate(input_ids=[base], sampling_params={"max_new_tokens": a.steps,
                                                                    "temperature": 0.0,
                                                                    "ignore_eos": True})
                cont = list(g[0]["output_ids"])
            # teacher-forced: the continuation is part of the input, so every position's
            # distribution is conditioned on a prefix we fixed, and no argmax is involved.
            # logprob_start_len is where logprob computation STARTS: the position just
            # before the continuation is the first one whose "next token" is a continuation
            # token, so only ~len(cont) positions are computed (the 32K prompt's are not).
            ids = base + cont
            out = llm.generate(input_ids=[ids],
                               sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                               return_logprob=True, logprob_start_len=len(base) - 1,
                               top_logprobs_num=a.topk)
            mi = out[0].get("meta_info", {}) or {}
            # input logprobs are aligned to the input positions that have a previous token
            lps = mi.get("input_token_logprobs") or []
            tks = mi.get("input_top_logprobs") or []
            steps = []
            for tri, step in zip(lps, tks):
                if not step:
                    continue
                steps.append([[float(t[0]), int(t[1])] for t in step])
            # keep the last len(cont) distributions: position i's value is the
            # probability of token i+1, so the continuation's own logprobs are the tail
            steps = steps[-len(cont):]
            rec = {"row": r["id"], "subset": r["subset"], "prompt_tokens": len(base),
                   "tokens": len(cont), "ids": cont, "topk": steps, "topk_n": a.topk,
                   "logprobs": [s[0][0] for s in steps] if steps else [],
                   "teacher_forced": True, "seconds": round(time.perf_counter() - dt0, 1), "tp": tp}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{r['id'][:48]}] {len(cont)} steps, first id {cont[0]}, "
                  f"{time.perf_counter() - dt0:.0f} s", flush=True)
    llm.shutdown()


if __name__ == "__main__":
    main()
