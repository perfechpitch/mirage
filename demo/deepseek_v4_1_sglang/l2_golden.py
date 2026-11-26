"""Full-model decode vs the golden set, in the golden set's own format.

What this closes: l2_check compares sublayers (one layer at a time, SGLang's kernel also
run), which locates a fault but does not answer "does the static chain produce the model's
tokens".  This runs ONE row of bench32k end to end -- all 40 layers static, real weights,
real 32K prefill -- for N greedy steps, records every step's top-32 distribution exactly as
golden_bench32k.jsonl does, and leaves a file compare_golden.py reads unchanged:

  DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_POOL=1 python l2_golden.py \
      --model DIR --row 0 --tokens 32 --out /tmp/static_row0.jsonl
  python compare_golden.py --golden golden_bench32k.jsonl --other /tmp/static_row0.jsonl

The baseline is the same script with the hook off (no DSV41_STATIC_* set): the engine alone
flips its argmax at bf16 ties (golden_forced's docstring measures it), so the static chain is
correct at least when its per-step error and top-5 overlap match that baseline's.
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
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=0.40)
    ap.add_argument("--watchdog", type=float, default=3600.0)
    ap.add_argument("--engram-config", default='{"placement": "gpu"}')
    ap.add_argument("--tag", default="static")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.data))
    rows = [json.loads(line) for f in files for line in open(f)]
    row = rows[a.row]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ids = tok.encode(row["prompt"], add_special_tokens=False)
    print(f"row {row['id']}\n  {len(ids)} prompt tokens, {a.tokens} greedy steps, batch {a.batch}", flush=True)

    import sglang as sgl
    extra = {"engram_config": json.loads(a.engram_config)} if a.engram_config else {}
    tp = int(os.environ.get("SGL_TP", "8"))
    ctx = -(-(len(ids) + a.tokens + 1024) // 1024) * 1024
    t0 = time.perf_counter()
    llm = sgl.Engine(model_path=a.model, tokenizer_path=a.model, trust_remote_code=True, tp_size=tp,
                     max_running_requests=a.batch, cuda_graph_max_bs_decode=a.batch, context_length=ctx,
                     max_total_tokens=ctx * a.batch + 4096, mem_fraction_static=a.mem_fraction,
                     disable_radix_cache=True, disable_cuda_graph=True, log_level="warning",
                     watchdog_timeout=a.watchdog, **extra)
    print(f"engine up in {time.perf_counter() - t0:.0f} s", flush=True)

    dt = time.perf_counter()
    out = llm.generate(input_ids=[ids] * a.batch,
                       sampling_params={"max_new_tokens": a.tokens, "temperature": 0.0, "ignore_eos": True},
                       return_logprob=True, top_logprobs_num=a.topk)
    secs = time.perf_counter() - dt
    llm.shutdown()

    o = out[0]
    mi = o.get("meta_info", {}) or {}
    steps = []
    for step in (mi.get("output_top_logprobs") or []):
        if not step:
            continue
        steps.append([[float(t[0]), int(t[1])] for t in step])
    rec = {"row": row["id"], "subset": row.get("subset"), "prompt_tokens": len(ids),
           "tokens": len(o.get("output_ids") or []), "ids": list(o.get("output_ids") or []),
           "topk": steps, "topk_n": a.topk,
           "logprobs": [s[0][0] for s in steps] if steps else [],
           "teacher_forced": False, "tag": a.tag, "seconds": round(secs, 1), "tp": tp}
    with open(a.out, "w") as fh:
        fh.write(json.dumps(rec) + "\n")
    same = len({tuple(x.get("output_ids") or []) for x in out}) == 1
    print(f"[{a.tag}] {len(rec['ids'])} tokens in {secs:.1f} s; ids {rec['ids']}\n"
          f"  {len(steps)} steps with top-{a.topk} distributions; the {a.batch} rows decoded identically: {same}",
          flush=True)
    if not steps:
        raise SystemExit("FAILED: no per-step distributions came back (return_logprob/top_logprobs_num)")


if __name__ == "__main__":
    main()
