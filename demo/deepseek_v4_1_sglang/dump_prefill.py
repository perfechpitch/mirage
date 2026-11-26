"""Dump the KV caches a real 32K prefill leaves in SGLang's pools, plus L0's input x, for full_decode.py.

Why: full_decode.py (the static megakernel's own chain, one process, 8 GPUs) fills its caches with random bytes,
so its tokens can never be compared with golden_bench32k.jsonl. This runs the same row's prefill through SGLang
-- real weights, real 32K -- and writes every layer's pool buffers as the static kernels read them (the same
accessors dsv41_static_layer.bind_pools uses), so full_decode.py can bind them instead of its own random ones.

  python dump_prefill.py --model DIR --row 0 --out /tmp/prefill_row0
  python full_decode.py --checkpoint DIR --layers 40 --gpus 8 --inputs-dir /tmp/prefill_row0

The model runner is captured by wrapping ModelRunner.init_cuda_graphs (what dsv41_static_hook.py patches too):
no static layer is built here, so this dumps the pools and nothing else.
"""
import argparse
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

A = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
A.add_argument("--model", required=True)
A.add_argument("--data", default="/opt/dlami/nvme/saboxu/data/long-code-arena/bench32k/*.jsonl")
A.add_argument("--row", type=int, default=0)
A.add_argument("--out", required=True)
A.add_argument("--tp", type=int, default=8)
A.add_argument("--mem-fraction", type=float, default=0.40)
A.add_argument("--engram-config", default='{"placement": "cpu"}')
A.add_argument("--max-new", type=int, default=1, help="decode tokens to run after the prefill (1: the next token)")
A.add_argument("--drop-last", type=int, default=0, metavar="K",
               help="prefill prompt[:-K] (1: the pools hold [0, n-2] and the decode step at n-1 has golden's ids[0])")
a = A.parse_args()


def log(*m):
    print("[dump]", *m, flush=True)


RUNNERS = []


def main():
    import sglang  # noqa: F401
    from sglang.srt.model_executor import model_runner as _mr

    orig = _mr.ModelRunner.init_cuda_graphs

    def patched(self, *args, **kw):
        RUNNERS.append(self)          # this rank's runner: its token_to_kv_pool is the one bind_pools reads
        return orig(self, *args, **kw)

    _mr.ModelRunner.init_cuda_graphs = patched

    files = sorted(glob.glob(a.data))
    rows = [json.loads(line) for f in files for line in open(f)]
    row = rows[a.row]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ids = tok.encode(row["prompt"], add_special_tokens=False)   # (the row's "prompt_tokens" field is stale: 32344
    n = len(ids)                                                # for a 31906-token prompt -- golden's 31906 is right)
    # --drop-last 1: prefill prompt[:-1] so the pools hold [0, n-2], and the decode step that produces golden's ids[0]
    # is then this row's LAST token at position n-1 (what full_decode.py replays). The default keeps the whole prompt.
    if a.drop_last:
        ids = ids[:-a.drop_last]
        log(f"row {a.row} ({row.get('subset', '?')}), prompt {n} tokens, prefilling the first {len(ids)} "
            f"(the decode step at position {len(ids)} produces golden's {json.loads(open(os.path.join(HERE, 'golden_bench32k.jsonl')).readline())['ids'][0]})")
        n = len(ids)
    else:
        log(f"row {a.row} ({row.get('subset', '?')}), prompt {n} tokens")
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("DSV41_DUMP_AT", str(n - 1))   # (n is the prefilled length after --drop-last)   # the last prefill chunk's end: the pools' [0, pos] is complete

    from sglang.srt.entrypoints.engine import Engine
    ctx = -(-(n + a.max_new + 1024) // 1024) * 1024
    log(f"context_length {ctx}")
    eng = Engine(model_path=a.model, tokenizer_path=a.model, trust_remote_code=True, tp_size=a.tp,
                 max_running_requests=1, cuda_graph_max_bs_decode=1, context_length=ctx,
                 max_total_tokens=ctx + 4096, mem_fraction_static=a.mem_fraction,
                 disable_radix_cache=True, disable_cuda_graph=True, log_level="warning",
                 engram_config=json.loads(a.engram_config))
    try:
        out = eng.generate(input_ids=ids, sampling_params={"max_new_tokens": a.max_new, "temperature": 0.0})
        log(f"raw output: {str(out)[:400]}")
        got = out[0]["output_ids"] if isinstance(out, list) and "output_ids" in out[0] else None
        g = [json.loads(l) for l in open(os.path.join(HERE, "golden_bench32k.jsonl"))][a.row]
        log(f"prefill + {a.max_new} decode done; SGLang's next id {got}, golden's {g['ids'][0]}")

        log("the TP workers dumped their pools through boot/sitecustomize.py + dump_hook.py")
        return
        r = RUNNERS[0]
        pool = r.token_to_kv_pool
        log(f"model runner {type(r).__name__}, pool {type(pool).__name__}")

        meta = {"row": a.row, "prompt_tokens": n, "subset": row.get("subset"),
                "golden_ids": g["ids"], "tp": a.tp, "pos": n - 1, "buffers": {}}
        for L in range(len(r.model_config.hf_config.compress_ratios) if hasattr(r, "model_config") else 40):
            got_any = False
            for name, getter in (("window_cache", lambda: pool.get_swa_raw_buffer(L)),
                                 ("compress_cache", lambda: pool.get_extra_key_buffer(L)),
                                 ("index_k", lambda: pool.get_index_k_with_scale_buffer(L)),
                                 ("cmp_state", lambda: pool.get_attention_compress_states(L))):
                try:
                    t = getter()
                except Exception:
                    continue
                if t is None or not torch.is_tensor(t):
                    continue
                b = t.detach().contiguous().cpu()
                fn = f"L{L:02d}_{name}.bin"
                b.view(torch.uint8).numpy().tofile(os.path.join(a.out, fn))
                meta["buffers"][f"L{L}.{name}"] = {"file": fn, "shape": list(b.shape), "dtype": str(b.dtype)}
                got_any = True
            if got_any and L % 8 == 0:
                log(f"L{L} dumped")
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        log(f"{len(meta['buffers'])} buffers + meta.json written to {a.out}")
    finally:
        eng.shutdown()


if __name__ == "__main__":   # SGLang spawns TP workers that re-import this module
    main()
