"""Dump prefill's KV caches from INSIDE the TP workers (the pools live in the worker processes, not the engine's).

Why a sitecustomize and not a patch in the dumping script: SGLang's TP workers are separate processes, and the
parent's patches do not reach them. This is the same mechanism boot/sitecustomize.py uses for the static layers
(DSV41_STATIC_ATTN/DSV41_STATIC_MOE): Python imports it at the start of every process, workers included, so the
ModelRunner here is the worker's own. After the prefill's first decode step the pools hold the row's real KV.

  DSV41_DUMP_PREFILL=/tmp/prefill_row0 PYTHONPATH=boot python dump_prefill.py ...

The buffers written are the four bind_pools binds (dsv41_static_layer): window_cache (the SWA window),
compress_cache (the compressed KV), index_k (the indexer's keys), cmp_state (the compressor's state). L0's input x
is the embedding of the prompt's last token -- full_decode.py already reads that the same way.
"""
import json
import os

_DIR = os.environ.get("DSV41_DUMP_PREFILL")
if _DIR:
    try:
        import torch
        from sglang.srt.model_executor import model_runner as _mr

        _orig = _mr.ModelRunner.forward

        def _forward(self, *args, **kw):
            out = _orig(self, *args, **kw)
            # Once, after the prefill: the pools now hold the row's [0, pos] rows (the decode step after it).
            if not getattr(self, "_dsv41_dumped", False):
                fbi = getattr(self, "forward_batch", None)
                seq = getattr(fbi, "seq_lens", None) if fbi is not None else None
                if seq is not None and int(max(seq)) > 1:   # a prefill (a decode's is 1 new row)
                    self._dsv41_dumped = True
                    _dump(self)
            return out

        def _dump(mr):
            pool = mr.token_to_kv_pool
            layers = len(mr.model_config.hf_config.compress_ratios)
            os.makedirs(_DIR, exist_ok=True)
            meta = {"buffers": {}, "tp_rank": int(os.environ.get("SGLANG_TP_RANK", "-1")),
                    "pos": int(max(mr.forward_batch.seq_lens)) - 1}
            for L in range(layers):
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
                    b.view(torch.uint8).numpy().tofile(os.path.join(_DIR, fn))
                    meta["buffers"][f"L{L}.{name}"] = {"file": fn, "shape": list(b.shape),
                                                       "dtype": str(b.dtype)}
            with open(os.path.join(_DIR, "meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            print(f"[dump] {len(meta['buffers'])} buffers of rank {meta['tp_rank']} at pos {meta['pos']} "
                  f"written to {_DIR}", flush=True)

        _mr.ModelRunner.forward = _forward
    except Exception as e:      # never break the engine: a dump that fails is a dump that printed why
        print(f"[dump] sitecustomize could not install: {type(e).__name__}: {e}", flush=True)
