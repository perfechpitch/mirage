"""Dump prefill's KV caches from inside the TP workers (the pools live in the worker processes).

Loaded by boot/sitecustomize.py in every process of a dump_prefill.py run, the TP workers included -- the parent's
patches never reach a worker, which is why the first attempt captured no ModelRunner.

The four buffers are the ones dsv41_static_layer.bind_pools binds: window_cache (the SWA window), compress_cache
(the compressed KV), index_k (the indexer's keys), cmp_state (the compressor's state).

Two things that are easy to get wrong, both of which the first dumps got wrong:
 - the TP rank: SGLANG_TP_RANK is not in the workers' env, so every rank wrote r00_* and the last one won. The
   rank is the model runner's gpu_id (each rank's caches are its heads' shard).
 - the chunk: SGLang prefills in chunks (16384 here), and a chunk's forward has seq_lens = that chunk's end, so
   the FIRST prefill forward does not hold the prompt. Wait for DSV41_DUMP_AT, the prompt's last position.
"""
import json
import os

_DIR = os.environ.get("DSV41_DUMP_PREFILL")
_AT = int(os.environ.get("DSV41_DUMP_AT", "0"))
_rank = [0]
_BE = []        # the attention backend the forward's layers asked for: get_attn_backend() needs the forward context
_LAYERS = os.environ.get("DSV41_DUMP_LAYERS", "0") != "0"


def _write_layers(rec) -> None:
    """Every layer's residual of the prefill's last position, one flat fp32 row each (a chain comparison's reference)."""
    import numpy as np
    out = os.path.join(_DIR, "layers")
    os.makedirs(out, exist_ok=True)
    ids = sorted(rec["layers"], key=lambda k: int(k) if k.isdigit() else 0)
    for k in ids:
        np.array(rec["layers"][k].numpy(), dtype=np.float32).tofile(os.path.join(out, f"L{int(k):02d}.f32"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"n": len(ids), "shape": list(rec["layers"][ids[0]].shape), "rows": ids}, f)
    print(f"[dump] {len(ids)} layers' last-row residuals written to {out} "
          f"({int(rec['layers'][ids[0]].numel())} values each)", flush=True)


def _dump(mr) -> None:
    import torch
    r = int(getattr(mr, "gpu_id", 0))
    pool = mr.token_to_kv_pool
    layers = len(mr.model_config.hf_config.compress_ratios)
    os.makedirs(_DIR, exist_ok=True)
    meta = {"tp_rank": r, "buffers": {}}
    for L in range(layers):
        for name, getter in (("window_cache", lambda: pool.get_swa_raw_buffer(L)),
                             ("compress_cache", lambda: pool.get_extra_key_buffer(L)),
                             ("index_k", lambda: pool.get_index_k_with_scale_buffer(L)),
                             ("cmp_state", lambda: pool.get_attention_compress_states(L))):
            try:
                t = getter()
            except Exception:
                continue
            if isinstance(t, (list, tuple)):
                t = t[0]
            if hasattr(t, "kv_score_buffer"):        # the compressor's state object, not a tensor
                t = t.kv_score_buffer.kv_score
            if t is None or not torch.is_tensor(t):
                continue
            b = t.detach().contiguous().cpu()
            fn = f"r{r:02d}_L{L:02d}_{name}.bin"
            b.view(torch.uint8).numpy().tofile(os.path.join(_DIR, fn))
            meta["buffers"][f"L{L}.{name}"] = {"file": fn, "shape": list(b.shape), "dtype": str(b.dtype)}
    with open(os.path.join(_DIR, f"meta_rank{r:02d}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[dump] rank {r}: {len(meta['buffers'])} buffers written to {_DIR}", flush=True)


def _dump_meta(mr, fb) -> None:
    """The last prefill row's addressing: the decode step at the next position continues this request, so its window rows
    and compressed page table are this row's (the window's rows hold positions pos-127..pos, the page table [0, pos]/ratio)."""
    import torch
    from sglang.srt.models import deepseek_v4 as _v4
    pool = mr.token_to_kv_pool
    be = _BE[-1] if _BE else None       # (get_attn_backend() reads the forward context: only valid inside the forward)
    fm = getattr(be, "forward_metadata", None)
    core = getattr(fm, "core_attn_metadata", None) if fm is not None else None
    if core is None and fm is not None:
        core = getattr(fm, "core_metadata", None)
    out = {"req_pool_indices": fb.req_pool_indices.cpu().tolist(),
           "seq_lens": fb.seq_lens.cpu().tolist(),
           "pool": {"swa_page": pool.get_swa_key_page_size(),
                    "swa_bytes_per_token": pool.get_swa_key_bytes_per_token(),
                    "swa_rows": int(pool.get_swa_raw_buffer(0).shape[0]),
                    "swa_row_bytes": int(pool.get_swa_raw_buffer(0).shape[1]),
                    "kv_layout": str(pool.kv_layout.value if hasattr(pool.kv_layout, "value") else pool.kv_layout)}}
    if core is not None:
        for name in ("swa_page_indices", "swa_topk_lengths", "swa_out_cache_loc", "positions",
                     "seq_lens_casual", "c1_out_loc", "c2_out_loc", "c1_topk_lengths", "c2_topk_lengths",
                     "candidates", "c128_topk_lengths_clamp1"):
            t = getattr(core, name, None)
            if t is not None and torch.is_tensor(t):
                out[name] = t.cpu().tolist()          # (every row of the prefill chunk, not just the last: the mapping
                                                        #  row(position) is a fit over these, not one guess)
        pt = getattr(core, "page_table", None)
        if pt is not None and torch.is_tensor(pt):
            out["page_table"] = pt.cpu().tolist()
        out["swa_page_indices_shape"] = list(core.swa_page_indices.shape) if hasattr(core, "swa_page_indices") else None
    with open(os.path.join(_DIR, f"meta_addr_rank{int(getattr(mr, 'gpu_id', 0)):02d}.json"), "w") as f:
        json.dump(out, f)
    print(f"[dump] rank {getattr(mr, 'gpu_id', 0)}: addressing {sorted(out)} "
          f"(swa_page_indices {len(out.get('swa_page_indices', []))} entries, page_table "
          f"{len(out.get('page_table', []))} pages)", flush=True)


def _install() -> None:
    from sglang.srt.model_executor import model_runner as _mr
    from sglang.srt.models import deepseek_v4 as _v4

    _orig_be = _v4.get_attn_backend

    def _be():
        be = _orig_be()
        if not _BE or _BE[-1] is not be:
            _BE.append(be)
        return be

    _v4.get_attn_backend = _be

    # --- per-layer residuals: the chain's per-sublayer streams_out can be compared with these, so a divergence
    # is localized to the first layer that has one instead of showing up only at the logits ---
    if _LAYERS:
        _orig_layer = _v4.DeepseekV4CudaDecoderLayer.forward
        _rec = {"layers": {}}

        def _layer_forward(self, *args, **kw):
            out = _orig_layer(self, *args, **kw)
            if not _rec.get("done", False):
                try:   # (the LAST row of each layer's output: the prefill's last position, the one the chain replays)
                    t = out if torch.is_tensor(out) else getattr(out, "residual", None)
                    if torch.is_tensor(t) and t.dim() >= 2:
                        _rec["layers"][str(getattr(self, "layer_id", len(_rec["layers"])))] = \
                            t[-1].detach().float().cpu().reshape(-1)
                    if len(_rec["layers"]) >= 40:      # (the last layer of the group: write, once)
                        _rec["done"] = True
                        _write_layers(_rec)
                except Exception as e:
                    print(f"[dump] layer record failed {type(e).__name__}: {e}", flush=True)
            return out

        _v4.DeepseekV4CudaDecoderLayer.forward = _layer_forward

    orig = _mr.ModelRunner.forward
    seen = [0]

    def forward(self, forward_batch=None, *args, **kw):
        out = orig(self, forward_batch, *args, **kw)
        if not getattr(self, "_dsv41_dumped", False):
            lens = getattr(forward_batch, "seq_lens", None)   # (a decode step's is all 1s: len(lens) > 1 alone is not a prefill)
            n = int(max(lens)) if lens is not None and len(lens) else 0
            if seen[0] < 6 and n > 1:
                seen[0] += 1
                print(f"[dump] rank {_rank[0]} forward #{self.forward_pass_id}: {len(lens)} rows, "
                      f"max seq_len {n} (waiting for {_AT})", flush=True)
            if n > 1 and n >= _AT:          # the last prefill chunk: the pools hold the prompt's [0, pos]
                self._dsv41_dumped = True
                try:
                    _dump(self)
                except Exception as e:
                    print(f"[dump] dump failed {type(e).__name__}: {e}", flush=True)
                try:
                    _dump_meta(self, forward_batch)
                except Exception as e:
                    print(f"[dump] addressing dump failed {type(e).__name__}: {e}", flush=True)
        return out

    _mr.ModelRunner.forward = forward


if _DIR:
    try:
        _install()
    except Exception as e:      # never break the engine
        print(f"[dump] hook not installed: {type(e).__name__}: {e}", flush=True)
