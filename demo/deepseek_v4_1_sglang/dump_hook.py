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
# DSV41_DUMP_STEPS: with _LAYERS, also record the DECODE steps (the prefill is written once, as before). A decode step
# is the forward whose seq_lens are all 1, and its layer outputs are what a decode chain's per-sublayer dump is compared
# with. The default 10 matches the trace hook's step count; 0 records only the prefill.
_STEPS = int(os.environ.get("DSV41_DUMP_STEPS", "10"))
_SHAPES = os.environ.get("DSV41_DUMP_SHAPES", "0") != "0"   # print what a layer hands on (debug the quantity, not the values)
_step = [0]      # decode forwards seen so far (host side, so every rank's layers land in the same directory)
# DSV41_DUMP_AT is the position the chain replays (the prompt's last, i.e. the first decode step's): the pool dump waits
# for it, and only the prefill chunk that ends there is recorded as the layer reference. Without that gate the dummy
# one-token warmup forward -- which walks all 40 layers too -- was recorded as the reference.
_AT = int(os.environ.get("DSV41_DUMP_AT", "0"))


def _is_decode(fb) -> bool:
    m = getattr(fb, "forward_mode", None)
    try:
        return bool(m.is_decode())
    except Exception:
        return False


def _last_pos(kw, args):
    """This forward's last position, or None when it cannot be told."""
    p = kw.get("positions")
    if p is None:
        for v in args:
            if torch.is_tensor(v) and v.dtype in (torch.int32, torch.int64) and v.dim() == 1:
                p = v
                break
    try:
        return int(p.max())
    except Exception:
        return None


def _key(k) -> int:
    """A record key ("12" or "12_st") as its layer number."""
    return int(k.split("_")[0])


def _suf(k) -> str:
    """A record key's suffix ("_st" for the 4-stream residual, "" for the 5120 collapsed row)."""
    return k[k.index("_"):] if "_" in k else ""


def _write_layers(rec) -> None:
    """Every layer's residual of the prefill's last position, one flat fp32 row each (a chain comparison's reference)."""
    import numpy as np
    out = os.path.join(_DIR, "layers")
    os.makedirs(out, exist_ok=True)
    ids = sorted(rec["layers"], key=_key)
    for k in ids:
        np.array(rec["layers"][k].numpy(), dtype=np.float32).tofile(os.path.join(out, f"L{_key(k):02d}{_suf(k)}.f32"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"n": len(ids), "shape": {k: list(rec["layers"][k].shape) for k in ids}, "rows": ids}, f)
    print(f"[dump] {len(ids)} layer rows written to {out} "
          f"({sorted({int(rec['layers'][k].numel()) for k in ids})} values each)", flush=True)


def _write_step_rows(step, rows) -> None:
    """One decode step's every layer, each row in full: DIR/layers/step{N}/L{LL}.f32 + that step's meta."""
    import numpy as np
    out = os.path.join(_DIR, "layers", f"step{step}")
    os.makedirs(out, exist_ok=True)
    ids = sorted(rows, key=_key)
    for k in ids:
        np.array(rows[k].numpy(), dtype=np.float32).tofile(os.path.join(out, f"L{_key(k):02d}{_suf(k)}.f32"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"step": step, "n": len(ids), "shape": {k: list(rows[k].shape) for k in ids},
                   "dtype": "float32 (bf16 values, exactly representable)"}, f)
    print(f"[dump] decode step {step}: {len(ids)} layer rows written to {out}", flush=True)


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
    import torch   # (module scope has no torch: only the functions below imported it, so the layer hook below used to
                   # raise NameError into its own except and record nothing -- the dump it wrote never existed)
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
    # The sublayer boundary: `run_attn_post`/`run_moe_post` hand back the residual right after the attention or the
    # MoE update, which is what the static chain's `attn{L}_streams_out`/`moe{L}_streams_out` hold. Without them a
    # mismatch can only be placed at a whole layer, and a layer holds two sublayers.
    try:
        from sglang.srt.models import deepseek_v4_mhc as _mhc

        def _record_post(r, suffix):
            if not _LAYERS or _gate["lid"] < 0:
                return
            v = getattr(r, "streams", None)
            if not (torch.is_tensor(v) and v.dim() >= 2):
                return
            row = v.detach().float().cpu().reshape(v.shape[0], -1)[-1]
            if _gate["pref"]:
                _rec["layers"][f"{_gate['lid']}{suffix}"] = row
            elif _gate["step"]:
                _step_rec["rows"][f"{_gate['lid']}{suffix}"] = row

        _oap, _omp = _mhc.run_attn_post, _mhc.run_moe_post

        def _attn_post(*p, **k):
            r = _oap(*p, **k)
            _record_post(r, "_apost")
            return r

        def _moe_post(*p, **k):
            r = _omp(*p, **k)
            _record_post(r, "_mpost")
            return r

        _mhc.run_attn_post, _mhc.run_moe_post = _attn_post, _moe_post
    except Exception as e:
        print(f"[dump] sublayer post hook failed: {type(e).__name__}: {e}", flush=True)

    if _LAYERS or _SHAPES:
        try:      # the chain starts from this: hc_broadcast's output IS layer 0's residual (the embedding in every
            from sglang.kernels.ops.layernorm import mhc as _mhc   # stream), so recording it pins the chain's input
            _ob = _mhc.hc_broadcast
            _bcn = [0]

            def _bc(x, hc_mult, *ba, **bk):
                y = _ob(x, hc_mult, *ba, **bk)
                _bcn[0] += 1
                if _bcn[0] <= 2:
                    try:
                        import numpy as _np
                        print(f"[dump] hc_broadcast#{_bcn[0]} x={tuple(x.shape)} y={tuple(y.shape)} "
                              f"x_rms={float(x.detach().float().pow(2).mean().sqrt()):.5f} "
                              f"y_rms={float(y.detach().float().pow(2).mean().sqrt()):.5f}", flush=True)
                        for nm, v in (("broadcast", y), ("embed", x)):
                            _np.array(v[-1].detach().float().cpu().reshape(-1).numpy(), dtype=_np.float32).tofile(
                                os.path.join(_DIR, f"{nm}{_bcn[0]}.f32"))
                    except Exception as e:
                        print(f"[dump] broadcast record failed: {type(e).__name__}: {e}", flush=True)
                return y

            _mhc.hc_broadcast = _bc
        except Exception as e:
            print(f"[dump] hc_broadcast hook failed: {type(e).__name__}: {e}", flush=True)

        _orig_layer = _v4.DeepseekV4CudaDecoderLayer.forward
        _rec = {"layers": {}}       # the prefill's last position, written once
        _step_rec = {"rows": {}}    # this decode step's layers, cleared when the step's counter moves

        _shapes = {"n": 0}
        _gate = {"lid": -1, "pref": False, "step": 0}   # the forward being recorded: set per layer, read by the posts

        def _layer_forward(self, *args, **kw):
            fb = kw.get("forward_batch")
            if fb is None and len(args) > 3:
                fb = args[3]
            lid0 = int(getattr(self, "layer_id", -1))
            dec0, last0 = _is_decode(fb), _last_pos(kw, args)
            _gate.update(lid=lid0,
                         pref=(not dec0) and last0 is not None and (_AT <= 0 or last0 == _AT)
                              and not _rec.get("done", False),
                         step=_step[0] if (dec0 and 0 < _step[0] <= _STEPS) else 0)
            if not _gate["pref"] and not _gate["step"]:
                return _orig_layer(self, *args, **kw)
            pre = None    # the residual the layer READS, taken before it runs: layer 0's is the embedding in every
            try:          # stream (hc_broadcast), so it is the input the chain must start from
                from sglang.srt.layers.layer_boundary.residual import batch as _rb
                _rs = _rb.written_residual(fb)
                pre = getattr(_rs, "streams", _rs)
                if not (torch.is_tensor(pre) and pre.dim() >= 2):
                    pre = None
            except Exception as e:
                if _SHAPES and _shapes["n"] < 2:
                    print(f"[dump] pre-residual failed: {type(e).__name__}: {e}", flush=True)
            out = _orig_layer(self, *args, **kw)
            if _SHAPES and _shapes["n"] < 2 and int(getattr(self, "layer_id", -1)) in (0, 1):
                _shapes["n"] += 1
                def s(v, nm):
                    return f"{nm}={tuple(v.shape)}" if torch.is_tensor(v) else f"{nm}=<{type(v).__name__}>"
                parts = [s(out, "out")]
                for nm in ("residual", "streams", "pre", "input", "stats"):
                    v = getattr(out, nm, None) if not torch.is_tensor(out) else None
                    if v is not None:
                        parts.append(s(v, nm))
                        rows = getattr(v, "rows", None)
                        if rows is not None:
                            parts.append(s(rows, f"{nm}.rows"))
                print(f"[dump] L{getattr(self, 'layer_id', -1)} out shapes: " + " ".join(parts), flush=True)
                try:   # the mHC residual the layer leaves in the batch: the 4-stream R' (20480), not the 5120 row it returns
                    from sglang.srt.layers.layer_boundary.residual import batch as _rb
                    fb = kw.get("forward_batch")
                    if fb is None and len(args) > 3:
                        fb = args[3]
                    r = _rb.written_residual(fb)
                    print(f"[dump] L{getattr(self, 'layer_id', -1)} written_residual: "
                          f"{tuple(r.shape) if torch.is_tensor(r) else type(r).__name__}", flush=True)
                except Exception as e:
                    print(f"[dump] written_residual failed: {type(e).__name__}: {e}", flush=True)
            t = out if torch.is_tensor(out) else getattr(out, "residual", None)
            st = None     # the 4 mHC streams the layer reads (20480): what a static chain's streams_out holds, not the
            try:          # 5120 row the layer returns (that one is the collapsed hidden, a different quantity)
                from sglang.srt.layers.layer_boundary.residual import batch as _rb
                rs = _rb.written_residual(fb)
                st = getattr(rs, "streams", rs)
                if not (torch.is_tensor(st) and st.dim() >= 2):
                    st = None
            except Exception:
                st = None
            if _LAYERS and torch.is_tensor(t) and t.dim() >= 2:
                lid = int(getattr(self, "layer_id", -1))
                pair = {"": t, "_st": st, "_in": pre}
                # WHICH forward this is: SGLang runs a one-token dummy forward before the real prefill (a warmup at
                # the decode shape), and it walks all 40 layers. Recording it made every "reference" layer the dummy
                # token's -- and it is also why L0 never matched. Keep the forward that carries the position the chain
                # replays (a prefill chunk ending at _AT) and the real decode steps, and nothing else.
                dec = _is_decode(fb)
                last = _last_pos(kw, args)
                if dec and 0 < _step[0] <= _STEPS:   # a decode step: every layer's row, in full
                    try:
                        r = _step_rec["rows"]
                        for suf, v in pair.items():
                            if v is not None:
                                r[f"{lid}{suf}"] = v.detach().float().cpu().reshape(v.shape[0], -1)[-1]
                        if len({_key(k) for k in r}) >= 40:  # (the last layer of the group: this step is complete)
                            _write_step_rows(_step[0], r)
                            r.clear()
                    except Exception as e:
                        print(f"[dump] step record failed {type(e).__name__}: {e}", flush=True)
                elif not dec and last is not None and (_AT <= 0 or last == _AT) and not _rec.get("done", False):
                    try:                             # the prefill chunk holding the chain's position: the last row
                        for suf, v in pair.items():
                            if v is not None:
                                _rec["layers"][f"{lid}{suf}"] = v[-1].detach().float().cpu().reshape(-1)
                        if len({_key(k) for k in _rec["layers"]}) >= 40:   # (the last layer of the group: write, once)
                            _rec["done"] = True
                            _write_layers(_rec)
                    except Exception as e:
                        print(f"[dump] layer record failed {type(e).__name__}: {e}", flush=True)
            _gate["lid"] = -1
            return out

        _v4.DeepseekV4CudaDecoderLayer.forward = _layer_forward

    orig = _mr.ModelRunner.forward
    seen = [0]

    def forward(self, forward_batch=None, *args, **kw):
        # a decode step's rows are one token each; the counter must move BEFORE the layers run,
        # or the step that just started records itself as the next number (and step 1 is never written).
        # The mode must say decode: the one-token warmup forward also has seq_lens == 1, and letting it through
        # shifted every step by one (and made step 1 the warmup's dummy token).
        lens = getattr(forward_batch, "seq_lens", None)
        if _LAYERS and _is_decode(forward_batch) and _step[0] < _STEPS:
            _step[0] += 1
            print(f"[dump] decode step {_step[0]} ({0 if lens is None else len(lens)} row(s))", flush=True)
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
