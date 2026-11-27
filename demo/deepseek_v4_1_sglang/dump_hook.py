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


_CAP = [False]   # set for the whole of a graph capture (any stream), see _install_cap_flag


def _capturing() -> bool:
    """True inside a CUDA-graph capture region. A `.cpu()` there (or any allocation) invalidates the capture --
    "operation failed due to a previous error during capture" -- and SGLang builds its decode graphs by replaying
    these very layers, so every probe must be a no-op while capturing."""
    if _CAP[0]:      # a capture in progress ANYWHERE, not just on the current stream (see _install_cap_flag)
        return True
    try:
        import torch
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _is_decode(fb) -> bool:
    m = getattr(fb, "forward_mode", None)
    try:
        return bool(m.is_decode())
    except Exception:
        return False


def _last_pos(kw, args, fb=None):
    """This forward's last position, or None when it cannot be told.

    `positions` is on the ForwardBatch in this tree (the layer's own kw/args no longer carry it), so without the
    `fb` fallback every prefill gate read None and no layer row was ever written -- the pools dumped and the layers
    directory stayed empty."""
    p = kw.get("positions")
    if p is None:
        for v in args:
            if torch.is_tensor(v) and v.dtype in (torch.int32, torch.int64) and v.dim() == 1:
                p = v
                break
    if p is None and fb is not None:
        p = getattr(fb, "positions", None)
        if p is not None and not torch.is_tensor(p):
            p = None
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


_ATTN_IO = os.environ.get("DSV41_DUMP_ATTN_IO", "0") != "0"
# The position the recorded decode step must carry: the one right after the prefill's last (= _AT). Without it the
# engine's dummy one-token decode is what gets recorded (see _layer_forward).
_DEC_POS = int(os.environ.get("DSV41_DUMP_DEC_POS", "0")) or ((_AT + 1) if _AT > 0 else 0)
# DSV41_DUMP_ATTN_IO: also record the attention's INSIDE, per layer per rank, for decode step 1:
#   o_rope  `o` as the attention core hands it back (rope space, per-rank heads) -- what the chain's attention writes
#   o_inv   the same buffer after `_project_wo_a`'s in-place inverse RoPE: the wo_a input (== the chain's `o_q`)
#   oa      `_project_wo_a`'s return, [T, G, R] bf16: the wo_b input (the chain's `oa_q` before act_quant)
#   y_part  `_project_wo_b`'s return with the all-reduce deferred: this rank's UNREDUCED bf16 partial
# Without `o_inv` and `oa` the last bit of L0 cannot be attributed: the chain's `o` differs from the reference's by
# 0.08 bf16 ulp, and the wo path then turns that into 0.44-0.58, with no way to tell which side owns it.
# The writes are plain .bin files (bf16 raw, [T, ...]) under DIR/layers/step1/io/, one name per layer and rank.


def _attn_io_writer():
    """(name, tensor) -> None, writing DIR/layers/step{N}/io/L{LL}_r{rank:02d}_{name}.bin as bf16 raw."""
    import numpy as np
    import torch

    _said = [0]

    def write(name, t, lid=None) -> None:
        if not (torch.is_tensor(t) and t.numel()):
            return
        if _capturing():
            if _said[0] < 3:      # say it once, so a run that records nothing explains itself
                _said[0] += 1
                print(f"[dump] attn IO: skipping {name} during a graph capture "
                      f"(the recorded step must be a real forward)", flush=True)
            return
        step = _step[0]
        if not (0 < step <= _STEPS):
            return
        if lid is None:
            lid = int(getattr(t, "_dsv41_lid", -1))
        if lid < 0:
            return
        # a decode step is one token per sequence; a prefill's tensor is many rows long
        if t.shape[0] != 1:
            return
        d = os.path.join(_DIR, "layers", f"step{step}", "io")
        os.makedirs(d, exist_ok=True)
        r = int(t.device.index if t.device.index is not None else torch.cuda.current_device())
        b = t.detach().float().cpu().contiguous().numpy().astype(np.float32)
        # bf16 raw, so the file is exactly the values the kernel produced (an fp32 widening is lossless)
        b.astype(np.float32).view(np.uint32).__rshift__(16).astype(np.uint16).tofile(
            os.path.join(d, f"L{lid:02d}_r{r:02d}_{name}.bin"))

    return write


def _install_attn_io() -> None:
    import numpy as np
    import torch
    from sglang.srt.models import deepseek_v4 as _v4

    write = _attn_io_writer()
    cls = _v4.MQALayer            # (not MqaAttentionBase: the wo_a/wo_b projections live on the subclass)

    _owa = cls._project_wo_a

    def _project_wo_a(self, o, *a, **kw):
        lid = int(getattr(self, "layer_id", -1))
        try:
            write("o_rope", o, lid)                       # before: the core's output, still in rope space
        except Exception:
            pass
        out = _owa(self, o, *a, **kw)
        try:
            # `fused_rope_inplace` mutates o[..., -64:] in place, so the SAME buffer is now the wo_a input
            write("o_inv", o, lid)
            write("oa", out, lid)                         # [T, G, R] bf16, or an (fp8, scale) pair on the fp8 path
        except Exception:
            pass
        return out

    _owb = cls._project_wo_b

    def _project_wo_b(self, o, *a, **kw):
        lid = int(getattr(self, "layer_id", -1))
        out = _owb(self, o, *a, **kw)
        try:
            # mhc.AttnOutput is a NamedTuple whose field is `partial` (not `.data`, which a NamedTuple does not have):
            # the first version silently wrote no y_part at all, because a NamedTuple is not a tensor and the writer
            # just returned. 960 io files instead of 1280 was the only symptom.
            v = getattr(out, "partial", None)
            if v is None:
                v = out.data if hasattr(out, "data") else out
            write("y_part", v, lid)
        except Exception:
            pass
        return out

    cls._project_wo_a, cls._project_wo_b = _project_wo_a, _project_wo_b

    try:      # the decode step's logits: the goal's last requirement, which no dump has ever recorded
        from sglang.srt.layers.logits_processor import LogitsProcessor

        _lp = LogitsProcessor.forward
        _said_lp = [0]

        def _lp_fwd(self, input_ids, hidden_states, lm_head, logits_metadata, *a, **kw):
            out = _lp(self, input_ids, hidden_states, lm_head, logits_metadata, *a, **kw)
            try:
                step = _step[0]
                # the field is `next_token_logits`; there is no `.logits`, so the first version recorded nothing and
                # raised nothing -- a silent no-op, the same failure mode as y_part above.
                lg = getattr(out, "next_token_logits", None)
                if lg is None:
                    lg = getattr(out, "logits", None)
                # `_capturing()` is not optional here: the row-bucket graph captures the logits step too, and a `.cpu()`
                # inside it invalidates the capture ("operation failed due to a previous error during capture"),
                # which takes the whole engine down rather than just this record.
                if 0 < step <= _STEPS and not _capturing() and torch.is_tensor(lg) and lg.shape[0] >= 1:
                    d = os.path.join(_DIR, "layers", f"step{step}")
                    os.makedirs(d, exist_ok=True)
                    row = lg.detach().float().cpu().reshape(lg.shape[0], -1)[-1]
                    row.numpy().astype(np.float32).tofile(os.path.join(d, "logits.f32"))
                    print(f"[dump] decode step {step}: logits {tuple(row.shape)} written", flush=True)
                elif _said_lp[0] < 2:
                    _said_lp[0] += 1
                    print(f"[dump] logits hook fired, nothing to record: step {step} capturing {_capturing()} "
                          f"tensor {torch.is_tensor(lg)} shape {None if lg is None else tuple(lg.shape)}", flush=True)
            except Exception as e:
                print(f"[dump] logits record failed: {type(e).__name__}: {e}", flush=True)
            return out

        LogitsProcessor.forward = _lp_fwd
    except Exception as e:
        print(f"[dump] logits hook failed: {type(e).__name__}: {e}", flush=True)

    print(f"[dump] attn IO hook installed (DSV41_DUMP_ATTN_IO) -> {_DIR}/layers/step*/io", flush=True)


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
            if not _LAYERS or _gate["lid"] < 0 or _capturing():
                return
            v = getattr(r, "streams", None)
            if not (torch.is_tensor(v) and v.dim() >= 2):
                return
            row = v.detach().float().cpu().reshape(v.shape[0], -1)[-1]
            # The coefficients that produced this residual: post[j] and comb[i][j] (R'[j] = sum_i comb[i][j] R[i] +
            # post[j] out, hc_post.cuh). They go into the same row record, so one comparison covers both the residual
            # and the mix that made it -- a wrong comb moves one stream and leaves its neighbours right, which looks
            # exactly like a wrong sublayer output in the residuals alone.
            coef = {}
            for nm in ("post", "comb"):
                o = getattr(r, nm, None)
                for cand in (getattr(o, "data", None), o):
                    if torch.is_tensor(cand) and cand.numel():
                        coef[nm] = cand.detach().float().cpu().reshape(-1)
                        break
            if _gate["pref"]:
                _rec["layers"][f"{_gate['lid']}{suffix}"] = row
                for nm, t in coef.items():
                    _rec["layers"][f"{_gate['lid']}{suffix}_{nm}"] = t
            elif _gate["step"]:
                _step_rec["rows"][f"{_gate['lid']}{suffix}"] = row
                for nm, t in coef.items():
                    _step_rec["rows"][f"{_gate['lid']}{suffix}_{nm}"] = t

        _oap, _omp = _mhc.run_attn_post, _mhc.run_moe_post
        _oct = getattr(_mhc, "_compute_triplet", None)

        def _triplet(*p, **k):
            """The (pre, post, comb) one post is about to apply. Recorded here because it is the one function both post
            paths (the fused collective and the plain one) build their coefficients in, and a layer whose post is fused
            into the next layer's boundary never reaches run_moe_post at all."""
            t = _oct(*p, **k)
            if _LAYERS and _gate["lid"] >= 0 and not _capturing():
                # The residual the coefficients were computed FROM. Recording it is what makes the triplet unambiguous:
                # the coefficient is a pure function of this tensor, so a fingerprint says which dumped residual
                # (`_in`, `_apost`, the previous layer's...) SGLang actually used -- instead of inferring it from the
                # post equation, which has several plausible readings once the layers' posts are fused.
                if len(p) > 1 and torch.is_tensor(p[1]) and p[1].numel():
                    res = p[1].detach().float().reshape(p[1].shape[0], -1)[-1]
                    fp = torch.cat([res.reshape(-1, res.numel() // 4).norm(dim=-1), res[:8]])
                    key = f"{_gate['lid']}_ct_resid_fp"
                    if _gate["pref"]:
                        _rec["layers"][key] = fp.cpu()
                    elif _gate["step"]:
                        _step_rec["rows"][key] = fp.cpu()
                for nm, v in zip(("pre", "post", "comb"), t):
                    if torch.is_tensor(v) and v.numel():
                        row = v.detach().float().cpu().reshape(-1)
                        if _gate["pref"]:
                            _rec["layers"][f"{_gate['lid']}_ct_{nm}"] = row
                        elif _gate["step"]:
                            _step_rec["rows"][f"{_gate['lid']}_ct_{nm}"] = row
            return t

        if _oct is not None:
            _mhc._compute_triplet = _triplet

        def _as_tensor(out):
            """The sublayer output as one tensor. The MoE hands the post its PIECES unmerged (`leave_output_sum=True`:
            the boundary owns the TP reduction and the replicated shared add), so its routes/outputs are nested
            objects, not a tensor -- trying the known accessors in order is what makes the FFN's own output
            recordable at all."""
            if torch.is_tensor(out):
                return out
            for get in ("get_merged", "merged"):
                f = getattr(out, get, None)
                if callable(f):
                    try:
                        v = f()
                    except Exception:
                        continue
                    if torch.is_tensor(v):
                        return v
            for name in ("shared", "routed", "partial"):
                v = getattr(out, name, None)
                if torch.is_tensor(v):
                    return v
            return None

        def _record_out(out, suffix):
            """The sublayer's own output, i.e. the `x` the post mixes in: R'[i] = post[i] x + sum_j comb[j][i] R[j].
            This is the quantity the static chain's kernel computes internally and does not expose as a node buffer for
            the MoE (its sum_store is fused with the post), so recording it here is the only way to compare it."""
            if not (_LAYERS and _gate["lid"] >= 0) or _capturing():
                return
            t = _as_tensor(out)
            if not (torch.is_tensor(t) and t.numel()):
                return
            row = t.detach().float().reshape(t.shape[0], -1)[-1]
            if _gate["pref"]:
                _rec["layers"][f"{_gate['lid']}{suffix}"] = row.cpu()
            elif _gate["step"]:
                _step_rec["rows"][f"{_gate['lid']}{suffix}"] = row.cpu()

        def _attn_post(*p, **k):
            if len(p) > 1:
                _record_out(p[1], "_apost_x")
            r = _oap(*p, **k)
            _record_post(r, "_apost")
            return r

        def _moe_post(*p, **k):
            if len(p) > 1:
                _record_out(p[1], "_mpost_x")
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
                if _bcn[0] <= 2 and not _capturing():
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

        try:      # (11-27) the Engram at layers 1 and 14: it REPLACES the residual before the layer runs
            from sglang.srt.layers.engram import Engram as _Engram   # (deepseek_v4.py:4360), so the chain cannot
            _oe = _Engram.forward                                    # follow SGLang past L1 without it. Record what it
            _egn = [0]                                               # consumes: x (the residual it gates), the hash ids
            _EG_LID = {}                                             # and the gathered rows (its two inputs), and its
            try:                                                     # output (to validate an off-line gate).
                _mp = os.environ.get("DSV41_DUMP_MODEL")
                if _mp:
                    _tc = json.load(open(os.path.join(_mp, "config.json"))).get("text_config", {})
                    for _i, _l in enumerate(_tc.get("engram_layer_ids", [])):
                        _EG_LID[_i] = int(_l)
            except Exception as _e:
                print(f"[dump] engram layer ids unavailable ({_e}); keys will use the hash index", flush=True)
            # The rows forward actually feeds to wkv, not a second gather: calling embed_rows() again after the fact
            # does NOT return the same rows (measured 11-27: the re-gathered emb reproduces neither the model's R''
            # nor, fed back through wkv, anything correlated with y - x). Patching the method that produces them is
            # the only way to be sure the recorded rows are the ones the GEMM saw.
            _orows, _EG_EMB = _Engram.embed_rows, {}
            def _egrows(self, hash_ids, fb, cp, *ba, **bk):
                out = _orows(self, hash_ids, fb, cp, *ba, **bk)
                if _egn_in_forward[0]:
                    _EG_EMB["e"] = out
                return out
            _Engram.embed_rows = _egrows
            _egn_in_forward = [False]
            def _egf(self, x, hash_ids, *a, **kw):                   # output (to validate an off-line gate).
                _egn_in_forward[0] = True
                try:
                    y = _oe(self, x, hash_ids, *a, **kw)
                finally:
                    _egn_in_forward[0] = False
                _egn[0] += 1
                gi = int(getattr(self, "layer_hash_index", -1))
                lid = _EG_LID.get(gi, gi)     # the LAYER number, not the hash index: a dump file named by the
                                              # hash index would be L00 for layer 1, which nothing else reads that way
                if (_LAYERS or _SHAPES) and not _capturing():
                    try:
                        import numpy as _np
                        fb = a[0] if a else kw.get("forward_batch")
                        cp = bool(kw.get("cp_all_tokens", False))
                        emb = _EG_EMB.pop("e", None)      # the rows forward used (see embed_rows' patch)
                        if emb is None:
                            print("[dump] engram: forward did not call embed_rows; recording no rows", flush=True)
                            return y
                        if _egn[0] <= 2:
                            print(f"[dump] engram#{_egn[0]} hidx={gi} x={tuple(x.shape)} ids={tuple(hash_ids.shape)} "
                                  f"emb={tuple(emb.shape)} x_rms={float(x.detach().float().pow(2).mean().sqrt()):.5f} "
                                  f"y_rms={float(y.detach().float().pow(2).mean().sqrt()):.5f}", flush=True)
                        row = lambda t: t[-1].detach().float().cpu().reshape(-1).clone()
                        # keys "N_eg_x"/"N_eg_y"/"N_eg_emb" land as L{N:02d}_eg_x.f32 ... next to layer N (the
                        # writers' _key/_suf split on the first "_"), by LAYER number. The hash ids are only printed:
                        # they exceed fp32's exact range and nothing downstream needs them once the rows are here.
                        rec = {"x": row(x), "y": row(y), "emb": row(emb.reshape(emb.shape[0], -1))}
                        if _egn[0] <= 2:
                            print(f"[dump] engram ids (first 8): {hash_ids[-1].detach().cpu().reshape(-1)[:8].tolist()}",
                                  flush=True)
                        if _gate["pref"]:
                            for k, v in rec.items():
                                _rec["layers"][f"{lid}_eg_{k}"] = v
                        elif _gate["step"]:
                            for k, v in rec.items():
                                _step_rec["rows"][f"{lid}_eg_{k}"] = v
                    except Exception as e:
                        print(f"[dump] engram record failed: {type(e).__name__}: {e}", flush=True)
                return y

            _Engram.forward = _egf
        except Exception as e:
            print(f"[dump] engram hook failed: {type(e).__name__}: {e}", flush=True)

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
            # BEFORE anything reads a tensor: `_last_pos` does `int(p.max())`, a device sync, and this forward runs
            # inside the prefill runner's breakable capture. A sync there invalidates the capture
            # ("operation failed due to a previous error during capture"), which is what killed three dump attempts
            # before the prompt was even fed -- so the guard has to be the first thing after the cheap attribute reads.
            if _capturing():
                return _orig_layer(self, *args, **kw)
            dec0, last0 = _is_decode(fb), _last_pos(kw, args, fb)
            # A decode step is counted here, at its FIRST layer, not in ModelRunner.forward: this tree runs the
            # decode through the replay runner, so the ModelRunner wrapper never saw a decode forward at all and
            # no step row (and no attn IO) was ever written.
            #
            # It also has to be identified by POSITION, not by being "the first decode forward": the engine runs a
            # one-token dummy decode before the real one, and recording it is not a near miss -- its token is not the
            # chain's. Measured: the first attempt's L00_in matched embed[103084] (cos 1.0), i.e. the warmup, against
            # the chain's embed[4588]; that reads as cos 0.04 with everything downstream, which looks like a broken
            # attention kernel rather than a wrong step. The real one is the position right after the prefill's last.
            dec_hit = dec0 and (_DEC_POS <= 0 or last0 == _DEC_POS)
            if dec0 and not dec_hit:
                _gate.update(lid=lid0, pref=False, step=0)
                return _orig_layer(self, *args, **kw)
            if dec_hit and lid0 == 0 and _step[0] < 1:
                _step[0] = 1
                print(f"[dump] decode step 1 (position {last0})", flush=True)
            _gate.update(lid=lid0,
                         pref=(not dec0) and last0 is not None and (_AT <= 0 or last0 == _AT)
                              and not _rec.get("done", False),
                         step=_step[0] if (dec0 and 0 < _step[0] <= _STEPS) else 0)
            if not _gate["pref"] and not _gate["step"]:
                return _orig_layer(self, *args, **kw)
            pre = None    # the residual the layer READS, taken before it runs: layer 0's is the embedding in every
            precoef = {}  # stream (hc_broadcast), so it is the input the chain must start from
            try:
                from sglang.srt.layers.layer_boundary.residual import batch as _rb
                _rs = _rb.written_residual(fb)
                pre = getattr(_rs, "streams", _rs)
                if not (torch.is_tensor(pre) and pre.dim() >= 2):
                    pre = None
                for nm in ("post", "comb"):   # the coefficients THIS stage's post will use (the fused boundary's
                    o = getattr(_rs, nm, None)  # included: those layers never reach run_moe_post, so this is the
                    for cand in (getattr(o, "data", None), o):   # only place their mix is visible)
                        if torch.is_tensor(cand) and cand.numel():
                            precoef[nm] = cand.detach().float().cpu().reshape(-1)
                            break
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
                for nm, c in precoef.items():
                    pair[f"_in{nm}"] = c      # (the mix this stage's post applies: see the pre-residual read above)
                # WHICH forward this is: SGLang runs a one-token dummy forward before the real prefill (a warmup at
                # the decode shape), and it walks all 40 layers. Recording it made every "reference" layer the dummy
                # token's -- and it is also why L0 never matched. Keep the forward that carries the position the chain
                # replays (a prefill chunk ending at _AT) and the real decode steps, and nothing else.
                dec = _is_decode(fb)
                last = _last_pos(kw, args, fb)
                if dec and 0 < _step[0] <= _STEPS and not _capturing():   # a decode step: every layer's row, in full
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
                elif not dec and last is not None and (_AT <= 0 or last == _AT) and not _rec.get("done", False) \
                        and not _capturing():
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
        # The step counter is advanced by the LAYER hook now (see _layer_forward): this tree runs the decode through
        # the replay runner, which does not pass through ModelRunner.forward, so counting here never fired. Kept as a
        # no-op comment rather than a second counter -- two counters would double-advance once a path uses both.
        lens = getattr(forward_batch, "seq_lens", None)
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


def _install_cap_flag() -> None:
    """Mark the WHOLE capture as capturing, not just the current stream.

    `torch.cuda.is_current_stream_capturing()` answers for one stream. SGLang captures the row-bucket graph while the
    layers' work can run on an alternate stream (TBO / `alt_streams`), so a `.cpu()` from a hook there sees "not
    capturing" and invalidates the capture -- the engine then dies at the next `torch.empty` with
    "Invalid stream capture status", which is how the first DSV41_DUMP_ATTN_IO run died. A global flag closes that
    hole regardless of which stream the hook happens to run on.
    """
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        breakable_cuda_graph as _bcg,
    )

    cls = _bcg.BreakableCUDAGraphCapture
    _ent, _ext = cls.__enter__, cls.__exit__

    def enter(self, *a, **kw):
        _CAP[0] = True
        return _ent(self, *a, **kw)

    def exit_(self, *a, **kw):
        try:
            return _ext(self, *a, **kw)
        finally:
            _CAP[0] = False

    cls.__enter__, cls.__exit__ = enter, exit_
    print("[dump] capture flag installed (a capture anywhere now suppresses every record)", flush=True)


if _DIR:
    try:
        _install()
    except Exception as e:      # never break the engine
        print(f"[dump] hook not installed: {type(e).__name__}: {e}", flush=True)
    if _ATTN_IO:
        try:
            _install_cap_flag()
        except Exception as e:
            print(f"[dump] capture flag not installed: {type(e).__name__}: {e}", flush=True)
        try:
            _install_attn_io()
        except Exception as e:
            print(f"[dump] attn IO hook not installed: {type(e).__name__}: {e}", flush=True)