"""Trace stock SGLang's DeepSeek-V4: every decoder layer's output, then the LM head's logits.

Nothing static runs here. No Mirage kernel, no build, no check mode: this is SGLang's own model, patched
on entry only to read values back to the host and print them. It answers "what does SGLang actually
compute, layer by layer", which is the reference a comparison needs in the first place.

    DSV41_TRACE=1 python l2_check.py --model <ckpt> --row 0 --tokens 1

DSV41_TRACE_STEPS (default 3): trace this many forward steps, then stop reading (the model keeps running).
DSV41_TRACE_N (default 8): values printed per tensor.

One line a sublayer, on rank 0 only:

    trace L07 out [1, 5120] norm 12.34 max|123|+4.5678 non-finite 0/5120 head[+0.1 +0.2 ...]
    trace head logits [1, 129280] argmax 12345 (val +12.5) top5 [[12345,12.5],...] norm ...

The norm/head/max are on the same flattening a comparison would use, so a NaN's first index and a
permutation both show up in these lines without opening a dump.
"""
import os
import sys

import torch

STEPS = int(os.environ.get("DSV41_TRACE_STEPS", "10"))
N = int(os.environ.get("DSV41_TRACE_N", "8"))
# DSV41_TRACE_DIR: every traced tensor in full, not just the N printed values. A line answers "what does layer 7's
# output look like"; the file is what an offline comparison needs (which columns differ, is it a permutation, does the
# ratio track the magnitude) without a second GPU run. float32: a bf16 value is exact in fp32, and numpy has no bf16.
TRACE_DIR = os.environ.get("DSV41_TRACE_DIR")
_logged = {"step": 0, "lines": 0}


def _dump(layer_id, what, t) -> None:
    if not TRACE_DIR:
        return
    try:
        import numpy as np
        d = os.path.join(TRACE_DIR, f"step{_logged['step']}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"L{layer_id:02d}_{what}.npy"),
                t.detach().float().cpu().contiguous().numpy())
    except Exception as e:   # a dump must never end a run that is otherwise fine
        _say(f"dump L{layer_id:02d} {what} failed: {type(e).__name__}: {e}")


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _is_capture() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _say(msg: str) -> None:
    if _rank() != 0 or _is_capture():
        return
    _logged["lines"] += 1
    print(f"[trace] {msg}", flush=True)


def _is_decode(forward_batch) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return True
    try:
        return bool(mode.is_decode())
    except Exception:
        try:
            return bool(mode.is_decode_or_idle())
        except Exception:
            return True


def _tensor_of(v):
    """A decoder layer's output as one tensor: the CUDA path returns one; a boundary path can return a
    state whose residual/pre is the tensor to read."""
    if torch.is_tensor(v):
        return v
    for name in ("residual", "rows", "streams", "hidden_states"):
        x = getattr(v, name, None)
        if torch.is_tensor(x):
            return x
    return None


def _nums(v, n=N):
    f = v.detach().float().flatten()
    head = " ".join(f"{x:+.4f}" for x in f[:n].tolist())
    n_bad = int((~torch.isfinite(f)).sum())
    if n_bad:
        return f"norm nan({n_bad}/{f.numel()} non-finite, first at {int((~torch.isfinite(f)).nonzero()[0])}) head[{head}]"
    i = int(f.abs().argmax())
    return f"norm {f.norm():.6g} max|{i}|{f[i]:+.4g} non-finite 0/{f.numel()} head[{head}]"


def _trace_layer(layer_id, out, forward_batch):
    if STEPS and _logged["step"] >= STEPS:
        return
    t = _tensor_of(out)
    if t is None:
        _say(f"L{layer_id:02d} out <no tensor: {type(out).__name__}>")
        return
    _say(f"L{layer_id:02d} out {list(t.shape)} {_nums(t)}")
    _dump(layer_id, "out", t)


def _trace_logits(t):
    am = int(t.detach().float().flatten().argmax())
    v = t.detach().float().flatten()
    k = min(5, v.numel())
    top = [[int(i), round(float(v[i]), 4)] for i in v.topk(k).indices.tolist()]
    _say(f"head logits {list(t.shape)} argmax {am} (val {v[am]:+.5g}) top{k} {top} | {_nums(t)}")


def _patch_layers() -> None:
    from sglang.srt.models import deepseek_v4 as dv4

    for cls_name in ("DeepseekV4CudaDecoderLayer", "DeepseekV4PlatformDecoderLayer"):
        cls = getattr(dv4, cls_name, None)
        if cls is None or getattr(cls.forward, "_dsv41_trace", False):
            continue
        orig = cls.forward

        def forward(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            fb = kwargs.get("forward_batch")
            if fb is None and len(args) > 3:
                fb = args[3]
            if fb is not None and _is_decode(fb):
                _trace_layer(getattr(self, "layer_id", -1), out, fb)
            return out

        forward._dsv41_trace = True
        cls.forward = forward


def _patch_logits() -> None:
    from sglang.srt.layers.logits_processor import LogitsProcessor

    orig = LogitsProcessor._get_logits
    if getattr(orig, "_dsv41_trace", False):
        return

    def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
        out = orig(self, hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer)
        fb = getattr(logits_metadata, "forward_batch", None)
        if (fb is None or _is_decode(fb)) and (not STEPS or _logged["step"] < STEPS):
            _trace_logits(out)
        return out

    _get_logits._dsv41_trace = True
    LogitsProcessor._get_logits = _get_logits


def _patch_step_counter() -> None:
    """A decode step is one model forward: count them where SGLang starts one, so the per-layer lines can
    be capped and a run is readable instead of forty lines a token for the whole generation."""
    from sglang.srt.model_executor import model_runner

    cls = getattr(model_runner, "ModelRunner", None)
    if cls is None or getattr(cls.forward, "_dsv41_trace", False):
        return
    orig = cls.forward

    def forward(self, *args, **kwargs):
        fb = kwargs.get("forward_batch")
        if fb is None and args:
            fb = args[0]
        if fb is not None and _is_decode(fb):
            if STEPS and _logged["step"] >= STEPS:
                _logged["step"] += 1
                return orig(self, *args, **kwargs)
            _say(f"--- step {_logged['step']} ---")
            _logged["step"] += 1
        return orig(self, *args, **kwargs)

    forward._dsv41_trace = True
    cls.forward = forward


def install() -> None:
    _patch_layers()
    _patch_logits()
    _patch_step_counter()
    print(f"[trace] stock SGLang trace on (steps {STEPS}, {N} values a tensor, rank 0)", flush=True)


install()
