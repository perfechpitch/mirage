"""Compare a full_decode.py dump with SGLang's decode-step dumps, every layer, every step.

The chain's per-sublayer residuals are the same quantity as the SGLang hook's `_apost`/`_mpost`:
`attn{L}_streams_out` is the residual right after layer L's attention, `moe{L}_streams_out` the one after its
MoE. For layers whose FFN post is fused into the next layer's boundary (0, 1, 14, 20) there is no `_mpost`, so
that residual has no reference and is skipped rather than fitted against a wrong tensor.

The sublayer OUTPUT `s` is compared DIRECTLY, not inverted: the hook records it as `_apost_x`/`_mpost_x` (the
`out` the post mixes: R'[i] = post[i] s + sum_j comb[j][i] R[j]) and the chain's `x_out` node is the same
tensor. The old code inverted the post through the largest-`post` stream because no such record existed; that
needs `post[j] > 2e-2` and silently reports nothing for the rest, so it is kept only as a cross-check on the
streams where it is well conditioned.

  python cmp_all.py --chain /tmp/chain_dec [--sgl /tmp/dec_coef6/layers] [--steps 10] [--layers 40]
  python cmp_all.py --chain /tmp/full_dec7 --sgl /tmp/dec_coef2/layers/step1        # one step, as before

A 10-step dump is the convention (see infer-from-model-code): a one-step match can be luck, and mHC's iterative
coefficients only get exercised once the window has moved.
"""
import argparse
import os
import re

import numpy as np

HC, HID, T = 4, 5120, 8


def sgl_vec(path):
    return np.fromfile(path, dtype=np.float32).astype(np.float64)


def chain_vec(path):
    a = np.fromfile(path, dtype=np.uint16)
    return ((a.astype(np.uint32) << 16).view(np.float32)).astype(np.float64)


def chain_f32(path):
    """A dump the chain wrote as fp32 (the mHC coefficients: [T, 32] = pre4|post4|comb16|0s). Reading those as
    bf16 produces finite-looking garbage (the packing shift mangles the exponent) and an inversion that explodes
    to 1e20 with no other warning."""
    return np.fromfile(path, dtype=np.float32).astype(np.float64)


def as_streams(v):
    """[T, HC*HID] (chain) or [HC*HID] (SGLang) -> [HC, HID] of the token's row.

    A [T, HID] (or [HID]) vector is layer 0's case: hc_broadcast puts the same embedding in every hc_mult stream,
    so the residual it starts from is that vector repeated HC times."""
    if v.size == T * HC * HID:
        v = v.reshape(T, HC, HID)[0]
    elif v.size == T * HID:
        v = np.tile(v.reshape(T, HID)[0], (HC, 1))
    elif v.size == HID:
        v = np.tile(v.reshape(HID), (HC, 1))
    return v.reshape(HC, HID)


def as_row(v):
    """[T, HID] (chain) or [HID] (SGLang) -> [HID] of the token's row."""
    return v.reshape(-1, HID)[0] if v.size % HID == 0 and v.size > HID else v.reshape(HID)


def cmp_streams(a, b):
    """(cos, relRMS, per-stream cos, per-stream relRMS) over the 4 streams.

    The per-stream numbers matter as much as the pooled one: mHC bookkeeping bugs move ONE stream, and a
    pooled relRMS over 4 streams dilutes a 5% single-stream error to 0.8% -- under any usable tolerance."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cos = (a * b).sum() / (na * nb).sum()
    rel_s = np.linalg.norm(a - b, axis=1) / nb
    return cos, np.linalg.norm(a - b) / np.linalg.norm(b), (a * b).sum(1) / (na * nb), rel_s


def cmp_row(a, b):
    """(cos, relRMS) of two hidden rows."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), float(np.linalg.norm(a - b) / np.linalg.norm(b))


def infer_s(resid, coef, Rin):
    """The chain's sublayer output `s`, from its own residual and coefficients:
        R'[j] = post[j] s + sum_i comb[i][j] R[i]   ->   s = (R'[j] - sum_i comb[i][j] R[i]) / post[j]
    only on the best-conditioned stream: at layer 0 post = [7e-4, 4.8e-4, 0, 0.136], so streams 0..2 carry ~1e-3
    of `s` and a "match" on them proves nothing. Returns None when even the best stream is ill conditioned.

    The chain's `x_out` node is NOT this quantity: it is rank 0's o_proj partial sum, before the all-reduce (norm
    8.9 vs the reference's 35.6 at layer 0), so comparing it directly reports a large error for a correct kernel."""
    post, comb = coef[4:8], coef[8:24].reshape(HC, HC)
    j = int(np.argmax(np.abs(post)))
    if abs(post[j]) < 2e-2:
        return None
    return (resid[j] - sum(comb[i, j] * Rin[i] for i in range(HC))) / post[j]


def compare_step(chain_dir, sgl_dir, layers, tol, quiet):
    """One decode step: (rows, worst_resid, worst_out). A row is (kind, L, resid_rel, out_rel or None)."""

    def c(name):
        p = os.path.join(chain_dir, name)
        return chain_vec(p) if os.path.exists(p) else None

    def s(name):
        p = os.path.join(sgl_dir, name)
        return sgl_vec(p) if os.path.exists(p) else None

    if not quiet:
        print(f"{'sublayer':10s} {'resid cos':>10s} {'resid relRMS':>13s} {'worst stream':>13s} "
              f"{'out cos':>9s} {'out relRMS':>11s}")
    rows = []
    for L in range(layers):
        # attn: the residual after it is this layer's _apost. The MoE's is _mpost, which the fused layers
        # (0/1/14/20) do not write -- their FFN post runs inside the NEXT layer's boundary -- so there is no
        # valid reference and the row is skipped instead of fitted against L{L+1}_in (a different tensor).
        for kind in ("attn", "moe"):
            # The MoE's residual is NOT `streams_out`: the engram rewrites it after the post (`streams_eg`), and the
            # chain follows that one (`full_decode.py`: `Rs = [wg["streams_eg"] if "streams_eg" in wg else ...]`,
            # and the next layer's x comes from `eg_next_x(src["streams_eg"], L)`). `moe{L}_streams_out` is the
            # pre-engram state, so comparing it with `L{L}_mpost` reports 0.59 for a MoE that is at 0.0057.
            # pre-engram throughout: the engram rewrites the residual AFTER the post (`streams_eg`), and the reference
            # records the pre-engram state, so `streams_out` is the buffer that lines up with it. (Measured: the
            # chain's post closes on `moe{L-1}_streams_out` at 1.3e-4 - 5e-4 and only at 9e-2 - 1.7e-1 on the
            # `_eg`/`attn{L-1}` alternative, so this is also the R the post actually used.)
            v = c(f"{kind}{L}_streams_out.bin")
            if v is None:
                continue
            # The reference's `L{L}_mpost` is the FFN post of layer L-1: the layer boundary fuses the MoE post into
            # the NEXT layer's forward, so it is recorded under that layer's id. Verified 11-27 on
            # moe2/3/4/5/10/20: vs L{L}_mpost 0.41-0.71, vs L{L+1}_mpost 0.024-0.11. (And `L{L}_in` is bit-identical
            # to `L{L-1}_apost`, i.e. the on-device residual still holds the previous attention post when the layer
            # starts -- the MoE's own write lands later.)
            suf = f"L{L:02d}_apost" if kind == "attn" else f"L{L + 1:02d}_mpost"
            w = s(f"{suf}.f32")
            if w is None or w.size != HC * HID:
                continue
            cos, rel, per_cos, per_rel = cmp_streams(as_streams(v), w.reshape(HC, HID))
            # the sublayer's own output: inverted from the CHAIN's residual and coefficients (all-reduced), then
            # compared with the reference's recorded `_apost_x`/`_mpost_x`. Only the chain side is inverted, so the
            # reference contributes no inversion noise. The MoE's is NOT used: `leave_output_sum=True` hands the post
            # its PIECES (unmerged), and `_as_tensor` picks one of them, so `_mpost_x` is a partial sum -- comparing
            # it with the chain's merged `x_out` reads 0.85 for a MoE whose residual is at 0.024.
            out_cos = out_rel = None
            cf = chain_f32(os.path.join(chain_dir, f"{kind}{L}_coef.bin")) \
                if os.path.exists(os.path.join(chain_dir, f"{kind}{L}_coef.bin")) else None
            xr = s(f"{suf}_x.f32") if kind == "attn" else None
            if cf is not None and cf.size >= 32 and xr is not None and xr.size == HID:
                # the residual this post mixed. Layer 0's attention is the special case: its post mixes the mHC
                # stream state, i.e. the RAW embedding in every stream -- SGLang records that as L00_in (verified:
                # it equals embed[4588] to 0.0), while the chain's attn0_x is the attention's POST-norm input. Using
                # attn0_x here inverts with the wrong R and reports cos 0.82 for a kernel that is at 0.9993.
                if kind == "attn" and L == 0:
                    Rin = s("L00_in.f32")
                    if Rin is None:
                        Rin = c("attn0_x.bin")      # fallback: self-consistent but the wrong quantity
                elif kind == "attn":
                    Rin = c(f"moe{L - 1}_streams_out.bin")
                else:
                    Rin = c(f"attn{L}_streams_out.bin")
                if Rin is not None and Rin.size >= HC * HID:
                    sch = infer_s(as_streams(v), cf[:32], as_streams(Rin))
                    if sch is not None:
                        out_cos, out_rel = cmp_row(sch, xr.reshape(-1)[:HID])
            worst_s = float(per_rel.max())
            ok = "ok " if rel < tol and worst_s < tol and (out_rel is None or out_rel < tol) else "BAD"
            oc = f"{out_cos:+9.5f}" if out_cos is not None else "        -"
            orr = f"{out_rel:11.4f}" if out_rel is not None else "          -"
            if not quiet:
                print(f"{kind}{L:<7d} {cos:+10.5f} {rel:13.4f} {worst_s:13.4f} {oc} {orr}  {ok}")
            rows.append((kind, L, rel, out_rel, worst_s))
    worst_r = max((r[2] for r in rows), default=0.0)
    worst_s = max((r[4] for r in rows), default=0.0)
    outs = [r[3] for r in rows if r[3] is not None]
    # None (not 0.0) when nothing was measured: a zero here would read as "the outputs match exactly",
    # which is the same class of false pass as a NaN-swallowing max().
    worst_o = max(outs) if outs else None
    return rows, worst_r, worst_s, worst_o


def step_dir(root, k):
    """SGLang writes <root>/step{k}; the chain's flat dump directory IS its step 1."""
    d = os.path.join(root, f"step{k}")
    return d if os.path.isdir(d) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, help="full_decode --dump-dir (flat = step 1, then step2..)")
    ap.add_argument("--sgl", default="/tmp/dec_coef6/layers",
                    help="SGLang dump 'layers' directory (holds step1..) or one step directory")
    ap.add_argument("--steps", type=int, default=1, help="compare step1..stepN (0: every step present)")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-2, help="relRMS counted as aligned")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    # --sgl may be the layers root, one step directory (stepN), or a directory holding layer rows directly.
    # The order matters: a dump's layers root ALSO holds the prefill's flat rows (L00_in, L00_apost, ...), so a
    # "has L00_apost.f32" test alone silently compares the chain's step 1 against the PREFILL row -- a different
    # token -- and reports a uniform ~1.3 relRMS that looks like a broken kernel. Prefer stepN when present.
    sd = a.sgl.rstrip("/")
    has_steps = any(os.path.isdir(os.path.join(sd, f"step{k}")) for k in range(1, 100))
    if re.search(r"[/\\]step\d+$", sd) or (not has_steps and os.path.exists(os.path.join(sd, "L00_apost.f32"))):
        m = re.search(r"step(\d+)$", sd)
        steps = [(int(m.group(1)) if m else 1, a.chain, sd)]
        if not m:
            print(f"{sd}: no stepN directories, so this is read as ONE step directory (steps 1.. are prefixed "
                  f"in the dump's layers root)")
    else:
        avail = [k for k in range(1, 100) if step_dir(a.sgl, k)]
        want = avail if a.steps == 0 else [k for k in avail if k <= a.steps]
        if not want:
            raise SystemExit(f"{a.sgl}: no step1.. present")
        steps = []
        for k in want:
            cd = a.chain if k == 1 else os.path.join(a.chain, f"step{k}")
            if not os.path.isdir(cd):
                continue
            steps.append((k, cd, os.path.join(a.sgl, f"step{k}")))
    if not steps:
        raise SystemExit(f"{a.chain}: nothing to compare (flat step 1 missing, or --steps beyond what was dumped)")

    summary = []
    for k, cd, sd in steps:
        if not a.quiet and len(steps) > 1:
            print(f"\n=== decode step {k}  (chain {cd} vs SGLang {sd})")
        rows, wr, ws, wo = compare_step(cd, sd, a.layers, a.tol, a.quiet)
        if not rows:
            # a present-but-empty step directory must not read as "0/0 aligned": that both hides a broken
            # dump and poisons the trajectory below.
            print(f"step {k:2d}: SKIP -- no comparable rows (chain {cd} has no *_streams_out/x_out, "
                  f"or SGLang {sd} has no L*_apost/_mpost)")
            continue
        bad = [r for r in rows if r[2] >= a.tol or r[3] is not None and r[3] >= a.tol or r[4] >= a.tol]
        first = f"{bad[0][0]}{bad[0][1]}" if bad else "-"
        summary.append((k, rows, wr, ws, wo, len(bad), first))
        print(f"step {k:2d}: {len(rows) - len(bad)}/{len(rows)} aligned (<{a.tol}), worst resid {wr:.4f}, "
              f"worst stream {ws:.4f}, worst out {'-' if wo is None else f'{wo:.4f}'}, first bad {first}")

    print(f"\n{'step':>4s} {'aligned':>9s} {'worst resid':>12s} {'worst strm':>11s} {'worst out':>10s} "
          f"{'first bad':>10s}")
    for k, rows, wr, ws, wo, nb, first in summary:
        print(f"{k:4d} {len(rows) - nb:4d}/{len(rows):<4d} {wr:12.4f} {ws:11.4f} "
              f"{'-' if wo is None else f'{wo:10.4f}'} {first:>10s}")
    # a trajectory matters: an error constant in absolute terms across steps is a bias, one that grows is a
    # decode-loop bug (the window moving under a wrong table), and only the second needs the loop examined.
    if len(summary) > 1:
        drift = [s[2] for s in summary]
        print(f"\nworst-residual trajectory: " + " ".join(f"{v:.4f}" for v in drift))
        print(f"{'growing' if drift[-1] > drift[0] * 1.2 + 1e-4 else 'stable'} "
              f"(first {drift[0]:.4f} -> last {drift[-1]:.4f})")


if __name__ == "__main__":
    main()
