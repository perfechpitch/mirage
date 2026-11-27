---
name: infer-from-model-code
description: >-
  Decide which reference implementation's quantity to compare against by reading the
  model-structure source code, not by inverting the math on dumps. Use when aligning two
  implementations of a model (a custom kernel against SGLang/vLLM/transformers), when a
  per-layer comparison diverges and it is unclear which buffer, residual or coefficient
  pair corresponds, when a dump is missing a layer because two stages were fused, or when
  generating the reference decode dump (which is always 10 decode tokens, steps 1..10).
---

# Infer from the model code, not from the numbers

When comparing two implementations of the same model, the mapping between their intermediate
values must be established from the code that produces them. Inverting the equations on dumps
is only admissible **after** the code has fixed what each quantity is.

## Why

Every alignment failure on DeepSeek-V4.1 in this repo came from an inverted inference that
looked consistent and was wrong:

| Symptom | Inferred explanation | What the code actually said |
|---|---|---|
| L0 attention off by 13.5%, cos 0.991 | "the sparse attention kernel is wrong" | `mhc._combine` starts with `if apply_pre is None: return norm(x[:, 0, :])` -- layer 0's input is `RMSNorm(embed) * attn_norm.weight`, and the 12% per-channel spread of `gamma` explains exactly the magnitude seen |
| L1 coefficients matched no recorded triplet | "the dump is filed under the wrong layer" | the FFN post of layers 0/1/14/20 is fused into the next layer's boundary, so `run_moe_post` never runs for them at all |
| FFN residuals off by 60% | "the mHC coefficients are miscomputed" | the coefficients equal the official formula to 5.7e-7 on the same residual; the residuals differed because the *input* differed |
| `moe0_streams_out` all zeros | "the dump path is broken" | `"h_comb" in w` is the dispatch key meaning "the framework runs the post", not a place to put a coefficient |

Cost: several GPU runs and one 40-minute rebuild, all avoidable by reading ~200 lines first.

## Procedure

1. **Locate the reference path before comparing anything.**
   Find the class and method that produce the value you intend to compare, and read until the
   data source is a concrete tensor expression. For SGLang that is usually
   `models/<model>.py` (the layer's `forward`) and `models/<model>_mhc.py` /
   `models/<model>_boundary.py` (the residual plumbing).

2. **Write down the chain, one line per hop, with the variable names from the code.**
   ```text
   layer.forward:
     h  = attn_boundary.prepare(hidden_states, ...)
     h  = self_attn(x=h, ...)
     h  = attn_boundary.finish(h, ...)
     h  = ffn_boundary.prepare(h, ...)      <- THIS h is the FFN's input residual
     ...
   ```
   If a hop is delegated to a helper object (`prepare`/`finish`/`update_and_read`), follow it.
   Do not stop at the layer's `forward`.

3. **Check what a skipped call means.**
   A function you expect to see and don't (no `run_moe_post` for some layers) is a finding, not
   a gap in the dump: fused stages run inside the next stage. Grep for the fusion flag
   (`allow_cross_layer_fusion`, `next=` on the post, `*_boundary`) and read the fused kernel's
   signature. Then decide which dump file holds that quantity, or add an instrument.

4. **Resolve identifier ambiguity from the source, not from values.**
   When a branch is keyed on a tensor's presence (`if "h_comb" in w:`, `if quantized is not None:`),
   that presence is a *mode switch*. Check every site of the key before adding or removing it:
   `rg '"<key>" in' `. A key that reads like a place to write a value is often a flag.

5. **Only then compare numbers.** If a comparison still fails, the state of knowledge is "one of
   my code-derived assumptions is wrong" -- go back to step 1 with the specific assumption, not to
   a wider numerical sweep.

6. **Prefer a recorded quantity to an inverted one.** If the source of a value is cheap to
   instrument (a hook on the call that produces it), record it. Inverting `R' = post*s + sum comb*R`
   for `s` needs a well-conditioned `post[j]`: at layer 0 three of the four streams have
   `post ~ 1e-4`, so a "match" on those streams proves nothing about `s`.

## Signs you are inverting instead of reading

- You are sweeping candidate input pairs and picking the one with the smallest fit residual.
- You are naming a tensor by the size of its norm.
- Your explanation needs a coefficient to differ, but you have not printed the coefficient from
  the formula on the *same* input.
- A conclusion depends on a `post[j]` below ~2e-2.

When any of these appear, stop and read the producing code.

## Validation that a comparison is well-posed

Before trusting a per-layer number, check that the reference and the chain agree on something
that must be identical by construction:

- the sublayer's **input** (`x`), computed on the host from the reference's own residual and
  norm weight, e.g. `coef_ref.py` / `normed_row`;
- the coefficients from the **same** residual -- `coef_ref.py --residual <chain's>` against the
  kernel's own `*_coef.bin` must agree to ~1e-6 (bf16), or the formula, not the data, is wrong.

If those two agree, a later divergence is a real difference in the sublayer's output and can be
attacked directly. If they don't, the comparison itself is still the bug.

## The reference decode dump is 10 tokens, always

A decode reference is **ten decode steps**, not one. Use this command shape (the model's own
`forward` runs eagerly, not under a graph, or the probes are no-ops or the capture dies):

```bash
cd demo/deepseek_v4_1_sglang
PATH=$SGLANG_VENV/bin:$PATH PYTHONPATH=$PWD/boot \
SGLANG_DSV4_KV_LAYOUT=v41_new \
SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS=0 \
DSV41_DUMP_PREFILL=<dir> DSV41_DUMP_LAYERS=1 DSV41_DUMP_STEPS=10 \
  python dump_prefill.py --model <ckpt> --row 0 --out <dir> --max-new 11
```

- `SGLANG_DSV4_KV_LAYOUT=v41_new` is **mandatory and silent when missing.** With the default
  `auto` on the trtllm path the pool comes out as the plain `V4` layout (584 B/token here) instead
  of `v41_new` (528 B/token). The dump still writes 12 GB of pools and every layer row; only the
  static kernel's row size reveals it, and the numbers are not comparable. **Check before using a
  dump**: `<dir>/meta_addr_rank00.json` -> `pool.kv_layout` must be `v41_new` (and
  `swa_bytes_per_token` the kernel's `demo.ROW`). `full_decode.py`'s `pool_rows` now asserts this
  with the layout name in the message.
- The decode step's input token is **the token the prefill sampled**, i.e. `golden.ids[0]` =
  4588, at position `len(prompt)` = 31906 -- not `prompt_ids[pos]`. Verified: `step1/L00_in`
  stream 0 equals `embed.weight[4588]` with maxdiff 0.0. (`decode_token` used to return
  `ids[a.pos]`, which is both wrong and an IndexError at `pos == len(prompt)`.)

- `--max-new 11` for 10 recorded steps: `dump_hook.py` counts every decode forward and stops at
  `DSV41_DUMP_STEPS=10`; the extra token is headroom so a warmup/dummy forward (a one-token
  forward that also walks all 40 layers) cannot eat a real step. **Check before comparing**:
  `ls <dir>/layers/step*` must be exactly `step1 ... step10`.
- `--max-new 1` is only enough to prove which token the prefill hands over; it produces a single
  decode step, so it cannot separate a persistent bias from a one-off, and it leaves nothing to
  check that the chain is stable step to step.

Why ten and not one:

1. **A one-step match can be luck.** With ten steps a divergence that is present at step 1 and
   grows (or is constant in absolute terms) is distinguishable from one that only appears once the
   KV window has moved -- and only the second is a decode-loop bug.
2. **mHC's sinkhorn is iterative across the residual's stream axis.** Coefficients at step 1 are
   the converged-ish case; steps 2..10 are where a wrong stream bookkeeping shows up.
3. **Per-step cheapness is asymmetric.** The prompt (prefill) is ~1 hour of the run; ten decode
   steps add seconds. Re-running for one extra step costs another engine load.
4. **Aligning "the whole 40 layers" needs a second axis.** Layer-by-layer error is one axis;
   step-by-step drift is the other. Fixing an error at L17 that is really a step-0 artifact is
   easy to do with one step and hard to notice.

Consumers (`cmp_all.py`) should loop over `step1..step10` and report per-layer error as a
*trajectory*, and quote the worst step, not just step 1.

### The chain side must produce the same ten steps

The reference having ten steps is useless if the chain replays one. In `full_decode.py`,
`--steps N` alone re-launches the **same** token at the **same** position (nothing advances
`positions`, and the layer's `x`/`streams` never change), so the N outputs are identical and
prove nothing. A real multi-token chain is:

```bash
python full_decode.py --checkpoint <ckpt> --inputs-dir <prefill-dump> --token <t> \
  --pos <p> --steps 9 --greedy --dump-dir /tmp/chain_dec
```

- `--steps 9` because the flat dump directory *is* step 1: `--steps 9` writes the flat dir plus
  `step2 ... step10`, matching SGLang's `step1 ... step10`.
- `--greedy` does the two things a decode loop needs: `advance(k)` moves every rank's
  `positions` (and the window rows that are a pure function of them) to `pos+k`, and
  `feed_back(k)` rewrites the layer-0 input in place -- `normed_row(embed(tok), 0)` and the same
  embedding in every `hc_mult` stream -- with step k's own argmax. Without both, a "10-step"
  dump is ten copies of one token.
## Layers 1 and 14 have an Engram: the residual is REPLACED there

SGLang's decoder loop does, before layer `i` (`deepseek_v4.py:4360`):

```python
engram = self.layers[i].engram
if engram is not None:
    hidden_states = engram(state.residual, hash_ids[:, engram.layer_hash_index], forward_batch, ...)
    state = state.with_residual(hidden_states)
```

`text_config.engram_layer_ids = [1, 14]`. So the residual that layer 1 (and 14) reads is **not**
the previous MoE's R' -- it is `engram(R', hash_ids)` = R''. Two consequences:

1. A chain comparison that skips the Engram is comparing a *different model* from L1 on. `moe0`'s
   R' matching SGLang's `L01_in` on streams 2 and 3 while streams 0 and 1 are off by 2x is the
   signature: `engram_gate` adds the **same** value vector to all four streams, scaled by a
   per-(token, stream) gate, so a large gate shows up on one or two streams only.
2. The gate's only inputs are `x` (the residual) and `emb` (the gathered n-gram rows,
   `[T, 3 n-grams x 8 heads x 256] = [T, 6144]`). `emb` comes from the hash table -- treat it as
   an **input from the dump**, like the KV pools, not something the chain re-derives.

In `full_decode.py` the gate lives in the MoE kernel of layer `L-1` (`moe_demo.engram_nodes`,
DSV41_ENGRAM=1): it writes R'' to `streams_eg`, so the chain must hand `streams_eg` (not
`streams_out`) to L1/L14. `eg_gamma`/`DSV41_EG_X` (the kernel computing the next attention's
input from R'') only works with **DSV41_ENGRAM=2**, where kv is a graph tensor from the attention
kernel of the layer before; with ENGRAM=1 `engram_gate_layer` asserts. Until that is wired, the
next layer's `x` is recomputed on the host from R'' (`coef_ref.triplet` -> `sum_c pre_c R''_c` ->
RMSNorm -> `attn_norm.weight`).

### Reading the Engram's fp8 weights offline (two traps)

To check `engram_gate` against a dump by hand:

```python
se = wkv_scale.float().view(R // 32, C // 32)        # e8m0: .float() IS ALREADY 2**(e-127)
wd = (wkv.float().view(R // 32, 32, C // 32, 32) * se[:, None, :, None]).reshape(R, C)   # no permute
kv = emb @ wd.T                                       # [ (hc+1) H ]; key = kv[:hc*H], value = kv[hc*H:]
```

- Do **not** subtract 127 from `scale.float()`: it yields ~5.9e-39, `key` becomes ~0, every gate lands on
  exactly 0.5, and the residual looks untouched -- a plausible-looking wrong answer.
- Do **not** `.permute(0,2,1,3)` the block tensor before reshaping: `[br, r, bc, c]` is already the row-major
  `[R, C]` order. Permuting scrambles the blocks, decorrelates the output from the reference (cos ~0.08) and
  still leaves the magnitudes believable.
- The gate is *small*: at L1 it is `[0.021, 0.080, 0, 0]`, at L14 `[0.003, 0.004, 0.003, 0.060]`, i.e. streams 2
  and 3 of L1 are untouched. Expect `R''` to be a few percent from `R'`, never a rewrite.
- `Engram.embed_rows` is idempotent, so a probe may call it a second time to record the rows (verified: the rows
  are identical either way). Use the rows the dump records rather than re-deriving the hash lookup.

### The chain dump's bit width: streams are bf16-as-int16, coefficients are fp32

`full_decode.py`'s `dump_chain` writes every tensor with `.view(torch.int16)`, so `<name>_streams_out.bin`,
`<name>_x_out.bin` and `<name>_streams_eg.bin` are **2 bytes a value**, while `<name>_coef.bin` is fp32 (4
bytes). The buffer is `[T, HC*H]` with `T` = the static token count (8 here), not 4. Read them as

```python
torch.from_numpy(np.fromfile(f, dtype=np.int16)).view(torch.bfloat16).float().view(8, HC, H)
```

Reading a stream file as `float32` pairs two bf16 values into one garbage float and, because the element count
halves, mis-slices the token axis too. It looks like a real result -- two of four streams come out as exactly
`0.0` and are bit-identical between runs -- and it will send you chasing a "dump artefact" that does not exist.
Confirmed the hard way (11-27): with the right dtype, `moe0_streams_eg` at step 1 is cos 0.99966-0.99999 against
SGLang's `L01_in` per stream, i.e. the Engram wiring was correct all along.
