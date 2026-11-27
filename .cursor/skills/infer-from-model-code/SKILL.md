---
name: infer-from-model-code
description: >-
  Decide which reference implementation's quantity to compare against by reading the
  model-structure source code, not by inverting the math on dumps. Use when aligning two
  implementations of a model (a custom kernel against SGLang/vLLM/transformers), when a
  per-layer comparison diverges and it is unclear which buffer, residual or coefficient
  pair corresponds, or when a dump is missing a layer because two stages were fused.
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
