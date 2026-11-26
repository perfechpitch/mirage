#!/bin/bash
# The TP-barrier verification, one command: does -DSTATIC_TP_BARRIER (core.cuh kernel_begin) remove the NaN the
# pure serving chain returned, and does turning it off bring the NaN back?
#
# What is compared (all on the *serving* build, -DSTATIC_RESET_IN_KERNEL, one rank per GPU, no SGLang):
#   1  TPB=1                  the kernel lines the 8 ranks up before its resets   -> expect every x_out finite
#   2  TPB=0, host barrier    nothing in the kernel, dist.barrier() between launches -> expect finite (the safe
#                             baseline, and what e2e_decode did before 11-26: it pays a host round trip per sublayer)
#   3  TPB=0, no barrier      the 11-25 configuration                              -> expect NOT FINITE from the first
#                             Reuse layer (L3 or L4, by timing); this is the negative control
#
# Runs 2 and 3 share the TPB=0 build (only the runtime env differs); 1 needs its own, since the flag is compiled in.
# Layers: DSV41_E2E_LAYERS=6 is enough for the control (the break is at L3/L4) and builds 12 instead of 80 kernels;
# leave it unset for the real 40-layer run.
#
#   ./verify_tpbar.sh
#   DSV41_E2E_LAYERS=40 ./verify_tpbar.sh
set -u
cd "$(dirname "$0")"

CKPT=${CKPT:-/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017}
LAYERS=${DSV41_E2E_LAYERS:-6}
REPS=${REPS:-2}
WARM=${WARM:-1}
ROW=${ROW:-0}
VENV=${VENV:-/opt/dlami/nvme/saboxu/venvs/mirage/bin}
OUT=${OUT:-/tmp/tpbar_verify}
export PATH=$VENV:$PATH
# the planner's z3, as full_decode needs it (FINDINGS §8.2); absent, the build still runs from the source tree
export LD_LIBRARY_PATH=/opt/dlami/nvme/saboxu/venvs/mirage/lib/python3.12/site-packages/z3/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$OUT"

# what e2e_decode reads: DSV41_E2E_LAYERS, DSV41_E2E_MHC, DSV41_STATIC_TPB / DSV41_E2E_NOSYNC (the barrier pair),
# and STATIC_TOKENS (its T; 1, the script's own default). DSV41_STATIC_ATTN/MOE/POOL belong to the SGLang hook and
# mean nothing here -- set them and a run would look configured while none of it took effect.
export DSV41_E2E_LAYERS=$LAYERS
export DSV41_E2E_MHC=1            # the model has hc_mult=4: mHC off is a diagnosis, not the model
export STATIC_TOKENS=1
# one step and the verdict, no timing loop: the question here is finite or not, and a chain with no barrier spins in
# the *next* launch (measured: the 11-25 configuration printed NOT FINITE from attn L3, then hung in the warmup step)
export DSV41_E2E_ONE=${DSV41_E2E_ONE:-1}

run() {   # run <tag> <out-dir> <env...>
  local tag=$1 dir=$2; shift 2
  echo "=== $tag"
  # a timeout, not just patience: with no barrier at all the 11-25 failure could be a spin (hc_post_combine's 4
  # CTAs never end) rather than a NaN, and a hung torchrun would eat the whole window
  timeout "${TIMEOUT:-900}" env "$@" torchrun --nproc_per_node=8 e2e_decode.py --checkpoint "$CKPT" --row "$ROW" \
      --layers "$LAYERS" --reps "$REPS" --warmup "$WARM" --out "$dir" > "$OUT/$tag.log" 2>&1
  local rc=$?
  local verdict
  if [ $rc -eq 124 ]; then
    verdict="TIMEOUT (${TIMEOUT:-900}s) -- it spun, which is the drift too"
  elif grep -q "NOT FINITE" "$OUT/$tag.log"; then
    verdict=$(grep -m1 -o "NOT FINITE.*" "$OUT/$tag.log")
  elif grep -q "x_out finite" "$OUT/$tag.log"; then
    verdict=$(grep -m1 -o "all .*x_out finite.*" "$OUT/$tag.log")
  else
    verdict="no verdict line (rc $rc)"
  fi
  echo "    rc=$rc  $verdict"
  echo "    $OUT/$tag.log"
}

# the control first: it builds the TPB=0 set, which tpb0_hostsync then reuses, and its answer is the one that says
# whether the barrier is the difference at all.
# The control also carries the buffer census: DSV41_RESET_DEBUG=1 has each layer print its reset entries' addresses
# to stderr, DSV41_E2E_DBGLOG points e2e_decode at that captured stderr, and it reads every one back in its own
# process after the step (buffers_rank{R}.txt). That answers the one question still open -- whether the NaN sat in a
# buffer the layer fills with 0xFF each launch (a local sentinel read early) or in an exchange buffer a peer never
# published this launch (FINDINGS 4.3). A NaN count equal to bytes/4 is "still the fill"; a smaller one with a
# nonzero first-NaN index is "a real NaN was pushed in".
export DSV41_RESET_DEBUG=1
export DSV41_E2E_DEBUG=1
export DSV41_E2E_DBGLOG=$OUT/tpb0_nosync.log   # run() sends stdout+stderr there
run tpb0_nosync     "$OUT/build_tpb0" DSV41_STATIC_TPB=0 DSV41_E2E_NOSYNC=1
unset DSV41_RESET_DEBUG DSV41_E2E_DEBUG DSV41_E2E_DBGLOG
run tpb1            "$OUT/build_tpb1" DSV41_STATIC_TPB=1
run tpb0_hostsync   "$OUT/build_tpb0" DSV41_STATIC_TPB=0

echo
echo "=== summary"
for t in tpb1 tpb0_hostsync tpb0_nosync; do
  printf '%-16s %s\n' "$t" "$(grep -m1 -oE "NOT FINITE.*|all [0-9]+ attn.*finite.*" "$OUT/$t.log" || echo '-')"
done
echo
echo "expected: tpb1 finite, tpb0_hostsync finite, tpb0_nosync NOT FINITE (the 11-25 NaN, reproduced on purpose)"
if [ -f "$OUT/build_tpb0/buffers_rank0.txt" ]; then
  echo
  echo "=== the control's buffer census (rank 0), which buffer held the NaN:"
  grep -E "NaN [1-9]" "$OUT/build_tpb0/buffers_rank0.txt" | head -12
  echo "  (full list: $OUT/build_tpb0/buffers_rank0.txt)"
else
  echo
  echo "no buffer census: the control's DEBUG block did not run (see $OUT/tpb0_nosync.log)"
fi
