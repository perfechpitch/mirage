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
VENV=${VENV:-/opt/dlami/nvme/saboxu/venvs/sglang/bin}
OUT=${OUT:-/tmp/tpbar_verify}
export PATH=$VENV:$PATH
mkdir -p "$OUT"

# the sglang venv has torch + torchrun; e2e_decode imports no sglang, only torch and the layer module
export DSV41_E2E_LAYERS=$LAYERS
export DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_POOL=1
export DSV41_STATIC_MHC=1 DSV41_STATIC_AR=0 DSV41_STATIC_TPBAR=0
export DSV41_STATIC_MAX_POS=33000

run() {   # run <tag> <out-dir> <env...>
  local tag=$1 dir=$2; shift 2
  echo "=== $tag"
  env "$@" torchrun --nproc_per_node=8 e2e_decode.py --checkpoint "$CKPT" --row "$ROW" \
      --layers "$LAYERS" --reps "$REPS" --warmup "$WARM" --out "$dir" > "$OUT/$tag.log" 2>&1
  local rc=$?
  local verdict
  if grep -q "NOT FINITE" "$OUT/$tag.log"; then
    verdict=$(grep -m1 -o "NOT FINITE.*" "$OUT/$tag.log")
  elif grep -q "x_out finite" "$OUT/$tag.log"; then
    verdict=$(grep -m1 -o "all .*x_out finite.*" "$OUT/$tag.log")
  else
    verdict="no verdict line (rc $rc)"
  fi
  echo "    rc=$rc  $verdict"
  echo "    $OUT/$tag.log"
}

run tpb1            "$OUT/build_tpb1" DSV41_STATIC_TPB=1
run tpb0_hostsync   "$OUT/build_tpb0" DSV41_STATIC_TPB=0
run tpb0_nosync     "$OUT/build_tpb0" DSV41_STATIC_TPB=0 DSV41_E2E_NOSYNC=1

echo
echo "=== summary"
for t in tpb1 tpb0_hostsync tpb0_nosync; do
  printf '%-16s %s\n' "$t" "$(grep -m1 -oE "NOT FINITE.*|all [0-9]+ attn.*finite.*" "$OUT/$t.log" || echo '-')"
done
echo
echo "expected: tpb1 finite, tpb0_hostsync finite, tpb0_nosync NOT FINITE (the 11-25 NaN, reproduced on purpose)"
