#!/bin/bash
# Layer by layer: at each sublayer, SGLang's own post and the static kernel's, on the same input, with the real
# weights and the real 32K pools. The point is to answer "which layer" instead of "the 40-layer output is wrong".
#
# Two runs, and the difference between them is the whole experiment:
#   sync  TPBAR=1  a host barrier after every sublayer. The ranks cannot drift, so this must be clean; it proves the
#                  comparison machinery itself is not what reports a failure. (This is also what today's defaults do.)
#   kern  TPBAR=0  no host barrier at all: the kernel's own start barrier (TPB=1) is the only thing lining the ranks
#                  up -- the strong question, and the configuration that goes NaN in replace mode.
# If sync is clean and kern breaks at one sublayer, that sublayer is the one the drift damages, and the layers above
# it are correct -- which is exactly what the goal needs: L0-L3 finite, then the first reuse layer.
#
# Needs nearly a whole card per rank (measured: 111 GB free after the 40-layer build at 32K, so ~164 GB in use plus
# SGLang's own pools), and the 32K prefill wants the real bench32k prompt.
#
#   ./layerwise.sh              # both runs
#   ONLY=sync ./layerwise.sh    # one of them
#   ROW=3 STEPS=3 ./layerwise.sh
set -u
cd "$(dirname "$0")"

CKPT=${CKPT:-/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017}
ROW=${ROW:-0}
TOKENS=${TOKENS:-1}          # prefill + this many decode steps; step 1 is where the failure was measured
STEPS=${STEPS:-3}            # DSV41_STATIC_CHECK_STEPS: printed comparison lines per sublayer
OUT=${OUT:-/tmp/layerwise}
VENV=${VENV:-/opt/dlami/nvme/saboxu/venvs/sglang/bin}
ONLY=${ONLY:-both}
# the KV cache share. The 40-layer static build is what wants the memory (measured: ~164 GB/rank in use at
# 32K with the default 0.40); raise this only if SGLang asks for less KV, never to make room for weights.
MEMFRAC=${MEMFRAC:-}
# DETAIL=1 prints the eight values behind each relRMS; DUMPDIR=<dir> writes the whole tensors (.npy, both sides, with
# a manifest of the relRMS). Both default off; DUMPDIR is what you want when the question is "what exactly differs",
# and `python dump_layers.py --dir <dir>` reads it back with no GPU.
DETAIL=${DETAIL:-}
DUMPDIR=${DUMPDIR:-}
# The planner needs the mirage venv's z3 (4.16); the sglang venv's is 4.15 and sitecustomize dies on it with
# "ImportError: libz3.so.4.16", which does NOT stop the run -- it silently runs stock SGLang instead. That is exactly
# the kind of quiet non-measurement the assertion below is for.
export LD_LIBRARY_PATH=/opt/dlami/nvme/saboxu/venvs/mirage/lib/python3.12/site-packages/z3/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$OUT"

# the two variables that decide whether the chain under test is the one that failed (see FINDINGS_correctness 8.3):
#   MHC=1    the sublayer's end -- the all-reduce over the exchange region -- inside the kernel
#   CHECK=1  SGLang's post runs too, on the same input, and the two are compared
# AR=0 is required by check mode (it compares this rank's partial y, so the all-reduce must not be added by SGLang).
# DSV41_STATIC_ROWS=1: the build's T is 8, and a decode step of 1 row is rows < T, which the hook refuses to run
# statically ("decode step of 1 rows NOT checked -- rows<st.T=True"). That rejection is silent in the summary -- the
# first two runs of this script came back "1 sublayer comparisons printed" because the only comparison was a prefill
# leftover. ROWS=1 gives the step a build of its own live rows, which is what --batch 1 needs.
run() {   # run <tag> <TPBAR>
  local tag=$1 tpbar=$2
  echo "=== $tag (MHC=1, CHECK=1, TPB=1, TPBAR=$tpbar) -> $OUT/$tag.log"
  timeout "${TIMEOUT:-5400}" env \
      PATH="$VENV:$PATH" PYTHONPATH="$PWD/boot" \
      SGL_TP=8 DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_POOL=1 \
      DSV41_STATIC_CHECK=1 DSV41_STATIC_MHC=1 DSV41_STATIC_AR=0 \
      DSV41_STATIC_TPB=1 DSV41_STATIC_TPBAR=$tpbar \
      DSV41_STATIC_CHECK_STEPS=$STEPS DSV41_STATIC_MAX_POS=33000 DSV41_STATIC_ROWS=1 \
      SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
      ${DETAIL:+DSV41_STATIC_DETAIL=$DETAIL} ${DUMPDIR:+DSV41_STATIC_DUMPDIR=$DUMPDIR} \
      python l2_check.py --model "$CKPT" --row "$ROW" --tokens "$TOKENS" --out "$OUT/$tag" \
      ${MEMFRAC:+--mem-fraction "$MEMFRAC"} \
      > "$OUT/$tag.log" 2>&1
  local rc=$?
  grep -a "static layers serving\|patched SGLang" "$OUT/$tag.log" | head -2 | sed 's/^/    /'
  [ $rc -ne 0 ] && echo "    rc=$rc (a timeout or a crash; see $OUT/$tag.log)"
  # A run that never loaded the hook is stock SGLang with a static-looking log. It happens (sitecustomize's imports
  # fail softly) and it produces a plausible answer to a question that was never asked, so refuse it here.
  if ! grep -qa "patched SGLang" "$OUT/$tag.log"; then
    echo "    INVALID: the hook never loaded, this was stock SGLang"
    grep -am2 "Error in sitecustomize\|ImportError" "$OUT/$tag.log" | sed 's/^/      /'
    return 9
  fi
  if ! grep -qa "check " "$OUT/$tag.log"; then
    echo "    INVALID: no comparison line; see the NOT checked reasons below"
    grep -am4 "NOT checked\|NOT FINITE" "$OUT/$tag.log" | cut -c1-170 | sed 's/^/      /'
    return 9
  fi
  return $rc
}

if [ "$ONLY" != "kern" ]; then run sync 1 || true; fi
if [ "$ONLY" != "sync" ]; then run kern 0 || true; fi

echo
for tag in sync kern; do
  [ -f "$OUT/$tag.log" ] || continue
  echo "=== $tag: which layer, if any"
  "$VENV/python" layerwise_report.py --log "$OUT/$tag.log" --cap "${CAP:-1e-1}" | sed 's/^/  /'
  echo
done
echo "expected: sync clean (the comparison works); kern the first divergence, which is the layer to fix."
echo "a 'check attn ... y relRMS' log instead of 'check mHC' means MHC=0 reached the build -- see the report's hint."
