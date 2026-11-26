#!/bin/bash
# The acceptance run the goal names: does the full 40-layer static chain in SGLang produce the same next-token
# distributions as stock SGLang? Golden = golden_bench32k.jsonl (stock SGLang, greedy, 32 steps per row).
#
# Four runs, one command:
#   1  static    all 40 layers static (DSV41_STATIC_ATTN=all MOE=all POOL=1)   -> the thing under test
#   2  baseline  the same script with the hook off                             -> the engine's own drift
#   3  baseline vs golden                                                      -> that drift, as a number:
#                                                                                 it is what --tol must be
#   4  static vs golden                                                        -> the acceptance
#   +  static vs baseline: engine noise cancels, only the kernels differ
#
# The first static run builds 80 layer kernels unless the cache holds them (FINDINGS_correctness.md §4.5: any
# change to core.cuh invalidates every cached .so, via code_tag -- so a cold cache costs a full rebuild, measured
# at ~1.8 min for 96 nvcc invocations (6 layers x 8 ranks), so ~12-15 min for all 80).
#
#   ./accept.sh                     # row 0, 32 steps (the golden set's own shape)
#   TOKENS=8 ./accept.sh            # a quick first look
#   ROWS="0 1" ./accept.sh          # one file per row
set -u
cd "$(dirname "$0")"

MODEL=${MODEL:-/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017}
ROWS=${ROWS:-0}
TOKENS=${TOKENS:-32}
OUT=${OUT:-/tmp/accept}
VENV=${VENV:-/opt/dlami/nvme/saboxu/venvs/sglang/bin}
export PATH=$VENV:$PATH
export PYTHONPATH=$PWD/boot
export SGL_TP=${SGL_TP:-8}
mkdir -p "$OUT"

run_static() {   # run_static <row> <out-file> <static:1|0>
  local row=$1 out=$2 stat=$3
  local envs=(DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_POOL=1)
  # the baseline must be provably stock: boot/sitecustomize.py loads the hook only when DSV41_STATIC_ATTN or
  # DSV41_STATIC_MOE is non-empty, and `env` without -u would inherit an export from this shell and quietly make
  # the control a second static run. Unset them explicitly, then check the log for the hook's banner below.
  [ "$stat" = "0" ] && envs=(-u DSV41_STATIC_ATTN -u DSV41_STATIC_MOE -u DSV41_STATIC_POOL -u DSV41_STATIC_CHECK)
  echo "=== row $row, static=$stat -> $out"
  timeout "${TIMEOUT:-5400}" env "${envs[@]}" python l2_golden.py \
      --model "$MODEL" --row "$row" --tokens "$TOKENS" --topk 32 --batch 1 \
      --tag $([ "$stat" = "1" ] && echo static || echo baseline) \
      --out "$out" > "$out.log" 2>&1
  local rc=$?
  grep -aE "^\[|engine up|tokens in|FAILED|Error|error:" "$out.log" | tail -4
  [ $rc -ne 0 ] && echo "    rc=$rc (see $out.log)"
  # did the run do what its label says? A baseline that actually ran the hook, or a static run that did not,
  # would make every number below meaningless -- and both would still "pass". The hook prints once per process:
  # the TP workers plus (with boot/sitecustomize.py whitelisting l2_golden) the parent, as l2_check's runs show.
  local np
  np=$(grep -ac "patched SGLang" "$out.log" || true)
  if [ "$stat" = "1" ]; then
    if [ "${np:-0}" -lt "${SGL_TP:-8}" ]; then
      echo "    INVALID: the static run patched only ${np:-0} processes (< $SGL_TP); the hook did not reach every rank"
      return 9
    fi
    echo "    hook loaded in ${np} processes"
  elif [ "${np:-0}" -ne 0 ]; then
    echo "    INVALID: the baseline loaded the hook (${np} processes); a DSV41_STATIC_* leaked into it"
    return 9
  else
    echo "    confirmed stock SGLang (the hook never loaded)"
  fi
  return $rc
}

for row in $ROWS; do
  S="$OUT/static_row$row.jsonl"
  B="$OUT/baseline_row$row.jsonl"

  run_static "$row" "$S" 1
  run_static "$row" "$B" 0

  echo
  echo "=== row $row: baseline vs golden (the noise floor -- this is the number --tol should be)"
  python compare_golden.py --golden golden_bench32k.jsonl --other "$B" | tee "$OUT/row${row}_baseline_vs_golden.txt" | tail -6
  echo
  echo "=== row $row: static vs golden (the acceptance)"
  python compare_golden.py --golden golden_bench32k.jsonl --other "$S" | tee "$OUT/row${row}_static_vs_golden.txt" | tail -8
  echo
  echo "=== row $row: static vs baseline (engine noise cancels; only the kernels differ)"
  python compare_golden.py --golden "$B" --other "$S" | tee "$OUT/row${row}_static_vs_baseline.txt" | tail -6
done

echo
echo "verdict files in $OUT:"
ls -l "$OUT"/*_vs_*.txt 2>/dev/null | awk '{print "  "$NF}'
echo "A pass is 'all rows within tolerance' in *_static_vs_golden.txt with the same maxval as the baseline's."
echo "A non-finite step or a collapsed argmax is a hard failure (compare_golden.py: isfinite + argument-collapse)."
