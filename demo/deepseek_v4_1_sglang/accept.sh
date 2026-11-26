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
  # The configuration is the model's, not a subset of it. mHC=1 is part of DeepSeek-V4.1's Block (hc_mult=4) and it
  # is what puts the sublayer's end -- the all-reduce over the exchange region -- inside the kernel; with mHC=0 that
  # work stays in SGLang's own code and the chain under test is a different one. (dsv41_static_hook.py: MHC defaults
  # to 0, so the goal's three variables alone do NOT select the static chain the NaN was measured in.)
  #
  # TPBAR=0 is deliberate: DSV41_STATIC_TPBAR (default 1 when MHC=1) is a host barrier after every sublayer, added
  # 11-26, and it masks exactly the failure this run is supposed to prove gone -- with it on, a broken kernel barrier
  # would still come out finite because the ranks are lined up on the host. TPB=1 is the kernel's own start barrier.
  # So this run asks the strong question: does the chain reproduce the engine with NO host round trip at all?
  local envs=(DSV41_STATIC_ATTN=all DSV41_STATIC_MOE=all DSV41_STATIC_POOL=1 DSV41_STATIC_MHC=1
              DSV41_STATIC_TPB=1 DSV41_STATIC_TPBAR=0)
  # the baseline must be provably stock: boot/sitecustomize.py loads the hook only when DSV41_STATIC_ATTN or
  # DSV41_STATIC_MOE is non-empty, and `env` without -u would inherit an export from this shell and quietly make
  # the control a second static run. Unset them explicitly, then check the log for the hook's banner below.
  [ "$stat" = "0" ] && envs=(-u DSV41_STATIC_ATTN -u DSV41_STATIC_MOE -u DSV41_STATIC_POOL -u DSV41_STATIC_CHECK
                             -u DSV41_STATIC_MHC -u DSV41_STATIC_TPB -u DSV41_STATIC_TPBAR)
  echo "=== row $row, static=$stat -> $out"
  echo "    ${envs[*]:-stock SGLang, no DSV41_STATIC_*}"
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
    # and it ran the chain this script intends: mHC in the kernels, the kernel's barrier on, and NO host barrier
    # (TPBAR would line the ranks up and hide a broken kernel barrier -- the whole point of this run).
    local want="MHC=1 TPBAR=0 TPB=1"
    if ! grep -qa "MHC=1 TPBAR=0 TPB=1" "$out.log"; then
      echo "    INVALID: not the chain this script means to test (want '$want'); the banner says:"
      grep -am1 -o "MHC=.*" "$out.log" | sed 's/^/      /'
      return 9
    fi
    echo "    hook loaded in ${np} processes, chain: $want"
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
  echo "=== row $row: static vs golden"
  python compare_golden.py --golden golden_bench32k.jsonl --other "$S" | tee "$OUT/row${row}_static_vs_golden.txt" | tail -8
  echo
  echo "=== row $row: static vs baseline (engine noise cancels; only the kernels differ)"
  python compare_golden.py --golden "$B" --other "$S" | tee "$OUT/row${row}_static_vs_baseline.txt" | tail -6

  # The verdict, with the rule stated instead of eyeballed. Golden was written with 8 rows in flight and
  # deterministic off; this run uses one row, so static-vs-golden carries that configuration gap too -- which is
  # why the baseline (the same script, hook off) is measured alongside it. The static chain is right when it is
  # no worse than the engine's own rerun, and the goldens' tuned 0.15 stays as the absolute cap.
  python - "$OUT" "$row" <<'PY'
import json, subprocess, sys, os
out, row = sys.argv[1], sys.argv[2]
def j(a, b):
    r = subprocess.run([sys.executable, "compare_golden.py", "--golden", a, "--other", b, "--json"],
                       capture_output=True, text=True)
    return json.loads(r.stdout) if r.stdout.strip().startswith("{") else dict(ok=False, why={"": r.stdout[-200:]})
base = j("golden_bench32k.jsonl", f"{out}/baseline_row{row}.jsonl")
stat = j("golden_bench32k.jsonl", f"{out}/static_row{row}.jsonl")
cross = j(f"{out}/baseline_row{row}.jsonl", f"{out}/static_row{row}.jsonl")
def mx(d):
    v = [r["maxval"] for r in d.get("rows", {}).values() if r.get("maxval") is not None]
    return max(v) if v else None
bm, sm, cm = mx(base), mx(stat), mx(cross)
cap = max(base["tol"], bm or 0.0)
print()
print(f"=== VERDICT row {row}")
print(f"  baseline vs golden   maxval {bm}   ok {base['ok']}")
print(f"  static   vs golden   maxval {sm}   ok {stat['ok']}")
print(f"  static   vs baseline maxval {cm}   ok {cross['ok']}   <- the sharp one")
print(f"  rule: static must be ok in all three; its maxval vs golden <= max(tol {base['tol']}, baseline {bm})")
bad = [n for n, d in (("static vs golden", stat), ("static vs baseline", cross)) if not d["ok"]]
if not stat["ok"] or not cross["ok"]:
    print(f"  RESULT: FAIL -- {', '.join(bad)}"
          + "".join(f"\n    {n}: {d['why']}" for n, d in
                    (("static vs golden", stat), ("static vs baseline", cross)) if d["why"]))
    open(f"{out}/row{row}_verdict.txt", "w").write("FAIL\n" + json.dumps(dict(static=stat, baseline=base, cross=cross), indent=1))
    sys.exit(1)
if sm is not None and bm is not None and sm > cap:
    print(f"  RESULT: FAIL -- static is noisier than the engine itself ({sm:.3f} > {cap:.3f})")
    sys.exit(1)
print(f"  RESULT: PASS -- the 40 layers reproduce the engine, within the engine's own noise"
      f"{'' if cm is None else f' (static vs baseline maxval {cm:.3f})'}")
open(f"{out}/row{row}_verdict.txt", "w").write("PASS\n" + json.dumps(dict(static=stat, baseline=base, cross=cross), indent=1))
PY
done

echo
echo "verdict files in $OUT:"
ls -l "$OUT"/*_vs_*.txt 2>/dev/null | awk '{print "  "$NF}'
echo "A pass is 'all rows within tolerance' in *_static_vs_golden.txt with the same maxval as the baseline's."
echo "A non-finite step or a collapsed argmax is a hard failure (compare_golden.py: isfinite + argument-collapse)."
