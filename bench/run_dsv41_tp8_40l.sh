#!/usr/bin/env bash
# deepseek-v4.1-flash-1017, 40 layers, TP = 8, through SGLang and the static megakernel.
#
#   bench/run_dsv41_tp8_40l.sh                 # 8 tokens, 8 prompts
#   TOKENS=16 GPUS=0,1,2,3,4,5,6,7 bench/run_dsv41_tp8_40l.sh
#   BENCH=0 bench/run_dsv41_tp8_40l.sh         # skip the baseline, the static path only
#
# What the run needs: 8 GPUs with about 60 GiB free each (TP8 = 59.4 GiB of weights) and /dev/shm.
# The first launch also compiles the FlashInfer JIT kernels (sm103a, about 10 min): --watchdog keeps
# SGLang's 300 s from killing it. Details and the measured numbers: demo/deepseek_v4_1_sglang/FINDINGS_bs1.md
set -euo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/saboxu/mirage}
export ROOT
D=$ROOT/demo/deepseek_v4_1_sglang
MODEL=${MODEL:-/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017}
SGLVENV=${SGLVENV:-/opt/dlami/nvme/saboxu/venvs/sglang}
Z3LIB=/opt/dlami/nvme/saboxu/venvs/mirage/lib/python3.12/site-packages/z3/lib

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TOKENS=${TOKENS:-8}
PROMPTS=${PROMPTS:-8}
OUT=${OUT:-/tmp/static_tp8_t${TOKENS}.json}
LOG=${LOG:-/tmp/static_tp8_t${TOKENS}.log}
BENCH=${BENCH:-1}            # 1: also run the official SGLang baseline and diff the tokens
WATCHDOG=${WATCHDOG:-7200}   # s; the first run spends it compiling the FlashInfer JIT kernels
MEMFRAC=${MEMFRAC:-0.90}

# ---- the static megakernel: every layer, mHC, Engram mode 2, the head -------------------------
# ATTN/MOE=all: all 40 layers. MHC=1: the residual mixes in the kernel. ENGRAM=2: the next layer's
# kv in the attention kernel's idle SMs (the more accurate of the two modes). HEAD=2: the LM head.
# EG_KV14=1: Engram's kv14 path. Nothing else on purpose: the README_bs8 flags (STATIC_NO_START_BARRIER,
# RESET_PARITY, HCP_FAST_END, PDL, the poll ns of 0) are for a bs = 8 configuration measured with
# DSV41_STATIC_ROWS=1,2,4 DSV41_STATIC_T=16,32,64 -- see FINDINGS_bs1.md 8.6, where mixing them in
# without those two hung the second decode step in a counter poll.
export DSV41_STATIC_ROOT=$ROOT
export DSV41_STATIC_ATTN=${DSV41_STATIC_ATTN:-all}
export DSV41_STATIC_MOE=${DSV41_STATIC_MOE:-all}
export DSV41_STATIC_MHC=${DSV41_STATIC_MHC:-1}
export DSV41_ENGRAM=${DSV41_ENGRAM:-2}
export DSV41_EG_KV14=${DSV41_EG_KV14:-1}
export DSV41_HEAD=${DSV41_HEAD:-2}
export DSV41_STATIC_CKPT=${DSV41_STATIC_CKPT:-$MODEL}
# the decode step's rows: 1 keeps a step to one row, which is what bs = 1 is about
export DSV41_STATIC_ROWS=${DSV41_STATIC_ROWS:-1}

export CUDA_VISIBLE_DEVICES=$GPUS
export SGL_TP=8
export SGLANG_DSV4_KV_LAYOUT=v41_new   # without it SGLang asserts on KVLayout.V4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$SGLVENV/bin:$PATH
export PYTHONPATH=$D/boot:$ROOT/python:$ROOT
export LD_LIBRARY_PATH=$Z3LIB:${LD_LIBRARY_PATH:-}

die() { echo "run_dsv41_tp8_40l: $*" >&2; exit 1; }
[ -d "$MODEL" ] || die "no checkpoint at $MODEL"
[ -x "$SGLVENV/bin/python" ] || die "no python in $SGLVENV"

n=$(python3 -c "print(len('$GPUS'.split(',')))")
[ "$n" = 8 ] || die "TP8 needs 8 GPUs, GPUS=$GPUS has $n"

# /dev/shm: the exchange pool and SGLang's IPC live there; a full one fails every rank at once
free_shm=$(df -Pm /dev/shm | awk 'NR==2 {print $4}')
[ "$free_shm" -gt 65536 ] || die "/dev/shm has $((free_shm / 1024)) GiB free, want > 64"

# Every rank holds the whole model's 1/8: 59.4 GiB of weights (475.2 GiB / 8) plus a KV cache. The
# weights are the hard floor -- no mem_fraction gets under them -- and this launch's KV and scratch
# on top. A GPU someone else is using is the usual reason a TP8 launch dies ten minutes in, so the
# check is up front rather than after the build.
NEED_MIB=${NEED_MIB:-73728}
IFS=, read -ra gi <<< "$GPUS"
short=()
for i in "${gi[@]}"; do
  m=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$i")
  [ "$m" -ge "$NEED_MIB" ] || short+=("$i has $((m / 1024)) GiB")
done
if [ ${#short[@]} -ne 0 ]; then
  echo "run_dsv41_tp8_40l: TP8 wants $((NEED_MIB / 1024)) GiB free per GPU (59.4 of weights, the rest KV):" >&2
  echo "  ${short[*]}" >&2
  echo "  weights alone are 59.4 GiB a rank, so a GPU below that cannot run TP8 at any MEMFRAC." >&2
  echo "  Free a GPU (GPUS=...) or use TP4, which needs 118.8 GiB a rank on 4 GPUs." >&2
  exit 1
fi

echo "TP8 40 layers: GPUs $GPUS, $PROMPTS prompts x $TOKENS tokens, mem-fraction $MEMFRAC"
echo "  log $LOG"
echo "  out $OUT"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed -n '1,8p' | sed 's/^/  /'
echo "  /dev/shm: $((free_shm / 1024)) GiB free"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN=1: the environment checks passed, nothing launched"
  echo "  static: ATTN=$DSV41_STATIC_ATTN MOE=$DSV41_STATIC_MOE MHC=$DSV41_STATIC_MHC ENGRAM=$DSV41_ENGRAM HEAD=$DSV41_HEAD ROWS=$DSV41_STATIC_ROWS"
  exit 0
fi

run() {  # run <out> <log> [extra flags]
  "$SGLVENV/bin/python" "$D/sgl_generate.py" --model "$MODEL" \
    --same-length-copies --ignore-eos --tokens "$TOKENS" --prompts "$PROMPTS" \
    --no-autotune --mem-fraction "$MEMFRAC" --watchdog "$WATCHDOG" \
    --out "$1" >"$2" 2>&1
}

if [ "$BENCH" = 1 ]; then
  # the official SGLang path, no static kernels: the reference tokens
  echo "== baseline (official SGLang) =="
  DSV41_STATIC_ATTN= DSV41_STATIC_MOE= DSV41_STATIC_MHC= DSV41_HEAD= DSV41_ENGRAM= DSV41_EG_KV14= \
    run "${OUT%.json}_ref.json" "${LOG%.log}_ref.log"
  grep -qiE "^(Error|Traceback)" "${LOG%.log}_ref.log" && die "the baseline failed: ${LOG%.log}_ref.log"
fi

echo "== static megakernel, TP8 =="
run "$OUT" "$LOG" || die "the static run failed: $LOG"

# ---- the tokens against the baseline ----------------------------------------------------------
if [ "$BENCH" = 1 ] && [ -f "${OUT%.json}_ref.json" ]; then
  "$SGLVENV/bin/python" - "$OUT" "${OUT%.json}_ref.json" <<'PY'
import json, sys
got = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))
def ids(o):
    if isinstance(o, dict):
        for k in ("output_ids", "ids", "tokens", "outputs"):
            if k in o:
                return ids(o[k])
        return list(o.values())
    if isinstance(o, list) and o and isinstance(o[0], dict):
        return [x.get("output_ids", x.get("ids", x.get("tokens"))) for x in o]
    return o
g, r = ids(got), ids(ref)
g = g if isinstance(g, list) else [g]
r = r if isinstance(r, list) else [r]
n = min(len(g), len(r))
same = sum(1 for i in range(n) if list(g[i]) == list(r[i]))
first = next((i for i in range(n) if list(g[i]) != list(r[i])), None)
print(f"tokens: {same}/{n} sequences identical to the baseline")
for i in range(n):
    m = "  " if list(g[i]) == list(r[i]) else "!!"
    print(f" {m} [{i}] static {list(g[i])}")
    if list(g[i]) != list(r[i]):
        print(f"      ref    {list(r[i])}")
if first is None and n:
    print("PASS: the 40 layers at TP8 give the official's tokens")
else:
    print(f"FAIL: sequence {first} differs")
    sys.exit(1)
PY
fi

echo "done. log $LOG"
