#!/usr/bin/env bash
# DeepSeek-V4.1-Flash, 40 layers, TP = 8, a *serving* SGLang whose decode steps run on the static megakernel,
# measured by the LCA bench32k client (32K-context prompts, one stream) -> TPOT.
#
#   bench/run_dsv41_tp8_bench32k.sh                 # 4 rows x 128 new tokens
#   N=8 OSL=512 bench/run_dsv41_tp8_bench32k.sh
#   KEEP=1 bench/run_dsv41_tp8_bench32k.sh          # leave the server up for a client of your own
#
# The hook: boot/sitecustomize.py on PYTHONPATH + DSV41_STATIC_*: SGLang's MQALayer/DeepseekV2MoE forwards of a
# decode step are replaced by one megakernel launch. SGLang's own attention runs the prefill, the static layers
# read/write its pools (DSV41_STATIC_POOL=1, SGLANG_DSV4_KV_LAYOUT=v41_new).
#
# Plain decode only: the static path is not speculative, so unlike the NVIDIA megakernel arm (DSPARK, ~1.9 tokens a
# step) a client step here is exactly one token. TPOT is therefore the client's step_p50_ms as it stands.
set -euo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/saboxu/mirage}
export ROOT
D=$ROOT/demo/deepseek_v4_1_sglang
MODEL=${MODEL:-/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017}
SGLVENV=${SGLVENV:-/opt/dlami/nvme/saboxu/venvs/sglang}
Z3LIB=/opt/dlami/nvme/saboxu/venvs/mirage/lib/python3.12/site-packages/z3/lib
CLIENT=${CLIENT:-/opt/dlami/nvme/saboxu/bench/bench32k_client.py}
DATA=${DATA:-/opt/dlami/nvme/saboxu/data/long-code-arena/bench32k/*.jsonl}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
N=${N:-4}                    # bench32k rows (evenly spaced: both small and medium contexts)
OSL=${OSL:-128}              # new tokens per row
PORT=${PORT:-30000}
MEMFRAC=${MEMFRAC:-0.85}
MAXPOS=${MAXPOS:-33024}      # 32K prompt + decode: the layers' index pools are sized by it
OUT=${OUT:-/opt/dlami/nvme/saboxu/results/bench32k_static_tp8_$(date +%Y%m%d_%H%M%S)}
LOG=${LOG:-/tmp/static_tp8_bench32k.log}
WATCHDOG=${WATCHDOG:-7200}   # s; the first launch spends it compiling the FlashInfer JIT kernels
KEEP=${KEEP:-0}

# ---- the static megakernel: every layer, mHC, Engram mode 2, the head -------------------------
export DSV41_STATIC_ROOT=$ROOT
export DSV41_STATIC_ATTN=${DSV41_STATIC_ATTN:-all}
export DSV41_STATIC_MOE=${DSV41_STATIC_MOE:-all}
export DSV41_STATIC_MHC=${DSV41_STATIC_MHC:-1}
export DSV41_ENGRAM=${DSV41_ENGRAM:-2}
export DSV41_EG_KV14=${DSV41_EG_KV14:-1}
export DSV41_HEAD=${DSV41_HEAD:-2}
export DSV41_STATIC_CKPT=${DSV41_STATIC_CKPT:-$MODEL}
export DSV41_STATIC_MAX_POS=$MAXPOS
export DSV41_STATIC_ROWS=${DSV41_STATIC_ROWS:-1}   # one live row a step: bs = 1
export DSV41_STATIC_POOL=${DSV41_STATIC_POOL:-1}   # read/write SGLang's pools: other row counts fall back to SGLang

export CUDA_VISIBLE_DEVICES=$GPUS
export SGLANG_DSV4_KV_LAYOUT=v41_new
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$SGLVENV/bin:$PATH
export PYTHONPATH=$D/boot:$ROOT/python:$ROOT
export LD_LIBRARY_PATH=$Z3LIB:${LD_LIBRARY_PATH:-}

die() { echo "run_dsv41_tp8_bench32k: $*" >&2; exit 1; }
[ -d "$MODEL" ] || die "no checkpoint at $MODEL"
[ -x "$SGLVENV/bin/python" ] || die "no python in $SGLVENV"
[ -f "$CLIENT" ] || die "no client at $CLIENT"
n=$(python3 -c "print(len('$GPUS'.split(',')))")
[ "$n" = 8 ] || die "TP8 needs 8 GPUs, GPUS=$GPUS has $n"
free_shm=$(df -Pm /dev/shm | awk 'NR==2 {print $4}')
[ "$free_shm" -gt 65536 ] || die "/dev/shm has $((free_shm / 1024)) GiB free, want > 64"
IFS=, read -ra gi <<< "$GPUS"
for i in "${gi[@]}"; do
  m=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$i")
  [ "$m" -ge 65536 ] || die "GPU $i has $((m / 1024)) GiB free, TP8 wants 60+ (59.4 of weights)"
done
mkdir -p "$OUT"

args=(--model-path "$MODEL" --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code
      --tp 8 --ep-size 1                    # pure TP8: the static MoE is TP8 (48 experts a rank)
      --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4
      --dsv4-attn-backend flashmla --kv-cache-dtype fp8_e4m3
      --engram-config '{"placement": "cpu"}'
      --cuda-graph-max-bs-decode 1 --max-running-requests 1
      --mem-fraction-static "$MEMFRAC" --chunked-prefill-size 8192
      --disable-radix-cache --startup-weight-load-mode serial
      --watchdog-timeout "$WATCHDOG" --host 127.0.0.1 --port "$PORT")

echo "== static TP8 server, 40 layers, max_pos $MAXPOS =="
echo "  out $OUT"
echo "  log $LOG"
echo "  gpus $GPUS, mem-fraction $MEMFRAC, plain decode (no DSPARK)"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "  DRY_RUN=1: would launch"
  printf '    %q' "$SGLVENV/bin/python" -m sglang.launch_server "${args[@]}"; echo
  exit 0
fi

setsid "$SGLVENV/bin/python" -m sglang.launch_server "${args[@]}" >"$LOG" 2>&1 &
PID=$!
trap '[ "$KEEP" = 1 ] || { kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; }' EXIT

echo "  server pid $PID, waiting for health (first launch compiles FlashInfer: ~10 min)"
for i in $(seq 1 360); do
  if ! kill -0 "$PID" 2>/dev/null; then tail -40 "$LOG" >&2; die "the server died, see $LOG"; fi
  if curl -sf -m 30 "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1; then
    echo "  ready after ${i}0 s"; break
  fi
  [ "$i" = 360 ] && { tail -40 "$LOG" >&2; die "not ready after 60 min"; }
  sleep 10
done
grep -m3 -E "static megakernel|first static decode step|static layers" "$LOG" | sed 's/^/  /' || true

echo "== bench32k client: $N rows x $OSL new tokens =="
"$SGLVENV/bin/python" "$CLIENT" --base-url "http://127.0.0.1:$PORT" --data $DATA \
  --osl "$OSL" --n "$N" --warmup 1 --label "static_tp8" --out "$OUT" 2>&1 | tee "$OUT/client.log"

echo "== TPOT =="
"$SGLVENV/bin/python" - "$OUT/client.log" <<'PY'
import json, sys
s = None
for line in open(sys.argv[1]):
    if line.startswith("SUMMARY "):
        s = json.loads(line[8:])
if not s:
    sys.exit("no SUMMARY line")
print(f"  TPOT (client step p50)  {s['step_p50_ms']:.3f} ms/token   (p10 {s['step_p10_ms']:.3f}, p90 {s['step_p90_ms']:.3f})")
print(f"  tokens/s/user p50       {s['tps_user_p50']}   isl_p50 {s['isl_p50']}   osl {s['osl_p50']}")
print(f"  TTFT p50                {s['ttft_p50_ms']} ms")
PY
[ "$KEEP" = 1 ] && echo "  server left up on port $PORT (kill: pkill -f 'sglang.launch_server.*--port $PORT')"
echo "done. $OUT"
