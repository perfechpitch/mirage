#!/bin/bash
# Repeatability bisect for the golden set: same row, same everything, run twice
# per configuration.  The static chain's reference is only worth as much as the
# engine's noise floor, and the plain single-row run was NOT repeatable (equal
# first five ids, then a fork at the sixth, logprobs off by ~0.1).
#
# Each configuration is two fresh engines (a fresh engine is what a rerun costs,
# and warm state -- an engram table still in page cache, a different autotune
# pick -- is exactly what we are trying to catch).
#
#   ./bisect_repeat.sh              # the default matrix, logs to /tmp/bisect_*
set -u
cd "$(dirname "$0")"
export PATH=/opt/dlami/nvme/saboxu/venvs/sglang/bin:$PATH
export LD_LIBRARY_PATH=/opt/dlami/nvme/saboxu/venvs/mirage/lib/python3.12/site-packages/z3/lib:${LD_LIBRARY_PATH:-}
# Another user's training job now shares the box: its 33-60 GB per GPU makes the
# capacity uneven, which startup treats as fatal unless we say otherwise.
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

MODEL=/opt/dlami/nvme/saboxu/models/DeepSeek-V4.1-Flash-1017
TOKENS=32
ROWS=0-0

run() {  # tag, engram-config, mem-fraction
  local tag=$1 cfg=$2 mem=$3
  echo "===== $tag  engram=${cfg:-<default>}  mem=${mem}  $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SGL_TP=8 timeout 2400 \
    python -u golden_all.py --model "$MODEL" --tokens "$TOKENS" --concurrent 1 \
      --rows "$ROWS" --mem-fraction "$mem" \
      --engram-config "$cfg" \
      --out "/tmp/bisect_${tag}.jsonl" 2>&1 | grep -E "^\[|engine up|Error|error|RuntimeError" | tail -5
}

# two runs per placement: agreement inside a configuration is the only thing that
# makes the golden set usable.
# The cpu arm comes first: every run so far used it, so it is the one whose noise
# the existing measurements are drowning in.  Its budget floor is ~0.29 (weights
# are 59.5 GB/card of the ~206 GB free here); 0.5 is the value the earlier runs
# used on this box and leaves ~40 GB of KV pool.
for i in a b; do run "cpu_$i"  '{"placement": "cpu"}' 0.50; done
# Engram tables on the GPU instead of host memory cost ~20 GB/card, which pushes
# the floor to ~0.39, so this arm gets a larger static budget.
for i in a b; do run "gpu_$i"  '{"placement": "gpu"}' 0.55; done
echo "===== done $(date +%H:%M:%S)"
