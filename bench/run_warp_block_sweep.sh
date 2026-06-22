#!/usr/bin/env bash
# Full warp-vs-block sweep for a quiet GPU. Compiles bench_warp_vs_block.cu and
# runs it across the NPROB=1 (single-problem latency) → large (throughput) regimes
# for all ops (dot/gemv/gemm/chol/trsv/posv) × sizes. Captures to a timestamped
# file. Run on an idle GPU; do NOT run other GPU/heavy-CPU work concurrently.
#
#   ./run_warp_block_sweep.sh [arch=sm_120]
set -e
cd "$(dirname "$0")"
ARCH="${1:-sm_120}"
BIN=/tmp/bwvb_sweep
OUT="warp_block_sweep_$(date +%Y%m%d_%H%M).txt"

nvcc -std=c++17 -arch="$ARCH" -O3 -Xptxas -O1 -I.. -I../src \
     bench_warp_vs_block.cu -o "$BIN"

{
  echo "# warp-vs-block sweep  $(date)"
  nvidia-smi --query-gpu=name,clocks.max.sm,clocks.sm,temperature.gpu --format=csv,noheader 2>/dev/null || true
  echo
  # (NPROB reps): more reps for small NPROB to amortize launch; fewer for large.
  for cfg in "1 3000" "64 1500" "1024 700" "8192 500" "32768 250"; do
    set -- $cfg
    echo "################ NPROB=$1  reps=$2 ################"
    "$BIN" "$1" "$2"
    echo
  done
} | tee "$OUT"

echo "==> wrote bench/$OUT"
