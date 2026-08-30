#!/usr/bin/env bash
# Full native/NVIDIA execution-scope sweep for a quiet GPU: thread, warp,
# block(SIMT), NVIDIA block, and NVIDIA thread where supported, across batch
# regimes, both precisions, all ladder ops, N up to 128.
# Compiles bench_mega_sweep.cu with cuBLASDx + cuSOLVERDx (needs MATHDX_ROOT). Captures to a
# timestamped file. Run on an idle GPU; do NOT run other GPU/heavy-CPU work concurrently.
#
#   ./run_mega_sweep.sh [arch=sm_120]
set -e
cd "$(dirname "$0")"
ARCH="${1:-sm_120}"
SMS="${ARCH#sm_}0"                       # sm_120 -> 1200
BIN=/tmp/bms_sweep
OUT="mega_sweep_$(date +%Y%m%d_%H%M).txt"

if [ -z "$MATHDX_ROOT" ] || [ ! -f "$MATHDX_ROOT/include/cublasdx.hpp" ]; then
  echo "ERROR: MATHDX_ROOT not set or cublasdx.hpp missing — the nvidia leg needs MathDx." >&2
  echo "  Set MATHDX_ROOT, or compile without the -D flags for the native-only ladder." >&2
  exit 1
fi

echo "==> compiling native + NVIDIA block/thread ladder (sm=$SMS) ..."
nvcc -std=c++17 -arch="$ARCH" -O3 --expt-relaxed-constexpr -Xptxas -O1 -I.. -I../src \
     -I"$MATHDX_ROOT/include" -I"$MATHDX_ROOT/external/cutlass/include" \
     -DGLASS_BENCH_CUBLASDX -DGLASS_BENCH_CUSOLVERDX -DSMS="$SMS" \
     -DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT -rdc=true -dlto \
     -L"$MATHDX_ROOT/lib" -lcusolverdx -lcublas -lcusolver -lcudart \
     bench_mega_sweep.cu -o "$BIN"

{
  echo "# mega sweep  $(date)"
  nvidia-smi --query-gpu=name,clocks.max.sm,clocks.sm,temperature.gpu --format=csv,noheader 2>/dev/null || true
  echo
  # (NPROB reps): more reps for small NPROB to amortize launch; fewer for large.
  for cfg in "1 3000" "64 1500" "1024 700" "8192 500" "32768 200"; do
    set -- $cfg
    echo "################ NPROB=$1  reps=$2  dtype=f32 ################"
    "$BIN" "$1" "$2" f32
    echo
    echo "################ NPROB=$1  reps=$2  dtype=f64 ################"
    "$BIN" "$1" "$2" f64
    echo
  done
} | tee "$OUT"

echo "==> wrote bench/$OUT"
