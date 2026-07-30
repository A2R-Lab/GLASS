#!/usr/bin/env bash
# run_jetson.sh — one-command Jetson benchmark capture for the GLASS paper.
#
# Runs, in order: device/software provenance capture -> builds -> the timed
# legs (3-tier ladder, hostblas+latency, fusion, robotics micro-ops) -> tars
# everything into jetson_<host>_<ts>.tar.gz for hand-off.
#
# Usage (on the Jetson, from the repo root or bench/):
#     ./bench/run_jetson.sh                 # everything (1-3 h; Nano is slowest)
#     ./bench/run_jetson.sh --build-only    # compile only (safe anytime)
#
# Pre-flight (see bench/JETSON.md): be on the latest JetPack you can, then
#     sudo nvpmodel -m 0 && sudo jetson_clocks
# The script warns if power mode / clocks are not pinned but keeps going —
# whatever ran is recorded in the provenance bundle.
set -uo pipefail
cd "$(dirname "$0")"

BUILD_ONLY=0
[[ "${1:-}" == "--build-only" ]] && BUILD_ONLY=1

ARCH="${GLASS_JETSON_ARCH:-sm_87}"     # all Orin-class boards
SMS="${ARCH#sm_}0"                     # sm_87 -> 870
TS=$(date +%Y%m%d_%H%M)
OUT="jetson_$(hostname)_${TS}"
PROV="$OUT/provenance"
mkdir -p "$PROV" build

log() { printf '\n=== %s ===\n' "$*"; }

# ── 1. provenance ────────────────────────────────────────────────────────────
log "provenance -> $PROV"
{
  echo "capture=$OUT arch=$ARCH date=$(date -Is)"
  echo "--- board ---";        tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "n/a"; echo
  echo "--- L4T ---";          cat /etc/nv_tegra_release 2>/dev/null || echo "n/a"
  echo "--- jetpack pkgs ---"; dpkg -l 2>/dev/null | grep -Ei 'nvidia-jetpack|nvidia-l4t-core' || echo "n/a"
  echo "--- nvcc ---";         nvcc --version 2>/dev/null || echo "n/a"
  echo "--- gcc ---";          gcc --version 2>/dev/null | head -1 || echo "n/a"
  echo "--- kernel ---";       uname -a
  echo "--- mem ---";          free -h
  echo "--- cpus ---";         nproc; lscpu 2>/dev/null | grep -E 'Model name|CPU\(s\)' || true
} > "$PROV/system.txt"

# power mode + clocks: record, and warn loudly if not pinned for timing.
nvpmodel -q > "$PROV/nvpmodel.txt" 2>&1 || echo "nvpmodel unavailable" > "$PROV/nvpmodel.txt"
sudo -n jetson_clocks --show > "$PROV/jetson_clocks.txt" 2>&1 \
  || jetson_clocks --show > "$PROV/jetson_clocks.txt" 2>&1 \
  || echo "jetson_clocks unavailable (run 'sudo jetson_clocks' before timing)" > "$PROV/jetson_clocks.txt"
if ! grep -qi 'maxn' "$PROV/nvpmodel.txt"; then
  echo "!!! WARNING: power mode is not MAXN — timing will understate the board."
  echo "!!!          Fix with: sudo nvpmodel -m 0 && sudo jetson_clocks ; then rerun."
fi

log "device_info probe"
nvcc -std=c++17 -O2 device_info.cu -o build/device_info \
  && ./build/device_info | tee "$PROV/device_info.txt"

# 10 s tegrastats sample (baseline idle power/clocks).
if command -v tegrastats >/dev/null 2>&1; then
  timeout 11 tegrastats --interval 1000 > "$PROV/tegrastats_idle.txt" 2>&1 || true
fi

# ── 2. builds (safe on a busy GPU) ───────────────────────────────────────────
log "builds ($ARCH)"
python3 tune.py --sm "$SMS" --allow-no-mathdx --legs ladder --prebuild || exit 1
python3 paper_sweeps.py --arch "$ARCH" --build-only || exit 1
ROBOT_BIN="build/bench_robotics_${ARCH}"
if [[ ! -x "$ROBOT_BIN" || bench_robotics.cu -nt "$ROBOT_BIN" ]]; then
  nvcc -std=c++17 -arch="$ARCH" -O3 -I.. -I../src -DSMS="$SMS" \
       bench_robotics.cu -o "$ROBOT_BIN" || exit 1
fi
BODY_BIN="build/bench_body_dispatch_${ARCH}"
if [[ ! -x "$BODY_BIN" || bench_body_dispatch.cu -nt "$BODY_BIN" ]]; then
  nvcc -std=c++17 -arch="$ARCH" -O3 -I.. -I../src -DSMS="$SMS" \
       bench_body_dispatch.cu -o "$BODY_BIN" || exit 1
fi
if [[ "$BUILD_ONLY" == 1 ]]; then
  log "build-only done — rerun without --build-only for the timed capture"
  exit 0
fi

# ── 3. timed legs (quiet GPU; tegrastats logs alongside for energy/solve) ───
TEGRA_PID=""
if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 1000 > "$PROV/tegrastats_run.txt" 2>&1 &
  TEGRA_PID=$!
fi
finish() { [[ -n "$TEGRA_PID" ]] && kill "$TEGRA_PID" 2>/dev/null; }
trap finish EXIT

log "leg 1/3: 3-tier ladder (tune.py, dry-run regen; table splice done off-box)"
python3 tune.py --sm "$SMS" --allow-no-mathdx --legs ladder --dry-run || exit 1

log "leg 2/3: hostblas + latency + fusion (paper_sweeps.py)"
python3 paper_sweeps.py --arch "$ARCH" || exit 1

log "leg 3/4: robotics micro-ops"
ROBOTXT="robotics_sweep_${TS}.txt"
{
  echo "# capture $ROBOTXT | $(hostname) | $ARCH jetson"
  for NPROB in 256 1024 4096 16384 32768; do
    for DT in f32 f64; do
      echo; echo "== NPROB=$NPROB reps=500 dtype=$DT =="
      "$ROBOT_BIN" "$NPROB" 500 "$DT" || exit 1
    done
  done
} > "$ROBOTXT" || exit 1
grep -q '||' "$ROBOTXT" || { echo "FATAL: robotics capture parsed empty"; exit 1; }

log "leg 4/4: in-block body dispatch (bare-face Phase-2 table data)"
BODYTXT="body_dispatch_sweep_${TS}.txt"
{
  echo "# capture $BODYTXT | $(hostname) | $ARCH jetson"
  for NPROB in 1024 4096 16384; do
    for DT in f32 f64; do
      echo; echo "== NPROB=$NPROB reps=250 dtype=$DT =="
      "$BODY_BIN" "$NPROB" 250 "$DT" || exit 1
    done
  done
} > "$BODYTXT" || exit 1
grep -q '||' "$BODYTXT" || { echo "FATAL: body-dispatch capture parsed empty"; exit 1; }

# ── 4. package ───────────────────────────────────────────────────────────────
log "packaging"
finish; trap - EXIT
for pat in mega_sweep_ paper_hostblas_ paper_fusion_; do
  latest=$(ls -t ${pat}*.txt 2>/dev/null | head -1)
  [[ -n "$latest" ]] && cp "$latest" "$OUT/" || echo "WARNING: no ${pat}*.txt capture found"
done
cp "$ROBOTXT" "$BODYTXT" "$OUT/"
tar czf "${OUT}.tar.gz" "$OUT"
log "DONE -> bench/${OUT}.tar.gz  (send this file back)"
