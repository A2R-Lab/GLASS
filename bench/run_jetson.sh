#!/usr/bin/env bash
# run_jetson.sh — one-command Jetson benchmark capture for the GLASS paper.
#
# Runs, in order: device/software provenance capture -> builds -> the timed
# legs (3-tier ladder, hostblas+latency, fusion, robotics micro-ops, in-block
# body dispatch) -> tars everything into jetson_<host>_<ts>.tar.gz.
#
# Usage (on the Jetson, from the repo root or bench/):
#     ./bench/run_jetson.sh                        # timed capture at the CURRENT power mode
#     ./bench/run_jetson.sh --build-only           # compile only (safe anytime)
#     ./bench/run_jetson.sh --power-modes all      # sweep the STANDARD wattage modes (MAXN excluded)
#     ./bench/run_jetson.sh --power-modes 3,2,1    # sweep specific mode IDs
#
# Policy: we do NOT run MAXN (unconstrained mode; owner's call to protect the
# boards) — "all" enumerates only the named-wattage modes from nvpmodel.conf.
# Pass a MAXN mode ID explicitly if you ever truly want it.
#
# Power-mode sweeps switch modes with nvpmodel + pin clocks with jetson_clocks,
# which need passwordless sudo for JUST those tools (see bench/JETSON.md):
#     echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel, /usr/bin/jetson_clocks" \
#       | sudo tee /etc/sudoers.d/glass-bench
# Single-mode runs (no --power-modes) never switch anything: they just record
# the mode that ran.
set -uo pipefail
cd "$(dirname "$0")"

# nvcc is often not on PATH on JetPack installs (lives in /usr/local/cuda/bin).
command -v nvcc >/dev/null 2>&1 || export PATH="/usr/local/cuda/bin:$PATH"
export PYTHONUNBUFFERED=1   # stream leg progress into nohup logs (Orin legs are LONG)

# MathDx: NVIDIA doesn't distribute it for Tegra, but the x86 tarball's
# headers + LTO-IR fatbins work here (tune.py stages a fatbin device-link on
# non-x86 hosts — verified on AGX Orin/CUDA 13.2). Auto-detect a copied tree
# so the ladder runs 4-tier; without one it falls back to the 3-tier SIMT sweep.
if [[ -z "${MATHDX_ROOT:-}" ]]; then
  for d in "$HOME"/mathdx/*/; do
    [[ -f "$d/include/cublasdx.hpp" ]] && export MATHDX_ROOT="${d%/}" && break
  done
fi
[[ -n "${MATHDX_ROOT:-}" ]] && echo "MathDx: $MATHDX_ROOT (4-tier ladder)" \
                            || echo "MathDx: absent (3-tier ladder)"

BUILD_ONLY=0
POWER_MODES=""            # empty = current mode only, no switching
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only)  BUILD_ONLY=1; shift ;;
    --power-modes) POWER_MODES="${2:?--power-modes needs a value}"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

ARCH="${GLASS_JETSON_ARCH:-sm_87}"     # all Orin-class boards
SMS="${ARCH#sm_}0"                     # sm_87 -> 870
TS=$(date +%Y%m%d_%H%M)
OUT="jetson_$(hostname)_${TS}"
PROV="$OUT/provenance"
mkdir -p "$PROV" build

log() { printf '\n=== %s ===\n' "$*"; }

mode_name() {  # mode id -> NAME from nvpmodel.conf
  awk -v id="$1" -F'[= ]+' '/POWER_MODEL ID=/{for(i=1;i<=NF;i++){if($i=="ID")mid=$(i+1);if($i=="NAME")nm=$(i+1)}; gsub(/>.*/,"",nm); if(mid==id){print nm; exit}}' \
    /etc/nvpmodel.conf 2>/dev/null || echo "mode$1"
}
current_mode_id() { nvpmodel -q 2>/dev/null | awk 'NR==2{print $1}'; }

# ── power-mode sweep preconditions ───────────────────────────────────────────
if [[ -n "$POWER_MODES" ]]; then
  if ! sudo -n /usr/sbin/nvpmodel -q >/dev/null 2>&1; then
    echo "FATAL: --power-modes needs passwordless sudo for nvpmodel/jetson_clocks."
    echo "Run once on this box (then rerun):"
    echo "  echo \"\$USER ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel, /usr/bin/jetson_clocks\" | sudo tee /etc/sudoers.d/glass-bench"
    exit 1
  fi
  if [[ "$POWER_MODES" == "all" ]]; then
    # standard wattage modes only — skip MAXN variants by name (policy above)
    POWER_MODES=$(grep -E 'POWER_MODEL ID=' /etc/nvpmodel.conf \
      | grep -vi 'NAME=MAXN' \
      | grep -oE 'ID=[0-9]+' | grep -oE '[0-9]+' | paste -sd, -)
    [[ -n "$POWER_MODES" ]] || { echo "FATAL: no non-MAXN modes found in /etc/nvpmodel.conf"; exit 1; }
  fi
  ORIG_MODE=$(current_mode_id)
  echo "power-mode sweep: [$POWER_MODES] (restore -> $ORIG_MODE when done)"
fi

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
  echo "--- power modes ---";  grep -E 'POWER_MODEL ID' /etc/nvpmodel.conf 2>/dev/null || echo "n/a"
} > "$PROV/system.txt"

# power mode + clocks at launch: record what this box is set to.
nvpmodel -q > "$PROV/nvpmodel.txt" 2>&1 || echo "nvpmodel unavailable" > "$PROV/nvpmodel.txt"
sudo -n jetson_clocks --show > "$PROV/jetson_clocks.txt" 2>&1 \
  || jetson_clocks --show > "$PROV/jetson_clocks.txt" 2>&1 \
  || echo "jetson_clocks unavailable (run 'sudo jetson_clocks' before timing)" > "$PROV/jetson_clocks.txt"
if [[ -z "$POWER_MODES" ]]; then
  echo "note: single-mode run — timing captured at: $(head -1 "$PROV/nvpmodel.txt")"
  echo "      (use --power-modes all for the standard-wattage sweep; MAXN is never run)"
fi

log "device_info probe"
nvcc -std=c++17 -O2 device_info.cu -o build/device_info \
  && ./build/device_info | tee "$PROV/device_info.txt"

# 10 s tegrastats sample (baseline idle power/clocks).
if command -v tegrastats >/dev/null 2>&1; then
  timeout 11 tegrastats --interval 1000 > "$PROV/tegrastats_idle.txt" 2>&1 || true
fi

# ── 2. builds (safe on a busy GPU; mode-independent) ─────────────────────────
log "builds ($ARCH)"
python3 tune.py --sm "$SMS" --allow-no-mathdx --legs ladder --prebuild || exit 1
python3 paper_sweeps.py --arch "$ARCH" --build-only || exit 1
ROBOT_BIN="build/bench_robotics_${ARCH}"
if [[ ! -x "$ROBOT_BIN" || bench_robotics.cu -nt "$ROBOT_BIN" ]]; then
  nvcc -std=c++17 -arch="$ARCH" -O3 -I.. -I../src -DGLASS_TARGET_SM="$SMS" \
       bench_robotics.cu -o "$ROBOT_BIN" || exit 1
fi
BODY_BIN="build/bench_body_dispatch_${ARCH}"
if [[ ! -x "$BODY_BIN" || bench_body_dispatch.cu -nt "$BODY_BIN" ]]; then
  nvcc -std=c++17 -arch="$ARCH" -O3 -I.. -I../src -DGLASS_TARGET_SM="$SMS" \
       bench_body_dispatch.cu -o "$BODY_BIN" || exit 1
fi
if [[ "$BUILD_ONLY" == 1 ]]; then
  log "build-only done — rerun without --build-only for the timed capture"
  exit 0
fi

# ── 3. timed legs ────────────────────────────────────────────────────────────
TEGRA_PID=""
tegra_start() {
  [[ -z "$TEGRA_PID" ]] && command -v tegrastats >/dev/null 2>&1 \
    && { tegrastats --interval 1000 > "$1" 2>&1 & TEGRA_PID=$!; }
}
tegra_stop() { [[ -n "$TEGRA_PID" ]] && kill "$TEGRA_PID" 2>/dev/null; TEGRA_PID=""; }
finish() {
  tegra_stop
  [[ -n "${ORIG_MODE:-}" ]] && sudo -n /usr/sbin/nvpmodel -m "$ORIG_MODE" >/dev/null 2>&1
}
trap finish EXIT

# One full set of timed legs; $1 = destination dir for every capture produced.
run_timed_legs() {
  local DEST="$1"; mkdir -p "$DEST"
  touch "$DEST/.legs_start"   # freshness sentinel: never package pre-existing captures

  log "leg 1/4: ladder (tune.py, dry-run regen; table splice done off-box)"
  # Tegra profile: the desktop-parity schedule measures 8-16 h/mode on a 16-SM
  # Orin (2026-07-31 calibration). Reduced reps keep the min-of ranking stable;
  # the 8192 section MUST stay (regenerated tables read it).
  python3 tune.py --sm "$SMS" --allow-no-mathdx --legs ladder --dry-run \
      --sched 64:200,1024:100,8192:50 || return 1

  log "leg 2/4: hostblas + latency + fusion (paper_sweeps.py)"
  python3 paper_sweeps.py --arch "$ARCH" --reps 15 || return 1

  log "leg 3/4: robotics micro-ops"
  local ROBOTXT="$DEST/robotics_sweep_${TS}.txt"
  {
    echo "# capture $(basename "$ROBOTXT") | $(hostname) | $ARCH jetson"
    for NR in 1024:100 4096:100 16384:50; do
      NPROB="${NR%:*}"; REPS="${NR#*:}"
      for DT in f32 f64; do
        echo; echo "== NPROB=$NPROB reps=$REPS dtype=$DT =="
        "$ROBOT_BIN" "$NPROB" "$REPS" "$DT" || return 1
      done
    done
  } > "$ROBOTXT" || return 1
  grep -q '||' "$ROBOTXT" || { echo "FATAL: robotics capture parsed empty"; return 1; }

  log "leg 4/4: in-block body dispatch (bare-face Phase-2 table data)"
  local BODYTXT="$DEST/body_dispatch_sweep_${TS}.txt"
  {
    echo "# capture $(basename "$BODYTXT") | $(hostname) | $ARCH jetson"
    # 8192 REQUIRED: tune.py --from-body reads it as the throughput regime.
    for NR in 64:200 1024:100 8192:50; do
      NPROB="${NR%:*}"; REPS="${NR#*:}"
      for DT in f32 f64; do
        echo; echo "== NPROB=$NPROB reps=$REPS dtype=$DT =="
        "$BODY_BIN" "$NPROB" "$REPS" "$DT" || return 1
      done
    done
  } > "$BODYTXT" || return 1
  grep -q '||' "$BODYTXT" || { echo "FATAL: body-dispatch capture parsed empty"; return 1; }

  # sweep outputs that tools write into bench/ under their own names: move the
  # freshest of each into DEST (mv, so the next mode can't pick up stale files;
  # the sentinel guards against packaging captures committed in the repo).
  local pat latest
  for pat in mega_sweep_ paper_hostblas_ paper_fusion_; do
    latest=$(ls -t ${pat}*.txt 2>/dev/null | head -1)
    if [[ -n "$latest" && "$latest" -nt "$DEST/.legs_start" ]]; then
      mv "$latest" "$DEST/"
    else
      echo "WARNING: no fresh ${pat}*.txt capture produced this run"
    fi
  done
  rm -f "$DEST/.legs_start"
  return 0
}

if [[ -z "$POWER_MODES" ]]; then
  tegra_start "$PROV/tegrastats_run.txt"
  run_timed_legs "$OUT" || exit 1
  tegra_stop
else
  IFS=',' read -ra MODES <<< "$POWER_MODES"
  for MID in "${MODES[@]}"; do
    MNAME=$(mode_name "$MID")
    MDIR="$OUT/mode_${MID}_${MNAME}"
    log "power mode $MID ($MNAME)"
    sudo -n /usr/sbin/nvpmodel -m "$MID" || { echo "FATAL: nvpmodel -m $MID failed"; exit 1; }
    sudo -n /usr/bin/jetson_clocks || echo "WARNING: jetson_clocks failed in mode $MID"
    sleep 10   # let clocks/thermals settle
    mkdir -p "$MDIR"
    { nvpmodel -q; echo; sudo -n /usr/bin/jetson_clocks --show; echo; nproc; } \
      > "$MDIR/mode_provenance.txt" 2>&1
    tegra_start "$MDIR/tegrastats_run.txt"
    run_timed_legs "$MDIR" || exit 1
    tegra_stop
  done
fi

# ── 4. package ───────────────────────────────────────────────────────────────
log "packaging"
finish; trap - EXIT
tar czf "${OUT}.tar.gz" "$OUT"
log "DONE -> bench/${OUT}.tar.gz  (send this file back)"
