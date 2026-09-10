#!/usr/bin/env bash
# Staged quiet-window confirmation capture: the maintained mega-sweep protocol
# with randomized per-cell measurement order (paper RUN item, ordering-bias
# check). bench_mega_sweep.cu executes each row's (contender, launch-shape)
# cells in shuffled order under GLASS_SHUFFLE_ORDER=<seed> while printing the
# canonical row layout, so the capture parses identically and picks can be
# compared cell-by-cell against the pinned in-order capture
# (mega_sweep_20260815_205919). Everything else is run_mega_sweep.sh verbatim.
#
# Usage (idle GPU, quiet box — takes hours, same as any full mega sweep):
#   cd bench && nohup ./run_shuffled_confirmation.sh \
#       > shuffled_confirm_$(date +%Y%m%d_%H%M).log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

# refuse to start on a busy GPU — a contended capture is worthless
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  echo "ERROR: GPU has active compute processes — aborting (quiet box required)" >&2
  exit 1
fi

export MATHDX_ROOT="${MATHDX_ROOT:-/opt/nvidia/mathdx/26.03}"
export GLASS_SHUFFLE_ORDER="${GLASS_SHUFFLE_ORDER:-20260820}"
echo "==> shuffled-order confirmation capture (seed=$GLASS_SHUFFLE_ORDER)"
exec ./run_mega_sweep.sh sm_120
