#!/usr/bin/env bash
# 50W TARGETED top-up — replaces the full 7-8 h re-run.
#
# check_zombie_contamination.py proved (50W-vs-30W winner-time ratio per
# section) exactly which parts of the 2026-08-02 50W capture the zombie
# tune.py spoiled: 50W must be FASTER than 30W, so a section at ratio ~1.8-2.2
# was sharing the GPU.
#   NPROB=64 f32    2.183  CONTENDED  -> re-measure
#   NPROB=64 f64    1.929  CONTENDED  -> re-measure
#   NPROB=1024 f32  1.812  CONTENDED  -> re-measure
#   NPROB=1024 f64  0.759  CLEAN      -> keep from 08-02
#   NPROB=8192 f32  0.768  CLEAN      -> keep  (these are the 1-3 h sections)
#   NPROB=8192 f64  0.761  CLEAN      -> keep
# The expensive 8192 sections are exactly the clean ones, so this runs ~1.5 h
# instead of ~8 h. Body leg: only its last section (8192 f64) was truncated at
# 29/32 rows by the 18:01 cap; the other five are complete and post-zombie.
# hostblas/fusion/robotics all ran 17:28+ (zombie died 11:16) = clean, kept.
# Merge with merge_50w_sections.py on the desktop after this lands.
set -u
cd "$(dirname "$0")"   # bench/
export PATH=/usr/local/cuda/bin:$PATH PYTHONUNBUFFERED=1
TS=$(date +%Y%m%d_%H%M)
DEST="topup50w_targeted_${TS}"; mkdir -p "$DEST"
MEGA=.tune_cache/sm870/mega_sweep_fatbin_08abe8616ccb

nvpmodel -q > "$DEST/nvpmodel.txt" 2>&1 || true
sudo -n /usr/bin/jetson_clocks --show > "$DEST/jetson_clocks.txt" 2>&1 || true
./build/device_info > "$DEST/device_info.txt" 2>&1 || true

MTXT="$DEST/mega_sweep_50w_recut_${TS}.txt"
echo "# mega_sweep 50W re-measured contended sections  $(date)  (topup_50w_targeted.sh)" > "$MTXT"
for SPEC in "64 200 f32" "64 200 f64" "1024 100 f32"; do
  set -- $SPEC; NPROB=$1; REPS=$2; DT=$3
  echo "== section NPROB=$NPROB $DT start $(date +%H:%M:%S)"
  { echo
    echo "# section start $(date '+%F %T')"
    echo "################ NPROB=$NPROB  reps=$REPS  dtype=$DT ################"
    "$MEGA" "$NPROB" "$REPS" "$DT"
  } >> "$MTXT" || { echo "FATAL: ladder section NPROB=$NPROB $DT failed"; exit 1; }
  echo "   section saved -> $MTXT  $(date +%H:%M:%S)"
done
grep -q '||' "$MTXT" || { echo "FATAL: ladder recut parsed empty"; exit 1; }

echo "== body leg: full re-run (cheap, ~20 min) for one consistent capture"
BODYTXT="$DEST/body_dispatch_sweep_${TS}.txt"
{ echo "# capture $(basename "$BODYTXT") | $(hostname) | sm_87 jetson 50W targeted"
  for NR in 64:200 1024:100 8192:50; do
    NPROB="${NR%:*}"; REPS="${NR#*:}"
    for DT in f32 f64; do
      echo; echo "== NPROB=$NPROB reps=$REPS dtype=$DT =="
      ./build/bench_body_dispatch_sm_87 "$NPROB" "$REPS" "$DT" || exit 1
    done
  done
} > "$BODYTXT" || { echo "FATAL: body leg failed"; exit 1; }
[[ $(grep -c '||' "$BODYTXT") -ge 192 ]] || { echo "FATAL: body capture short ($(grep -c '||' "$BODYTXT") rows, want 192)"; exit 1; }
echo "   saved -> $BODYTXT  $(date +%H:%M:%S)"

tar czf "topup50w_targeted_${TS}.tar.gz" "$DEST"
echo "=== DONE -> bench/topup50w_targeted_${TS}.tar.gz  (send this file back) ==="
