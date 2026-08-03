#!/usr/bin/env bash
# v7 orchestrator (persistent-disk; survives /tmp wipes and session ends):
#     nohup setsid ~/Desktop/GLASS/bench/jetson_captures/RESUME_topup_orchestrator.sh \
#         > ~/Desktop/GLASS/bench/jetson_captures/resume_topup.log 2>&1 &
#
#   leg A  mode 3 -> topup_50w_targeted.sh   (3 contended ladder sections +
#          body leg; ~1.5 h — see check_zombie_contamination.py for why only
#          these; the expensive 8192 sections were measured clean on 08-02)
#   leg B  mode 1 -> topup_15w.sh            (8192-f64 + the four short legs)
#   then   restore mode 2 (30W default) and pull every tarball back.
#
# Traps baked in: mode switches REBOOT the AGX (nvpmodel prompts YES); CUDA
# lags ssh after boot (device_info gate); pkill patterns are bracket-armored
# (v4's self-match killed the killing shell and orphaned a sweep that then
# contended the next capture for 3 h); stall detection is crash-only because
# one 8192 section is legitimately silent for 1-3 h.
set -u
J="ssh -o BatchMode=yes -o ConnectTimeout=15 jetson-orin"
CAPT=~/Desktop/GLASS/bench/jetson_captures
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

kill_capture() {
  $J 'pkill -f "topup_15[w]"; pkill -f "topup_50[w]"; pkill -f "run_jetso[n]"; pkill -f "bench_pape[r]"; pkill -f "bench_robotic[s]"; pkill -f "bench_bod[y]"; pkill -f "mega_sweep_fatbi[n]"; pkill -f "tune.p[y]"; true' >/dev/null 2>&1
  sleep 3
  local left
  left=$($J 'pgrep -af "tune.p[y]|mega_sweep_fatbi[n]|bench_bod[y]|topup_" | grep -v "pgrep" | head -3' 2>/dev/null)
  [[ -n "$left" ]] && log "WARN: survivors after kill: $left"
}

wait_ssh_mode() {
  local want="$1" i cur
  for i in $(seq 1 60); do
    cur=$($J 'nvpmodel -q 2>/dev/null | awk "NR==2{print \$1}"' 2>/dev/null)
    [[ "$cur" == "$want" ]] && return 0
    sleep 20
  done
  return 1
}

wait_cuda_ready() {
  local i
  for i in $(seq 1 30); do
    $J '~/GLASS/bench/build/device_info >/dev/null 2>&1' 2>/dev/null && return 0
    sleep 20
  done
  return 1
}

set_mode() {
  local m="$1"
  log "switching to mode $m (kills any capture; may reboot the box)"
  kill_capture
  $J "echo YES | sudo -n /usr/sbin/nvpmodel -m $m" >/dev/null 2>&1
  sleep 30
  wait_ssh_mode "$m" || { log "FATAL: box not back in mode $m"; return 1; }
  wait_cuda_ready    || { log "FATAL: CUDA never came up in mode $m"; return 1; }
  $J 'sudo -n /usr/bin/jetson_clocks' >/dev/null 2>&1 || log "warn: jetson_clocks failed"
  sleep 10
}

poll_capture() {  # $1 = log name, $2 = bracketed script pattern, $3 = cap (x2min)
  local lg="$1" pat="$2" cap="$3" i
  for i in $(seq 1 "$cap"); do
    sleep 120
    if $J "grep -qE 'DONE ->' ~/GLASS/bench/${lg}" 2>/dev/null; then
      log "capture DONE"; $J "grep 'DONE ->' ~/GLASS/bench/${lg}"; return 0
    fi
    if $J "grep -qE 'FATAL' ~/GLASS/bench/${lg}" 2>/dev/null; then
      log "capture FAILED:"; $J "tail -6 ~/GLASS/bench/${lg}"; kill_capture; return 1
    fi
    if ! $J "pgrep -f '${pat}' >/dev/null" 2>/dev/null; then
      sleep 10
      $J "grep -qE 'DONE ->' ~/GLASS/bench/${lg}" 2>/dev/null && { log "capture DONE"; return 0; }
      log "capture process DIED without a verdict:"; $J "tail -6 ~/GLASS/bench/${lg}"; return 1
    fi
  done
  log "hard cap hit — killing capture (sections persisted)"
  kill_capture; return 1
}

log "waiting for jetson-orin to be reachable"
until $J true 2>/dev/null; do sleep 60; done
log "box reachable"

FAILED=0
# ── leg A: 50W targeted (~1.5 h; cap 4 h) ──
if set_mode 3; then
  log "launching 50W TARGETED top-up (3 contended ladder sections + body leg)"
  $J 'cd ~/GLASS && rm -f bench/topup50w_run.log && nohup ./bench/topup_50w_targeted.sh > bench/topup50w_run.log 2>&1 & echo launched'
  poll_capture topup50w_run.log 'topup_50w_targete[d]' 120 || FAILED=1
else
  FAILED=1
fi

# ── leg B: 15W top-up (~6-8 h; cap 12 h) ──
if set_mode 1; then
  log "launching 15W top-up (8192-f64 + short legs)"
  $J 'cd ~/GLASS && rm -f bench/topup15w_run.log && nohup ./bench/topup_15w.sh > bench/topup15w_run.log 2>&1 & echo launched'
  poll_capture topup15w_run.log 'topup_15[w]' 360 || FAILED=1
else
  FAILED=1
fi

log "restoring default MODE_30W (2)"
set_mode 2 || log "warn: could not restore mode 2"

mkdir -p "$CAPT/raw_sweeps"
scp -o BatchMode=yes "jetson-orin:GLASS/bench/topup*.tar.gz" "$CAPT/" 2>/dev/null
scp -o BatchMode=yes "jetson-orin:GLASS/bench/mega_sweep_2026080*.txt" "$CAPT/raw_sweeps/" 2>/dev/null
log "capture dir now:"; ls -la "$CAPT"
log "v7 finished (failures=$FAILED)"
exit $FAILED
