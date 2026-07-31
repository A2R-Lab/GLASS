# Jetson benchmark runbook (paper portability leg)

One capture per box (Orin AGX / Orin NX / Orin Nano — all `sm_87`). Each run
produces a single `bench/jetson_<host>_<ts>.tar.gz` containing the timings
plus a full device/JetPack provenance bundle; send those back for ingestion
(figure columns + the `ideal_sm87` ladder table, spliced off-box via
`tune.py --from-ladder --sm 870`).

## 0. Pre-flight: get on the latest JetPack the box supports

We want each board on the newest JetPack it can run (all Orin-class boards
support JetPack 6.x / CUDA 12.x). Check what you have:

```bash
cat /etc/nv_tegra_release          # L4T release: r36.x = JetPack 6, r35.x = JetPack 5
sudo apt show nvidia-jetpack       # installed JetPack metapackage version
```

**Already on JetPack 6.x (L4T r36)** — upgrade in place with apt only:

```bash
sudo apt update
sudo apt full-upgrade              # accept the bootloader-update prompts
sudo apt install nvidia-jetpack
sudo reboot
```

6.0 → 6.1/6.2 includes a bootloader/QSPI update: say yes to the
`nvidia-l4t-bootloader` prompts and do not power-cut during the post-reboot
flash finalization.

**Still on JetPack 5.x (L4T r35)** — the 5→6 jump needs a reflash (~1 h/box):
use NVIDIA SDK Manager from an x86 host (recommended), or the L4T
apt-source migration documented by NVIDIA (edit
`/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` from `r35.x` to
`r36.x` per their upgrade note — riskier than SDK Manager). If reflashing is
not practical, benchmarking on 5.x is an acceptable fallback: CUDA 11.4
compiles `sm_87` fine, and the provenance bundle records exactly what ran.

Newer boards may already be past JetPack 6 (e.g. L4T r39 / CUDA 13.x, seen on
the AGX Orin dev kit mid-2026) — that's fine; newest available wins and the
provenance bundle records it. Note `nvcc` often isn't on `PATH` on JetPack
installs; `run_jetson.sh` auto-prepends `/usr/local/cuda/bin`.

Then install the run deps and pin the clocks. **Policy: we never run MAXN**
(unconstrained power/thermals; owner's call to protect the boards) — timed
captures use the standard wattage modes, with the highest wattage mode as the
headline (AGX Orin: 3=50W):

```bash
sudo apt install -y python3-numpy python3-matplotlib   # matplotlib optional
sudo nvpmodel -m 3        # highest STANDARD wattage mode (AGX: 50W; check nvpmodel -q)
sudo jetson_clocks        # pin clocks (within the mode's budget) for stable timing
```

**For power-mode sweeps (and for letting an agent drive the box over ssh),
grant passwordless sudo for exactly the two clock tools** (one-time, per box):

```bash
echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel, /usr/bin/jetson_clocks" \
  | sudo tee /etc/sudoers.d/glass-bench
```

## 1. Run

```bash
git clone https://github.com/A2R-Lab/GLASS && cd GLASS
./bench/run_jetson.sh --build-only         # optional separate compile pass (safe anytime)
./bench/run_jetson.sh                      # full capture at the CURRENT power mode
./bench/run_jetson.sh --power-modes all    # sweep the STANDARD wattage modes (needs the sudoers rule)
./bench/run_jetson.sh --power-modes 3,2,1  # or a subset by mode ID (AGX: 3=50W, 2=30W, 1=15W)
```

Edge deployments run at whatever power budget the platform allows, so the
paper wants the timed legs at each standard `nvpmodel` wattage mode (AGX Orin:
1=15W, 2=30W, 3=50W; NX/Nano have their own tables — `--power-modes all`
enumerates the non-MAXN modes from `/etc/nvpmodel.conf`; MAXN is never run,
per the policy above). The sweep compiles ONCE, then
per mode: switches with `nvpmodel -m`, pins clocks with `jetson_clocks`,
settles 10 s, records per-mode provenance (`nvpmodel -q`, clock readback,
online-CPU count) and a per-mode `tegrastats` log (power rails → energy/solve
and perf-per-watt), and re-runs all four timed legs into
`mode_<id>_<name>/` subdirs. The original power mode is restored on exit.

Nothing else should use the GPU or heavy CPU while it runs (timing
methodology). The script:

1. captures provenance: board model, L4T/JetPack, `nvcc`/`gcc`/kernel, RAM,
   CPUs, `nvpmodel -q`, `jetson_clocks --show`, a `cudaGetDeviceProperties`
   probe (SM count is what separates AGX/NX/Nano), and idle `tegrastats`;
2. builds everything for `sm_87`;
3. runs the timed legs serially — the 3-tier SIMT ladder
   (`tune.py --allow-no-mathdx`; MathDx does not ship for Tegra, which is
   itself a paper datum), the hostblas + single-call-latency + fusion
   harnesses (host cuBLAS/cuSOLVER are on JetPack), and the robotics
   micro-op sweep — with `tegrastats` logging alongside for energy/solve;
4. tars captures + provenance into `bench/jetson_<host>_<ts>.tar.gz`.

## 2. What comes back / what we do with it

| Capture | Feeds |
|---|---|
| `mega_sweep_*.txt` | `ideal_sm87` ladder table (spliced on the desktop via `python bench/tune.py --from-ladder <txt> --sm 870 --allow-no-mathdx`), paper §portability "which crossovers moved" |
| `paper_hostblas_*.txt` | Jetson columns for the hostblas + latency figures |
| `paper_fusion_*.txt` | Jetson fusion curves |
| `robotics_sweep_*.txt` | Jetson robotics tier panels |
| `body_dispatch_sweep_*.txt` | the sm_87 in-block body-dispatch table (the bare `glass::` face's Phase-2 `dispatch_body()` cells for this arch — same sweep the desktop quiet window runs) |
| `provenance/` | paper's exact hardware/software statement; AGX-vs-NX-vs-Nano scaling axis (SM count, clocks); energy/solve from `tegrastats_run.txt` |
| `mode_<id>_<name>/` (power sweeps) | one full capture set per power mode + `mode_provenance.txt` + per-mode `tegrastats` → the latency-vs-power-budget and perf-per-watt story |

Troubleshooting:
- `could not detect GPU arch` from `paper_sweeps.py`: the script pins
  `--arch sm_87` already; if you see this you ran a driver by hand — pass
  `--arch sm_87`.
- OOM during builds on the Nano (8 GB): rerun `./bench/run_jetson.sh
  --build-only` after `sudo systemctl stop docker` (or close browsers);
  builds resume from the cache.
- A non-MAXN warning means the numbers will be low — fix the power mode and
  rerun; the tarball records whichever mode actually ran.
