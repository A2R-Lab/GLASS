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

Then install the run deps and pin the clocks:

```bash
sudo apt install -y python3-numpy python3-matplotlib   # matplotlib optional
sudo nvpmodel -m 0        # MAXN power mode (the script warns if you skip this)
sudo jetson_clocks        # pin clocks for stable timing
```

## 1. Run

```bash
git clone https://github.com/A2R-Lab/GLASS && cd GLASS
./bench/run_jetson.sh --build-only    # optional separate compile pass (safe anytime)
./bench/run_jetson.sh                 # full capture: 1–3 h (Nano slowest), GPU otherwise idle
```

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

Troubleshooting:
- `could not detect GPU arch` from `paper_sweeps.py`: the script pins
  `--arch sm_87` already; if you see this you ran a driver by hand — pass
  `--arch sm_87`.
- OOM during builds on the Nano (8 GB): rerun `./bench/run_jetson.sh
  --build-only` after `sudo systemctl stop docker` (or close browsers);
  builds resume from the cache.
- A non-MAXN warning means the numbers will be low — fix the power mode and
  rerun; the tarball records whichever mode actually ran.
