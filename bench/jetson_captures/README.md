# Jetson AGX Orin captures (sm_87)

Raw benchmark captures behind the paper's portability section and behind the
committed `ideal_sm87` / `body_sm87` dispatch tables. One board (AGX Orin,
16 SM, CUDA 13.2, L4T r39) measured at the three **standard** nvpmodel power
modes; MAXN is excluded by policy (owner's call — never run unconstrained).

Every timed capture ran with `jetson_clocks` pinned and its own provenance
(`nvpmodel -q`, `jetson_clocks --show`, `device_info`) captured beside it.

## What is here

| Path | Mode | Contents |
|---|---|---|
| `jetson_localhost.localdomain_20260801_1226/` | 30 W | full run: 6-section ladder, hostblas+latency, fusion, robotics, body, provenance |
| `50w_20260802_partial/` | 50 W | the 08-02 run: hostblas/fusion/robotics/provenance clean; ladder sections 1-3 spoiled (see below) |
| `topup50w_targeted_20260803_0936/` | 50 W | re-measurement of exactly the spoiled sections + a full body leg |
| `mega_sweep_50W_merged.txt` | 50 W | **the canonical 50 W ladder** — each section taken from whichever run measured it clean, with per-section provenance comments. This is what generated `ideal_sm87`. |
| `mega_sweep_15W_partial_20260801_2150.txt` | 15 W | 5 of 6 ladder sections (the run hit its wall-clock cap mid 8192-f64) |
| `tegrastats_summary_*.txt` | — | condensed board power / thermals / GPU-busy fraction (raw logs are ~13 MB each and are not shipped) |

## The 2026-08-02 contamination, and how it was bounded

An orchestrator bug (a `pkill` pattern that matched the shell running it) left a
15 W-launched sweep alive when its supervisor died; it kept running on the GPU
until 11:16 while the 50 W capture had already started — three hours of two
benchmarks sharing one device.

Rather than discard the whole capture, `check_zombie_contamination.py` bounds
the damage with a physical invariant: **50 W has strictly higher clocks than
30 W, so an uncontended 50 W section must be faster than the 30 W reference.**

| section | 50 W / 30 W (08-02) | verdict | after re-measurement |
|---|---|---|---|
| NPROB=64 f32 | 2.183 | contended | 0.782 |
| NPROB=64 f64 | 1.929 | contended | 0.760 |
| NPROB=1024 f32 | 1.812 | contended | 0.761 |
| NPROB=1024 f64 | 0.759 | clean | kept |
| NPROB=8192 f32 | 0.768 | clean | kept |
| NPROB=8192 f64 | 0.761 | clean | kept |

The break falls exactly at the zombie's death. The three clean sections are the
expensive ones (each 8192 section runs 1-3 h), so re-measuring only the spoiled
three took 1.5 h instead of ~8 h, and the re-measured sections land on the same
0.76-0.78 ratio as the sections that were never disturbed.

## Reproducing / extending

```bash
./check_zombie_contamination.py            # per-section sanity vs the 30W reference
./merge_50w_sections.py --selftest         # round-trip the section splitter
./merge_50w_sections.py                    # rebuild mega_sweep_50W_merged.txt
./summarize_tegrastats.py <log> --after 11:20   # power/thermal summary, optional window

# regenerate the shipped tables from a capture (⚠ pass --legs, or tune.py will
# start building the OTHER legs' binaries and run them live on this GPU)
cd ../.. && .venv/bin/python bench/tune.py --legs ladder \
    --from-ladder bench/jetson_captures/mega_sweep_50W_merged.txt --sm 870 --dry-run
```

`RESUME_topup_orchestrator.sh` drives a full multi-mode capture from the
desktop (mode switch → reboot → CUDA-ready gate → capture → pull). Traps worth
knowing before editing it are documented in its header comment.
