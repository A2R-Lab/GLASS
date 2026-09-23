Paper Results
=============

The paper **GLASS: Architecture-Tuned, Composable, Device-Side Linear Algebra
for Edge Robotics and Beyond** (Brian Plancher, 2026) evaluates reusable GPU
numerical infrastructure across three NVIDIA architectures. The
`research cover page <https://a2r-lab.org/GLASS/>`_ presents the full story,
figures, and case studies. The arXiv public release is pending.

This page summarizes the release at source commit ``afc0149``.
PDF figures are unchanged copies of the release; PNGs are web previews.
Their source commit and hashes are recorded in the
:download:`figure provenance manifest </_static/paper/provenance.json>`.
Earlier campaigns remain in :doc:`sweep_results`.

Methodology and populations
------------------------------------------------------------

* **Placement:** six operations, eleven sizes (4–128), three batches
  (64, 1024, 8192), and fp32/fp64: 396 cells per architecture.
* **Placement spread:** slowest / fastest valid measured tier time per cell over
  the 132 Orin cells at B=8192 across both precisions: median 4.9×, max 81×.
* **Device comparisons:** NVIDIA supports 195/198 fp32 Orin cells and 192/198
  RTX 5090 cells. Unsupported cells are excluded from these comparable pools.
* **Host comparisons:** GEMM, POTRF, POSV; nine sizes (4–64), seven batches
  (1, 4, 16, 64, 256, 1024, 8192), both precisions: 378 cells. GLASS candidates
  are dependency-free native implementations. Framework comparison uses the
  per-cell better of PyTorch and JAX with device-resident inputs/outputs and
  precompiled/jitted code.

Orin (sm_87, CUDA 13.2) is primary, with Xavier (sm_72, CUDA 11.4) and RTX 5090
(sm_120, CUDA 13.2) comparisons. Xavier has no evaluated NVIDIA device-library
candidates. Numerical validation is independent of timing; destructive solver
benchmarks consume fresh valid inputs. Timing protocols differ by benchmark;
see the paper methodology rather than treating normalized ns/problem as
synchronized end-to-end call latency.

Execution placement
-------------------

Orin placements differ from RTX 5090 in 145/396 cells and from Xavier in
162/396. Restricting all GPUs to native candidates changes 107/396 RTX–Orin,
94/396 RTX–Xavier, and 66/396 Orin–Xavier choices. These full-sweep statistics
include both precisions; the figure below shows fp32.

.. figure:: /_static/paper/tier_heatmap_three_arch.png
   :alt: Three GPUs by three batch regimes; outlines mark recommendations differing from Orin.
   :target: ../../_static/paper/tier_heatmap_three_arch.pdf

   Orin is the center shaded row. Colors identify execution tiers.

Transferring native placements and launch configurations between architectures
adds 4–20% geometric-mean runtime; 95th-percentile penalties reach 2.59×.
One Orin fp64 GEMM launch exceeds RTX shared-memory capacity, leaving 395
transferable cells in that direction.

On the measured Orin, reducing the power ceiling from 50 W to 30 W and 15 W
increases geometric-mean runtime by 29% and 91%, while each changes 9/396
recommendations. This supports reuse within these measured power modes.
Complete tuning took 5.5 hours on Orin, 9 on Xavier, and 2 on RTX 5090.

Host libraries and compiler frameworks
------------------------------------------------------------

GLASS is faster than the per-cell best of PyTorch/JAX in 359/378 Orin cells
across both precisions, with reported geometric means of 4.3–9.5× and a
73× peak. RTX 5090 wins 320/378 at 1.5–2.6× geometric means and up to 12×.

.. figure:: /_static/paper/host_performance_orin_heatmap.png
   :alt: Orin fp32 baseline/GLASS ratios for GEMM, POTRF, POSV against host vendors and frameworks.
   :target: ../../_static/paper/host_performance_orin_heatmap.pdf

   Above one favors GLASS. **Colors are clipped at 16×; the true maximum
   is 113.9×.** This fp32 figure is a subset of the 378-cell comparison.

In **fp32**, GLASS beats cuBLAS/cuSOLVER at every measured batch for POSV
through N=64, GEMM through N=48, and POTRF through N=24, reaching 113.9×,
12.1×, and 12.8× respectively. These universal ranges do not extend to fp64.
The 73× framework peak is fp32 POSV at N=4, B=8192. Larger problems include
regions where host libraries or compiler frameworks regain an advantage.

Robotics operations and composition
-----------------------------------

Representative Orin fp32 robotics operations at B=4096 gain 3.5–9.7× from the
best fine-grained placement versus whole-block execution. Thread scope wins
most rows; fp32 softmax favors a warp.

.. figure:: /_static/paper/riccati_tradeoff_orin.png
   :alt: Unfused/fused GLASS Riccati runtime ratios; small batches can favor fusion and larger batches favor separate kernels.
   :target: ../../_static/paper/riccati_tradeoff_orin.pdf

   Above one favors fusion. Solid curves are fp32; dashed curves fp64.

Fusion improves the measured Riccati composition by up to 1.75× at small
batches. Unfused GLASS wins by up to 2.9× at larger batches and in resource-heavy
fp64 cases. This comparison isolates composition within GLASS rather than
attributing the full host-vendor gap to fusion.

Integrations in published robotics systems
------------------------------------------

* **Sampling-based MPC:** replacing bespoke reductions exposed a numerical
  bug whose repair was accepted upstream. Relative to the corrected baseline,
  GLASS speeds the replaced computation by 1.21–1.50× on Orin over 128–8192
  rollouts (1.21–1.46× on RTX 5090).
* **Batched IK:** approximately 355 lines of specialized numerics become 29
  lines of application code (92% less). At batch 2,000, Orin runtime changes
  from 9.94 ms to 7.92 ms (1.26×); RTX 5090 improves by 1.02×.

See :doc:`../concepts/tuning` for retuning and
:doc:`../concepts/testing_oracles` for independent numerical validation.
