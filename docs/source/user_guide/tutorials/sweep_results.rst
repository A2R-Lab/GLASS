Backend Sweep Results
=====================

GLASS ships three interchangeable block-scoped backends plus a warp-scoped
surface, and which one is fastest depends on the operation, the matrix size
``N``, and the dtype. The **mega sweep** (``bench/tune.py``'s ladder leg) times all
of them head-to-head so the choice is data-driven rather than guessed — this is
exactly the measurement behind ``glass-defaults.cuh``'s ``suggested_backend<>()``
(see :doc:`../../api_reference/defaults`).

The figures and table below are from an RTX 5090 / sm_120 run, shown across three
batch regimes — **NPROB=64** (low batch, latency-leaning), **NPROB=1024** (mid),
and **NPROB=8192** (the throughput regime that feeds the dispatch tables). The
winner can shift with batch size: at low batch the vendor (``nvidia``) path often
wins the factor/solve ops on launch-amortized latency, while at high batch the
hand-rolled SIMT paths scale back in. They are committed static assets —
regenerate them for your own hardware with::

   python bench/tune.py --sm auto      # remeasures + regenerates tables AND figures
   # or just the figures from an existing sweep .txt:
   python bench/export_sweep_figures.py bench/mega_sweep_*.txt

``bench/explore_sweep.ipynb`` is the interactive version of the same analysis.

The ladder — ns/problem vs N, per backend
------------------------------------------

Lower is faster. Each subplot is one op; the three curves are ``warp`` (green),
``block`` (blue), and ``nvidia`` / MathDx (red). The crossover points are where
``suggested_backend`` switches tiers. ``suggested_backend<>()`` is keyed on the
**NPROB=8192** throughput regime; the 64/1024 figures show how the crossovers
move at smaller batch.

float32
~~~~~~~

.. image:: /_static/mega_sweep_ladder_f32_n64.png
   :alt: f32 ladder, NPROB=64
   :width: 100%

.. image:: /_static/mega_sweep_ladder_f32_n1024.png
   :alt: f32 ladder, NPROB=1024
   :width: 100%

.. image:: /_static/mega_sweep_ladder_f32_n8192.png
   :alt: f32 ladder, NPROB=8192 (throughput — feeds suggested_backend)
   :width: 100%

float64
~~~~~~~

.. image:: /_static/mega_sweep_ladder_f64_n64.png
   :alt: f64 ladder, NPROB=64
   :width: 100%

.. image:: /_static/mega_sweep_ladder_f64_n1024.png
   :alt: f64 ladder, NPROB=1024
   :width: 100%

.. image:: /_static/mega_sweep_ladder_f64_n8192.png
   :alt: f64 ladder, NPROB=8192 (throughput — feeds suggested_backend)
   :width: 100%

Winner per (op, N), per regime
------------------------------

The backend with the lowest ns/problem at each ``(op, N)``, listed for all three
NPROB regimes — the ``NPROB=8192`` block is what ``suggested_backend<>()``
encodes. The broad shape at high batch: tiny ``N`` favors ``warp``; mid sizes
favor ``nvidia`` for the factor/solve ops (chol/posv/trsv) once MathDx amortizes;
``gemv`` crosses to ``block`` early; ``dot`` stays ``warp`` throughout. At
``NPROB=64`` the ``nvidia`` band widens (launch latency dominates, so the vendor
kernels win sooner).

.. literalinclude:: /_static/sweep_winners.txt
   :language: text

See :doc:`../concepts/tuning` for how to emit a per-host override table from a
sweep, and :doc:`../../api_reference/defaults` for the picker API.
