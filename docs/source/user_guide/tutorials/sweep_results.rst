Backend Sweep Results
=====================

GLASS ships three interchangeable block-scoped backends plus a warp-scoped
surface, and which one is fastest depends on the operation, the matrix size
``N``, and the dtype. The **mega sweep** (``bench/run_mega_sweep.sh``) times all
of them head-to-head so the choice is data-driven rather than guessed — this is
exactly the measurement behind ``glass-defaults.cuh``'s ``suggested_backend<>()``
(see :doc:`../../api_reference/defaults`).

The figures and table below are from an RTX 5090 / sm_120 run in the throughput
regime (``NPROB=8192``). They are committed static assets — regenerate them for
your own hardware with::

   cd bench
   ./run_mega_sweep.sh
   python export_sweep_figures.py     # writes docs/source/_static/ assets

``bench/explore_sweep.ipynb`` is the interactive version of the same analysis.

The ladder — ns/problem vs N, per backend
------------------------------------------

Lower is faster. Each subplot is one op; the three curves are ``warp`` (green),
``block`` (blue), and ``nvidia`` / MathDx (red). The crossover points are where
``suggested_backend`` switches tiers.

**float32:**

.. image:: /_static/mega_sweep_ladder_f32.png
   :alt: f32 warp/block/nvidia ladder, ns/problem vs N per op
   :width: 100%

**float64:**

.. image:: /_static/mega_sweep_ladder_f64.png
   :alt: f64 warp/block/nvidia ladder, ns/problem vs N per op
   :width: 100%

Winner per (op, N)
------------------

The backend with the lowest ns/problem at each ``(op, N)`` — the table
``suggested_backend<>()`` encodes. The broad shape: tiny ``N`` favors ``warp``;
mid sizes favor ``nvidia`` for the factor/solve ops (chol/posv/trsv) once MathDx
amortizes; ``gemv`` crosses to ``block`` early; ``dot`` stays ``warp`` throughout.

.. literalinclude:: /_static/sweep_winners.txt
   :language: text

See :doc:`../concepts/tuning` for how to emit a per-host override table from a
sweep, and :doc:`../../api_reference/defaults` for the picker API.
