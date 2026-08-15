Contraction-parallel ops (the ``*_reduced`` family)
===================================================

Every default L2/L3 product in GLASS maps **one thread to one output element**
and walks the contraction dimension *serially* inside that thread:
``gemm`` / ``gemv`` / ``syrk`` all loop ``for k: acc += A[..]*B[..]``. When the
output count is large that saturates the block. When the output count is *small*
— a 7×7 control Hessian, a length-14 mat-vec — most of the block sits idle while
a handful of threads grind through the sum.

The ``*_reduced`` family flips the mapping: **one warp owns one output**, and its
32 lanes split the contraction, combining with a single warp-shuffle reduce
(``glass::warp::reduce``). The engine appears as:

- ``glass::gemm_reduced`` — the core (and the mechanism behind FR-4);
- ``glass::gemv_reduced``, ``glass::syrk_reduced`` — the L2 / SYRK siblings;
- the tensor and congruence families (``tensor_vec_contract``, ``vec_tensor_vec``,
  ``congruence_sym``, ``bilinear``) — products the serial BLAS surface cannot
  express in one call, built on the same engine.

All ship in the three SIMT surfaces (``glass::`` block, ``glass::warp::``,
``glass::cgrps::``).

.. warning::

   **Explicit opt-in, not a general throughput default.** On the 2026-08-14
   quiet RTX 5090 (sm_120) sweep, reduced cleared the ±5% decision margin in
   0/48 f32 cells and 2/48 f64 cells. Both f64 wins were the same 4×4×64
   shape at 128/256 threads (full table: ``bench/RESULTS.md``). The
   dtype-independent ``glass::suggested_use_reduced`` picker therefore
   declines it everywhere rather than regressing f32.
   **Prefer the plain ops** (``gemm`` / ``gemv`` / ``syrk``) for throughput;
   reach for this family only for the fused forms (``tensor_vec_contract``,
   ``vec_tensor_vec``, ``congruence_sym``, ``bilinear``) that the serial surface
   cannot express in one call.

The honest win-condition
------------------------

The total multiply-add work is **identical** to the serial op — the contraction
is the same length either way. The *only* thing ``*_reduced`` buys is thread
**utilization**, and only when both of these hold:

#. **The output count is smaller than the block** (``n_out < blockDim``) — so the
   serial path would leave threads idle, and there is spare parallelism for the
   warp-per-output mapping to soak up.
#. **The contraction K amortizes the shuffle tail** — the warp reduce costs a
   ~5-step ``__shfl_down`` tail per output, so K must be large enough (roughly the
   14–21 range of a trajectory-optimization knot) for the split to pay for it.

When ``n_out >= blockDim`` with a small K, ``*_reduced`` is **neutral or slower**
— the serial op already keeps every thread busy and avoids the shuffle. So this
is **opt-in**: the default ops are unchanged, and a caller (or GRiD-style codegen)
chooses ``*_reduced`` only where it wins.

What the measurement actually says
----------------------------------

The crossover sweep (``bench/bench_reduced.cu``, full table in
``bench/RESULTS.md`` (reduced section)) was run on a quiet **RTX 5090 / sm_120**. The
result is blunt: **the contraction-parallel path does not clear the serial
baseline in 94 of 96 configurations**. The serial
``gemm`` over shared-resident data is a tight per-thread loop that is very hard to
beat at these sizes, while ``*_reduced`` pays a ~5-step shuffle latency per output
and, at the typical short contraction (K = 14–21), leaves most of a warp's lanes
idle. The two wins are f64 4×4×64 at 128/256 threads (1.41× and 1.97×).
That single dtype-specific shape is worth a future targeted sweep, but it does
not justify a general or dtype-blind default.

``glass::suggested_use_reduced`` encodes that measurement — it returns
``false`` unconditionally on sm_120. Its ``<n_out, K_contract, blockDim>``
signature has no scalar-type parameter, so it cannot safely encode the observed
f64-only corner:

.. code-block:: cuda

   if constexpr (glass::suggested_use_reduced<n_out, K, blockDim>())
       glass::gemm_reduced<float, M, N, K>(1.f, A, B, 0.f, C);
   else
       glass::gemm<float, M, N, K>(1.f, A, B, 0.f, C);

.. note::

   The tensor / congruence families (``tensor_vec_contract``, ``vec_tensor_vec``,
   ``congruence_sym``, ``bilinear``) share this engine and so inherit the same
   overhead. Their value is **expressiveness and fusion** — operations the serial
   surface cannot express in one call — not beating a hand-tuned serial loop. If
   you are optimizing for latency, benchmark against your own serial code first.

Thread-count invariance
-----------------------

Unlike the ``_fast`` shuffle reductions (whose summation grouping varies with
``blockDim``), the ``*_reduced`` ops are **thread-count invariant**:
identical output at 1 thread, a partial warp, or many warps. Each output is
reduced by the *same* fixed 32-way tree regardless of how many warps the block
has — a trailing partial warp (``blockDim % 32``) idles, and below 32 threads a
register path (``reduced_tree32``) reproduces the warp-shuffle summation
order **bit-for-bit**, so the result does not change across the 32-thread
boundary.

Within a surface this is bit-identical. *Across* surfaces (block vs warp vs
cgrps) the single-step ops agree bit-for-bit, while the composed two-step ops
(``congruence_sym``, ``bilinear``, ``riccati_gain``) agree only to floating-point
tolerance, because their intermediate gemm may fuse its FMA differently per
instantiation. Both are correct; only the rounding of the last bit differs.
