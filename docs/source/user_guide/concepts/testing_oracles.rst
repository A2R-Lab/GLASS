Testing, oracles & receipts
===========================

Every Doxygen-documented public overload is compile-covered by a CUDA test TU
(the 100% overload badge), while each behavioral family is exercised on a GPU
against the obligations below. Every push to ``main`` that touches library or
test sources carries a **signed hardware receipt**
(`pytest-gpu-proof <https://pypi.org/project/pytest-gpu-proof/>`_): the selected
suite ran on a physical GPU at that source tree, attested and re-verified by CI.
This page states *how* each family is validated so the compile-coverage number
is never confused with numerical depth.

Coverage model
--------------

``test/api-contracts.json`` is generated from Doxygen XML. Its stable unit is a
public function overload—not a name—matched to a compatible call in
``test/cuda`` or an example. The maintained
``test/api-coverage-policy.json`` lists internal implementation symbols excluded
from the supported surface, with a reason for every exclusion. Compile-only
canary TUs close overload-shape gaps; they do not claim numerical correctness.
Numerical correctness, dtype/layout coverage, conditioning, thread-count
invariance, and cross-tier agreement are separate required obligations backed
by the GPU receipt.

Validation classes
------------------

Each op family falls into one (often several) of four classes, in decreasing
order of independence:

**A. External oracle** — output compared elementwise against an independently
maintained reference implementation on the same inputs.

**B. Identity / finite difference** — validated by a defining mathematical
property: derivative ops against central differences of the *adjacent-order
device output* (in f64), factorizations by reconstruction (e.g.
:math:`\|LL^\top - A\| / \|A\|` at machine precision), solves by the
normalized backward residual, inverses by round trip.

**C. Construction mirror** — compared against a NumPy transcription of the
same published formula (used where no independent library implements the op).
A shared misreading of the convention could pass both sides, which is why the
Lie family additionally carries a class-A Pinocchio leg (below).

**D. Contract pin** — host-side query/size helpers (``*_scratch_bytes``,
``required_smem``, ``should_use_*``, dispatch-table lookups) are pinned to
their documented values by ``static_assert`` at compile time. These are
contract locks, not numerical comparisons — appropriate for functions whose
"correct answer" *is* the documented contract.

Oracles by family
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Family
     - Class
     - Reference
   * - L1 (``dot``/``reduce``/``nrm2``/``asum``/``axpy``/…)
     - A
     - NumPy (``np.dot``, norms, elementwise), incl. a mixed 1e3/1e-3
       magnitude input kind to stress accumulation order.
   * - L2 (``gemv``/``ger``/``trmv``/``trsv``)
     - A
     - NumPy ``@`` / ``scipy.linalg.solve_triangular``.
   * - L3 ``gemm`` (+ batched/strided)
     - A
     - NumPy ``@`` **plus** an independent hand-written triple-loop reference
       (guards against sharing a dim-mapping bug with ``@``); full
       shape × transpose × layout × alpha/beta grid.
   * - ``potrf``/``posv``/``potrs``
     - A + B
     - ``np.linalg.cholesky``, ``scipy.linalg.cho_solve``; backward-residual
       checks under 1e∓3 input scale and cond ≈ 1e6 spectra.
   * - ``getrf``/``getrs``/``gesv``/``laswp``
     - A
     - ``scipy.linalg.lu_factor`` convention exactly — our (LU, piv) is
       consumed by ``scipy.linalg.lu_solve`` as a drop-in; includes a
       zero-leading-pivot matrix that unpivoted LU fails on.
   * - ``ldlt`` (± Bunch–Kaufman)
     - A + B
     - ``scipy.linalg.ldl`` + block-diagonal reconstruction
       :math:`P A P^\top = L D L^\top`; indefinite spectra, zero diagonals.
   * - ``trsm``, ``inv``/``inv_pivoted``
     - A + B
     - ``np.linalg.inv`` / backward residuals; geometric-diagonal
       (cond ≈ 1e6) triangular solves.
   * - ``syev``/``eigh``/``eig3``/``svd3``/``psd_project``
     - A
     - ``np.linalg.eigh``/``eigvalsh``/``svd`` (eigenvector sign/order
       handled structurally), controlled-spectrum draws.
   * - ``pcg``/``bdmv``, ``internal::box_qp``
     - A + B
     - Dense assembly of the block-tridiagonal operator vs ``np.linalg.solve``;
       QP KKT conditions vs ``scipy.optimize.minimize``.
   * - Quaternion family
     - A
     - ``scipy.spatial.transform.Rotation`` (double cover handled by explicit
       sign alignment, both ``QuatLayout`` storages).
   * - SO(3)/SE(3) tangent maps (``so3_exp``/``log``/jacobians,
       ``se3_retract``/``se3_difference``/``dIntegrate`` jacobians/hessians)
     - A + B + C
     - **Pinocchio** (``pin.integrate``/``difference``/``dIntegrate`` on a
       free-flyer model; ``exp3``/``log3``/``Jexp3``/``Jlog3``) + f64 central
       differences of the defining composition identities + NumPy mirrors;
       inputs bracket every series threshold (0, 1e-9, both sides of the
       1e-8/1e-4 switches, near-π).
   * - Spatial 6-D algebra, cones/AL, sphere collision, softmax/argreduce
     - C (+ A)
     - NumPy transcriptions of the published formulas (Featherstone spatial
       algebra; cone projection by region enumeration); ``scipy.special``
       for softmax/logsumexp; FD for the AL/interval derivative chain.
   * - Host query/size helpers, dispatch tables
     - D
     - ``static_assert`` pins in ``test/cuda/test_defaults.cu`` and the
       ``test_nvidia_*`` drivers.

Structural properties (enforced for every op)
---------------------------------------------

These hold regardless of oracle class and are what make the suite hard to fool:

- **Thread-count invariance** — block ops are re-run across thread counts
  (1, partial warps, 32, 64, 256; the full sweep adds ragged counts like
  7/31/33/57/96) and must produce **byte-identical** output. This catches the
  #1 single-block bug class (missing barriers) that any fixed-configuration
  test misses.
- **Cross-tier agreement** — ``thread::``/``warp::``/``block::`` instantiate
  the same serial core; tiers are compared to ULP-level bounds on identical
  inputs. The bare dispatched face is pinned to be the same entity as the
  block tier where no cell moved.
- **Many problems, seeded randomness** — robotics ops run 53 problems per
  test (odd/ragged in every launch geometry); linear-algebra ops sweep sizes
  including 1 and non-square all-distinct dims. All draws come from seeded
  generators (robotics reseeds per-test from the test's node id), so any
  failure replays exactly.
- **Input energy & conditioning** — draws are mean-zero (mixed sign) at unit
  scale, with dedicated sweeps at 1e∓3 input scale (relative-accuracy
  preservation), mixed 1e3/1e-3 magnitude vectors, cond ≈ 1e6 spectra
  (residual-based checks — backward stability is asserted, forward error at
  ``cond·eps`` is not penalized), and the Lie-series branch points listed
  above.
- **Documented range contracts** — where GLASS intentionally trades
  robustness for speed, the boundary is documented rather than tested around:
  the ``nrm2``/``vector_norm`` family uses a naive sum of squares (no
  LAPACK-style ``snrm2`` scaling), so its intermediate overflows for
  ``‖x‖ ≳ 1e19`` (f32) / ``1e154`` (f64); the test suite exercises inputs
  well inside that contract, matching the header doc-comments.
- **No-read guarantees** — ``beta = 0`` paths run against NaN-poisoned
  buffers; triangular ops run with NaN-poisoned dead triangles, so any stray
  read fails loudly.

Receipt shards
--------------

``test/run_gpu_proof.sh`` emits eight independently fingerprinted shards:
``vector``, ``dense``, ``factor``, ``tiers``, ``solvers``, ``robotics``,
``mathdx``, and ``integration``. A local change reruns only the affected
families; untouched shards may carry forward only when their source-and-test
fingerprint matches and their attested commit is an ancestor. Release runs do
not carry results: all eight shards rerun at the release commit. The merged
receipt records each shard's test count and duration, which makes future shard
rebalancing evidence-based.

What is *not* oracle-tested
---------------------------

Honesty requires the short list: the class-D contract pins above (query
helpers — their spec *is* the pinned value), and performance-related claims
(dispatch-table *choices* are measured by ``bench/tune.py``, not asserted by
the correctness suite; the suite only pins that every choice is numerically
correct). Everything else on the public surface traces to class A, B, or C —
and every A/B/C comparison in the table runs on real hardware under the
receipt.
