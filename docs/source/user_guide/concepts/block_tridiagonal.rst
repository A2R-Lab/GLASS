Block-tridiagonal Solves
========================

The linear systems at the heart of trajectory optimization and model-predictive
control — the KKT / Schur-complement systems of a multi-knot problem — are
**block-tridiagonal** and symmetric positive definite. GLASS provides the two
single-block primitives needed to solve them on the GPU one problem per block:
:doc:`../../api_reference/banded` (the matvec) and
:doc:`../../api_reference/pcg` (a preconditioned conjugate-gradient solver).

The ``[L|D|R]`` block-tridiagonal layout
----------------------------------------

A block-tridiagonal matrix with ``knot_points`` block-rows of size
``state_size`` is stored as ``knot_points`` contiguous **row-major strips**. Each
strip is a ``state_size × (3·state_size)`` tile holding the three blocks of that
block-row side by side — left, diagonal, right:

.. code-block:: text

   block-row br:   [  L_br  |  D_br  |  R_br  ]      (each state_size × state_size)
                    \________________________/
                       3·state_size columns, row-major

   full matrix strips:  [row 0][row 1]...[row knot_points-1]   contiguous

The block-row at index ``br`` starts at
``s_matrix + br · (3·state_size) · state_size``. ``L`` of the first row and ``R``
of the last row are off the matrix; they are stored but multiply the zero pad
(below), so no edge special-casing is needed.

The padded vector layout
-------------------------

Vectors use a **padded** layout of length ``(knot_points + 2)·state_size`` — one
``state_size`` pad block on each end:

.. code-block:: text

   [ pad ][ x_0 ][ x_1 ] ... [ x_{kp-1} ][ pad ]
     ^^^                                    ^^^
     must be zero                      must be zero

Block-row ``br`` reads the window
``s_vector[br·state_size : br·state_size + 3·state_size)`` — i.e. the previous /
current / next state blocks — and writes its result at
``s_output[(br+1)·state_size + row]``. **The caller must pre-zero the leading and
trailing pad blocks**: that is what lets the first/last block-rows multiply their
absent ``L``/``R`` against zero instead of branching.

Matvec — ``glass::bdmv``
--------------------------------

``glass::bdmv`` computes ``s_output = A_bd · s_vector`` in one block, thread-count
invariant. A second overload writes the identical result into two buffers at once
(used by the PCG initialization ``z = p = Pinv·r``). It emits **no trailing**
``__syncthreads()`` — barrier before reusing the output, per the GLASS surface
convention.

.. code-block:: cpp

   // one block; threads stride over rows
   glass::bdmv<float, knot_points, state_size>(s_out, s_matrix, s_vec);
   __syncthreads();

Solve — ``glass::pcg``
-----------------------------

``solve`` runs preconditioned conjugate gradient on ``S x = b`` with the
block-tridiagonal preconditioner ``Pinv`` (both in ``[L|D|R]`` form). Convergence
is tested on the preconditioned residual ``rho = rᵀz``:
``|rho| < abs_tol + rel_tol·|rho_init|``.

Sizing and launch:

* Dynamic shared memory =
  ``glass::pcg_smem_size<T, state_size, knot_points>(threads)`` elements (five
  padded work vectors + the warp-dot scratch); five scalars live in static
  ``__shared__``.
* The block thread count **must be a multiple of 32** — the inner dot product
  uses a warp-shuffle reduction (``glass::high_speed::dot``).
* Seed ``x`` with an initial guess (zeros are fine); the solution is written back
  into ``x`` and the iteration count into ``iters``.

.. code-block:: cpp

   constexpr int SS = 6, KP = 32;
   size_t smem = glass::pcg_smem_size<float, SS, KP>(threads) * sizeof(float);
   pcg_kernel<<<num_problems, threads, smem>>>(d_x, d_S, d_Pinv, d_b, ...);
   // inside the kernel:
   extern __shared__ float s_mem[];
   glass::pcg<float, SS, KP>(x, S, Pinv, b, s_mem,
                                    max_iters, rel_tol, abs_tol, iters);

A cooperative *grid-wide* PCG (one solve spanning the whole grid) is future work
— see the ``glass::cgrps::grid`` backlog note referenced in
``src/base/pcg/solve.cuh``.
